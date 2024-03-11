import torch
import torch.nn as nn
import sys
import os
from collections.abc import MutableMapping
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict, is_dataclass
from omegaconf import DictConfig, ListConfig
import neptune
import logging
import numpy as np
# A logger for this file
log = logging.getLogger(__name__)


def infer_num_storage_locations(num_skus, num_shelves, pi=None, num_storage_locations=None):
    size = (num_skus, num_shelves)
    if pi is not None and num_storage_locations is not None:
        log.info("Warning! Set both, pi and num_storage_locations. I will ignore Pi")
        assert num_storage_locations >= num_shelves
        assert num_storage_locations <= np.prod(size)
        num_storage_locations = num_storage_locations
    elif pi is None and num_storage_locations is None:
        raise ValueError("Specify either pi or num physical items")
    elif pi is not None:
        num_storage_locations = max(num_shelves, int(np.prod(size)*(1-pi)))
    else: 
        assert num_storage_locations >= num_shelves
        assert num_storage_locations <= np.prod(size)
        num_storage_locations = num_storage_locations
    return num_storage_locations


def add_depot(data):
    if len(data.shape) == 3:
        bs, _, emb_dim = data.shape
        dummy = torch.zeros((bs, 1, emb_dim), device=data.device)
        # [BS, num_items+1, emb_dim]
        emb = torch.cat((dummy, data), 1)
    elif len(data.shape) == 2:
        dummy = torch.zeros((data.size(0), 1), device=data.device)
        emb = torch.cat((dummy, data), 1)
    else:
        raise ValueError("wrong shape")
    return emb


def add_item(data):
    if len(data.shape) == 3:
        # assume shape: (bs, shelves, items)
        a=torch.zeros_like(data[:,:,:1])
        return torch.cat((a, data), 2)
    
    elif len(data.shape) == 2:
        # assume shape: (bs, items)
        dummy = torch.zeros_like(data[:, :1], device=data.device)
        # [BS, num_items+1]
        emb = torch.cat((dummy, data), 1)
    else:
        raise ValueError("wrong shape")
    return emb


def make_serializable(params: dataclass) -> Dict:
    """make a dataclass serializable (json / yaml ...)"""
    param_dict = asdict(params) if is_dataclass(params) else params
    assert isinstance(param_dict, Dict) or isinstance(param_dict, DictConfig)
    for k,v in param_dict.items():
        # Numeric values that cannot be represented as sequences of digits
        # (such as Infinity and NaN) are not permitted
        if isinstance(v, Dict) or isinstance(v, DictConfig):
            param_dict[k] = make_serializable(v)

        elif isinstance(v, RangeDict):
            param_dict[k] = "; ".join([f"{kk}: {v[kk[0]]}" for kk in v.keys()])

        elif isinstance(v, List) or isinstance(v, Tuple) or isinstance(v, ListConfig):
            if len(v) == 0:
                param_dict[k] = "-"
            elif isinstance(v[0], str):
                param_dict[k] = ", ".join(v)
            else:
                param_dict[k] = str(v)

        elif v == float("inf"):
            param_dict[k] = "inf"

        elif v == float("nan"):
            param_dict[k] = 0

        elif v is None:
            param_dict[k] = "-"

    return param_dict


def get_inner_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model



def set_decode_type(model, decode_type, *args, **kwargs):
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type, *args, **kwargs)



def get_neptune_paths_from_prefix(run, prefix):
    return [
        x for x in run._get_subpath_suggestions() if prefix in x
    ]


def get_latest_neptune_test_run_id(run) -> int:
    paths = get_neptune_paths_from_prefix(run, "test")
    latest = max([int(path.split("/")[1]) for path in paths])
    return latest
    

def get_neptune_run_ids_for_project(project: str) -> list:
    proj = neptune.init_project(project)
    run_table=proj.fetch_runs_table().to_pandas()
    if not run_table.empty:
        try:
            ids = run_table["sys/custom_run_id"].to_list()
        except KeyError:
            ids = run_table["sys/id"].to_list()
    else:
        ids = []
    return ids


def get_neptune_run_id_w_prefix(project: str, prefix: str):
    run_ids = get_neptune_run_ids_for_project(project)
    run_ids_w_prefix = [_id for _id in run_ids if prefix in _id]
    if run_ids_w_prefix:
        latest_num = max([int(_id.split("-")[-1]) for _id in run_ids_w_prefix])
        new_num = latest_num + 1
    else:
        new_num = 1
    return "%s-%s" % (prefix, str(new_num))


def get_neptune_run_id(project, model_type, baseline):
    prefix = "%s-%s" % (model_type, baseline)
    _id = get_neptune_run_id_w_prefix(project, prefix)
    return _id


def get_neptune_project_id(instance_params):
    from collections.abc import Iterable
    import os
    NEPTUNE_USER = os.getenv("NEPTUNE_USER")
    assert NEPTUNE_USER is not None, "set neptune username as environment variable"
    if isinstance(instance_params.num_shelves, Iterable):
        # multiple instance types used for training
        _id = "%s/rl4mixed-multi-instance"
    else:
        _id = "%s/rl4mixed-%ss-%si-%sn" % (
            NEPTUNE_USER,
            instance_params.num_shelves, 
            instance_params.num_skus, 
            instance_params.num_storage_locations
        )

    projects = neptune.management.get_project_list()

    if not any([proj == _id for proj in projects]):
        neptune.management.create_project(_id)

    return _id


def pairwise_distance(locations: torch.Tensor):
    """function which, given a tensor of size [L,D], with L the number of 
    locations and D the dimensionality of the coordinates vector (typically 2 for x,y)
    returns an LxL tensor with pairwise euclidean distances
    """
    num_shelves = locations.size(0)
    p = locations.unsqueeze(-1).expand(num_shelves,-1,num_shelves)
    q = locations.T.unsqueeze(0).expand_as(p)
    l2 = ((p-q)**2).sum(1).sqrt()
    return l2 # [S,S]


class NormByConstant(nn.Module):
    """torch module to apply a constant norm factor on an input sequence"""
    def __init__(self, const, static_size) -> None:
        super().__init__()
        self.static_size = static_size
        self.const = const
        
    def forward(self, x):
        x[...,self.static_size:] /= self.const
        return x


class RangeDict(MutableMapping):
    """dictionary class which allows for a number range as keys"""
    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[self.transform_key(key)]

    def __setitem__(self, key, value):
        self.store[key] = value

    # two possible behaviours
    def __delitem__(self, key):
        del self.store[self.transform_key(key)]
        # del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def transform_key(self, key):
        for k in self.keys():
            if k[0] <= key <= k[1]:
                return k
        raise KeyError(key)


def _plot_pomo_new_res(batch, trajectories, state, rewards, IDX=0):
    def _get_flat_idx(batch, trajectories):
        batch_ids = []
        for b_id, trajectory in enumerate(trajectories):
            flat_shelf_ids = []
            for t in trajectory:
                shelf = t.shelf.to("cpu")
                sku = t.sku.to("cpu")
                flat_shelf_ids.append(get_flat_ids(shelf, sku, batch[b_id]))

            batch_ids.append(flat_shelf_ids)

        return torch.Tensor(batch_ids)

    assert batch.bs == 1

    from rl4mixed.render import render_solution, get_full_tour_information
    from rl4mixed.problems.dataset import get_flat_ids
    from rl4mixed.problems.vrp import InstanceSolution

    tour_indices = _get_flat_idx(batch[[0]], trajectories[[IDX]])
    units_w_depot = torch.cat((torch.zeros_like(state.taken_units[:, :1, :]), state.taken_units), 1)
    units = units_w_depot[[IDX]].gather(1, trajectories[[IDX]].shelf[...,None].expand(1, -1, batch.num_items))
    units = (units * batch.original_capacity).int()
    tours_and_units = get_full_tour_information(tour_indices, units)[0]
    sol=InstanceSolution(rewards[[IDX]], tours_and_units, runtime=1)
    return render_solution(batch, sol, complex_legend=False)


def hydra_get_conf_path(FILEPATH):
    """save method to find the hydra config file"""
    import sys
    import os
    from hydra.core.hydra_config import HydraConfig
    # check if config path is passed through cli
    if "--config-path" in sys.argv:
        fp = sys.argv[sys.argv.index("--config-path")+1]
        fp = os.path.join(FILEPATH, fp)
        if "--config-name" in sys.argv:
            cn = sys.argv[sys.argv.index("--config-name")+1] + ".yaml"
        else:
            cn = "config.yaml"

    # otherwise retrieve config path from hydra config
    else:
        hc = HydraConfig.get()
        # try to get config from config sources specified in hydra config
        config_sources = [x["path"] for x in hc.runtime.config_sources if x["provider"] == "main"]
        if len(config_sources) == 1:
            fp = config_sources[0]
        else:
            # get config from output dir
            fp = os.path.join(hc.runtime.output_dir, ".hydra")
        cn = ".".join([HydraConfig.get().job.config_name, "yaml"])
    path = os.path.join(fp,cn)
    return path


def track_tensor_size():
    import gc
    i = 1
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                log.info("%s: Object with dict %s has size %s" % (i, obj.__dict__, obj.size()))
                i += 1
        except:
            pass


def check_restart_option():
    """function to check if a formerly generated hydra config is used for this run. In this case, 
    we assume a trained model to be in respective run directory, which will be retrieved by this 
    function. This model can either be fine tuned or evaluated on the test set"""
    from datetime import datetime
    from rl4mixed import ROOTDIR
    # in case we load a config, we must add the hydra run dir bc this is not saved within the config
    model_dir, run_dir_abs = None, None
    assert not ("--config-path" in sys.argv and "--neptune-id" in sys.argv)
    if "--config-path" in sys.argv:
        fp = sys.argv[sys.argv.index("--config-path")+1]
        if fp.split(os.sep)[-1] == ".hydra":
            # we detected a config used in a former run
            run_dir_abs = "${ROOTDIR:}/outputs/${instance.id}/runs/${model.model_type}/${now:%Y-%m-%d}/${now:%H-%M-%S}-0"
            model_dir = os.path.join(run_dir_abs, "checkpoints")

        elif os.path.isfile(os.path.join(fp, "actor.pt")):
            model_dir = fp
            run_dir_abs = "${ROOTDIR:}/outputs/${instance.id}/runs/${model.model_type}/${now:%Y-%m-%d}/${now:%H-%M-%S}-0"

    if "--neptune-id" in sys.argv:
        neptune_idx = sys.argv.index("--neptune-id")
        sys.argv.pop(neptune_idx)
        nid = sys.argv.pop(neptune_idx)
        neptune_user = os.getenv("NEPTUNE_USER")
        assert neptune_user, "set NEPTUNE_USER environment variable!"
        project = neptune_user + "/" + nid.split("/")[0]
        id_ = nid.split("/")[1]
        run = neptune.init_run(project=project, with_id=id_, mode="read-only")

        model_params = run["parameters/model"].fetch()
        inst_id = model_params["instance"]["id"]
        md_type = model_params["model_type"]

        ymd, hms = datetime.now().strftime("%Y-%m-%d %H-%M-%S").split(" ")
        outputs_dir = os.path.join(ROOTDIR, "outputs")
        run_dir_abs = os.path.join(outputs_dir, inst_id, "runs", ymd, f"{hms}-{md_type}")

        config_path = os.path.join(run_dir_abs, ".hydra")
        os.makedirs(config_path)
        model_dir = os.path.join(run_dir_abs, "checkpoints")
        os.mkdir(model_dir)
        run[f"{md_type}/params.yaml"].download(destination=os.path.join(config_path, "config.yaml"))
        run[f"{md_type}/actor.pt"].download(destination=os.path.join(model_dir, "actor.pt"))

        # sys.argv.append(f"neptune_id={id_}") # NOTE this would log into old neptune run. Is this what we want?

        sys.argv.append(f"--config-path")
        sys.argv.append(config_path)

    if model_dir is not None:
        # insert before --config-path 
        sys.argv.insert(sys.argv.index("--config-path"), f"train.checkpoint={model_dir}")

    if run_dir_abs is not None:
        sys.argv.insert(sys.argv.index("--config-path"), f"hydra.run.dir={run_dir_abs}")

    return run_dir_abs, model_dir