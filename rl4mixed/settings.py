import torch.nn as nn
from types import ModuleType
from typing import List, Optional, Union, Dict, Literal
import logging
from sys import platform
from dataclasses import dataclass, fields, field
from omegaconf import OmegaConf
from rl4mixed.utils import NormByConstant, RangeDict, add_depot, add_item, infer_num_storage_locations



log = logging.getLogger(__name__)

# instance
TASK: str = "vrp"
NUM_ITEMS: int = 10
NUM_SHELVES: int = 5
SUPPLY_SCALAR = True
MAX_DEMAND = 5
CAPACITY: RangeDict = RangeDict({
    (0,5): 6,
    (6,10): 9,
    (11,15): 12,
    (16,float("inf")): 15,
})

PI: float = 0.7
LAMBDA: float = 5.0
DEPOT_IDX: int = 0
STATIC_SIZE = 2 

# network
HIDDEN_DIM: int = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 2
DROPOUT:float = 0.1
NORM_CONST: float = 10
INPUT_NORM: nn.Module = NormByConstant(NORM_CONST, STATIC_SIZE)
CRITIC_USE_BN:bool = False

# training
DEBUG = True and platform == "darwin"
EPOCHS = 10 if not DEBUG else 2 
BEAM_WIDTH = 10
TRAIN_SAMPLES: int = 2_000_000 if not DEBUG else 1000
VAL_SAMPLES: int = 30000 if not DEBUG else 256
BATCH_SIZE: int = 128 if not DEBUG else 16
BATCH_SIZE_VALID: int = BATCH_SIZE
ACTOR_LR: float = 2e-4
CRITIC_LR: float= 1e-4

# eval
CHECKPOINT = None
TEST_ONLY = False
TEST_SAMPLES: int = VAL_SAMPLES
TEST_BATCH_SIZE: int = BATCH_SIZE_VALID
GUROBI_TIMEOUT = 120 if not DEBUG else 1
NUM_EVAL_INSTANCES = 50 if not DEBUG else 2
EVAL_WITH_GUROBI = True
GUROBI_ADD_TIMEOUT = 0

MODEL_TYPES = {
    "flat": "FlattenedAttnModel",
    "flat_het": "HetEncoderFlatDecoderModel",
    "hybrid": "HybridAttnModel",
    "matnet": "MatNet",
    "han": "HAN",
    "hete": "HeteGCN",
    "comb": "CombNet"
}

known_operators = {
    "depot": (1, add_depot),
    "dummy": (2, add_item),
    "cvec": (50, lambda x: x.unsqueeze(-1)),
    "trans": (100, lambda x: x.transpose(1,2)),
    "w": (0, lambda x: x),
    "flat": (0, lambda x: x.flatten(1,2))
}


def get_instance_id(instance_params):
    _id = "%ss_%si_%sp" % (
        instance_params.num_shelves, 
        instance_params.num_items, 
        instance_params.num_storage_locations
    )
    return _id



@dataclass
class BaseParamClass:
# class BaseParamClass(Serializable):

    def get_list_fields(self) -> List[str]:
        list_fields = []
        for f in fields(self):
            if type(getattr(self, f.name)) is list:
                list_fields.append(f.name)
        return list_fields

    def has_list_field(self) -> bool:
        return len(self.get_list_fields()) > 0
    
    def parse_list_fields(self):
        """function to avoid lists for parameters with only one specified value"""
        lfields = self.get_list_fields()
        for lfield in lfields:
            l = getattr(self, lfield)
            if len(l) == 1:
                setattr(self, lfield, l[0])

    def __post_init__(self):
        pass
        # this is not a good idea anymore. Think of model features = ["loc"], 
        # list would disappear and throw error in torch.cat()
        # self.parse_list_fields()

    @property
    def valid(self):
        return True




@dataclass
class InstanceParams(BaseParamClass):

    seed: int = 12345678
    task: str = TASK

    debug: bool = DEBUG

    num_shelves: Union[List, int] = NUM_SHELVES
    num_skus: Union[List, int] = NUM_ITEMS

    pi: float = PI
    num_physical_items: Optional[int] = None  # leave for backward compatability
    num_storage_locations: Optional[int] = None

    min_demand: int = 0
    max_demand: int = MAX_DEMAND
    min_supply: int = 1
    max_supply: int = None
    avg_supply_to_demand_ratio: float = 2.

    capacity: RangeDict = CAPACITY
    lam: float = LAMBDA
    normalize: bool = True

    id: str = None
    is_multi_instance: bool = field(init=False)

    def __post_init__(self):
        if self.num_physical_items is not None and self.num_storage_locations is None:
            self.num_storage_locations = self.num_physical_items

        if isinstance(self.num_shelves, int):
            self.num_storage_locations = infer_num_storage_locations(self.num_skus, self.num_shelves, self.pi, self.num_storage_locations)
            self.is_multi_instance = False
        else:
            self.is_multi_instance = True

        if self.max_supply is not None and self.avg_supply_to_demand_ratio is not None:
            log.info("Warning! Set both, max_supply and supply_demand_ratio. I will ignore max_supply")
            self.max_supply = None
        



@dataclass
class ModelParams(BaseParamClass):

    instance: InstanceParams

    # model features
    encoder_features: Union[List[str], Dict[str, List[str]]]
    encoder_context_features: Union[List[str], Dict[str, List[str]]] = None
    decoder_dynamic_features: Union[List[str], Dict[str, List[str]]] = None
    decoder_context_features: Union[List[str], Dict[str, List[str]]] = None
    size_agn_updates: bool = False

    # model architecture
    model_type: str = "comb"
    embedding_dim: int = HIDDEN_DIM
    qkv_dim: int = field(init=False)
    num_mhsa_layers: int = NUM_ENCODER_LAYERS # TODO refactor as num_encoder_layers
    num_heads: int = NUM_HEADS
    dropout: float = DROPOUT
    feed_forward_hidden: int = 2*HIDDEN_DIM

    normalization: Literal["batch", "instance"] = "batch"
    tanh_clipping: float = 10.

    baseline_beam_width: int = BEAM_WIDTH

    sku_rank_beam_penalty: float = 0.0
    shelf_rank_beam_penalty: float = 0.0

    use_graph_emb_in_decoder:bool = False
    use_q_as_v: bool = True
    ms_hidden_dim: int = 0
    max_item_slots: int = None

    w_context_bias: bool = False
    project_context_step: bool = False


    def __post_init__(self):
        super().__post_init__()
        self.qkv_dim = self.embedding_dim // self.num_heads
        if self.max_item_slots is not None:

            self.max_item_slots = self.max_item_slots 
        else:
            self.max_item_slots = self.instance.num_skus if isinstance(self.instance.num_skus, int) else max(self.instance.num_skus)



    def infer_feature_size(self, features:list=None):

        feature_size_mapping = {
            "demand": (self.instance.num_skus,),
            "remaining_load": (1,),
            "loc": (2,),
            "supply": (self.num_nodes, self.instance.num_skus),
            "items_ohe": (self.instance.num_skus,),
            "demand_ohe": (self.instance.num_skus,),
            "items_ohe_rand": (self.max_item_slots,),
            "demand_rand_ohe": (self.max_item_slots,),
            "demand_column_vec": (1,), # NOTE: leave for backwards compatability
            "taken_items": (1,),
            "taken_units": (self.num_nodes, self.instance.num_skus),
        }

        def parse_feat_size(feature):
            import torch

            size = feature_size_mapping.get(feature, None)
            if size is not None:
                return size[-1]
            
            splits = feature.split("_")
            op_indx = [i for i, split in enumerate(splits) if split in list(known_operators.keys())]
            used_ops = [v for k, v in known_operators.items() if k in splits]
            used_ops = sorted(used_ops, key=lambda x: x[0])

            feat_str = "_".join(splits[:min(op_indx)])
            size = feature_size_mapping.get(feat_str, None)

            feat = torch.empty((1, *size))
            for _, fn in used_ops:
                feat = fn(feat)

            return feat.size(-1)
        
        def parse_sizeagn_feat(feature):
            import torch
            size = feature_size_mapping.get(feature, None)
            if size is not None:
                return size[-1]
            else:
                splits = feature.split("_")
                assert "cvec" in splits, "in multi instance training, only size agnostic models are allowed"
                return 1

        if features is None:
            return 0
        
        ds = 0
        for feat in features:
            if self.instance.is_multi_instance:
                size = parse_sizeagn_feat(feat)
            else:
                size = parse_feat_size(feat)
            ds += size
        
        return ds

    @property
    def num_nodes(self):
        return self.instance.num_storage_locations if self.model_type == "flat" else self.instance.num_shelves    
    
    def get_model(self, models: ModuleType):
        return getattr(models, MODEL_TYPES[self.model_type])(self)
    


@dataclass
class TrainingParams(BaseParamClass):
    train: bool = True
    checkpoint: Optional[str] = CHECKPOINT
    actor_lr: float = ACTOR_LR
    critic_lr: float = CRITIC_LR
    max_grad_norm: float = 1.
    batch_size: int = BATCH_SIZE
    batch_size_valid: int = BATCH_SIZE_VALID
    bl_warmup_epochs: int = 1
    baseline: Literal["critic", "rollout", "exponential", "bs_rollout", "pomo", "pomo_new"] = "rollout"
    epochs: int = EPOCHS
    train_size: int = TRAIN_SAMPLES
    val_size: int = VAL_SAMPLES
    exp_beta: float = 0.8
    pomo_size: int = BEAM_WIDTH
    decode_type: Literal["beam_search", "sampling", "pomo"] = "sampling"
    num_gpus: int = 8
    first_gpu: int = 0

    def __post_init__(self):
        self.batch_size_valid = self.batch_size
        super().__post_init__()

        if self.decode_type in ["beam_search", "pomo", "pomo_new"]:
            assert self.baseline == "pomo", "You use %s as decoder but not the POMO baseline" % self.decode_type
            assert self.pomo_size > 1, "must set a POMO size larger than 1 if using %s decoder" % self.decode_type

        if self.decode_type == "sampling":
            assert not self.baseline == "pomo", "POMO baseline does not make sense when using sampling decoder"

        if self.baseline in ["rollout", "bs_rollout"]:
            assert self.bl_warmup_epochs > 0, "Warm up the policy when using rollouts as baseline"


@dataclass
class TestParams(BaseParamClass):
    batch_size_test: int = TEST_BATCH_SIZE
    beam_width: int = BEAM_WIDTH
    test_size: int = TEST_SAMPLES
    decode_type: Literal["beam_search", "greedy", "sampling"] = "greedy"
    gurobi_timeout: int = GUROBI_TIMEOUT
    n_exact_instances: int = NUM_EVAL_INSTANCES
    solve_exact: bool = EVAL_WITH_GUROBI
    render: bool = True
    dataset_path: str = None
    no_sol_add_timeout: int = GUROBI_ADD_TIMEOUT
    active_search_iterations: int = 0
    augment_8_fold: bool = True
    active_search_lr: float = 5e-6
    active_search_runtime: int = 0
    comp_beam_width: int = 100 # can be much larger since we evaluate only a single instance

    def __post_init__(self):
        super().__post_init__()
        if self.beam_width > 1 and self.decode_type != "beam_search":
            log.warn("you set a beam width but decode types are %s" % self.decode_type)
        

if __name__ == "__main__":
    a = TestParams()
    a.dataset_path = ""
    log.info(OmegaConf.to_yaml(a))

