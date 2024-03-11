import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import os
import logging
import neptune
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import rl4mixed.models as models
import rl4mixed.settings as opt
import rl4mixed.callbacks as callbacks
from rl4mixed.baselines import Baseline, WarmupBaseline, BaselineArguments
from rl4mixed.problems.dataset import MixedShelvesDataset
from rl4mixed.problems.vrp import calc_reward
import rl4mixed.utils as utils
from rl4mixed.validation import ValidationHandler, get_validation_score


# A logger for this file
log = logging.getLogger(__name__)



def finish_from_exception(e: Exception, nc: neptune.Run):
    log.critical(e, exc_info=True)
    nc["status/code"] = 1
    nc["status/error"] = str(e)
    nc["status/logs"].upload(HydraConfig.get().job_logging.handlers.file.filename)
    nc.stop()
    raise e


def get_device(train_params: opt.TrainingParams):
    if torch.cuda.is_available():

        if HydraConfig.get().mode.name == "MULTIRUN":
            num_gpus = torch.cuda.device_count()
            job_num = HydraConfig.get().job.num
            gpu_id = (job_num % (num_gpus-train_params.first_gpu)) + train_params.first_gpu
            device_str = f"cuda:{gpu_id}"
        else:
            if train_params.first_gpu != 0:
                device_str = f"cuda:{train_params.first_gpu}"
            else:
                device_str = "cuda"

    else:
        device_str = "cpu"
    log.info("use device %s for training" % device_str)
    return torch.device(device_str)


def hydra_get_conf_path_simple():
    hc = HydraConfig.get()
    fp = os.path.join(hc.runtime.output_dir, ".hydra")
    cn = "config.yaml"  # this seems to be always the case
    path = os.path.join(fp,cn)
    return path



def train_actor(actor: models.BaseAttnModel,
                train_loader: DataLoader,
                val_loader: DataLoader,
                baseline: Baseline,
                train_params: opt.TrainingParams,
                model_params: opt.ModelParams,
                neptune_client: neptune.Run,
                device: torch.device,
                run_dir):
    
    """Constructs the main actor & critic networks, and performs all training."""

    # if device.type == "cuda" and torch.cuda.device_count() > 1:
    #     log.info("Using %s GPUs for model training" % torch.cuda.device_count())
    #     actor = torch.nn.DataParallel(actor)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")

    optimizer = Adam(
        [{'params': actor.parameters(), 'lr': train_params.actor_lr}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': train_params.critic_lr}] 
            if len(baseline.get_learnable_parameters()) > 0 else []
        )
    )

    bs_normed_stepsize = int(len(train_loader) / 20) # logs 20 times per epoch

    lr_scheduler = callbacks.LRCallback(LambdaLR, optimizer, epoch_step_size=1, lr_lambda=lambda epoch: 0.95 ** epoch)
    lr_on_plateau = callbacks.ReduceLRonPlateau(optimizer, epoch_step_size=1, patience=3)

    save_callback = callbacks.SaveModelCallback(actor, baseline, epoch_step_size=1, model_path=checkpoint_dir)

    log_callback = callbacks.LoggerCallback(len(train_loader), batch_step_size=100, epoch_step_size=1)

    grad_stats = callbacks.TrackGradientStatistics("grad_var", optimizer)
    neptune = callbacks.NeptuneCallback(neptune_client, batch_step_size=bs_normed_stepsize, epoch_step_size=1, modules=[grad_stats])

    early_stopper = callbacks.EarlyStopper(patience=10)

    callback_handler = callbacks.CallbackHandler(
        log_callback, 
        save_callback, 
        lr_scheduler, 
        neptune,
        lr_on_plateau
    )
    
    decode_type = train_params.decode_type
    if decode_type == "beam_search":
        log.info("using stoachstic beam search during training with beam width %s" % train_params.pomo_size)
    else:
        log.info("using decode type %s during training" % decode_type)



    for epoch in range(train_params.epochs):

        actor.train()
        if epoch < train_params.bl_warmup_epochs:
            # during baseline warmup we want to use simple sampling as decoding strategy
            utils.set_decode_type(actor, "sampling", train_params.pomo_size)
        else:
            utils.set_decode_type(actor, decode_type, train_params.pomo_size)

        for batch in train_loader:

            batch = batch.to(device)

            # Full forward pass through the dataset
            trajectories, tour_logp, state = actor(batch)
            reward = calc_reward(state, trajectories.shelf)

            # Evaluate baseline, get baseline loss if any (only for critic)
            bl_val, bl_loss = baseline.eval(batch, reward)
            
            actor_loss = torch.mean((reward - bl_val) * tour_logp)
            loss = actor_loss + bl_loss

            optimizer.zero_grad()
            loss.backward()

            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], train_params.max_grad_norm)
            optimizer.step()

            callback_handler.run_batch(callbacks.ActorReward(reward),
                                       callbacks.ActorLoss(actor_loss),
                                       callbacks.CriticLoss(bl_loss))
        
        baseline.epoch_callback(actor, epoch)
        val_reward = get_validation_score(actor, val_loader, decode_type="greedy")
        callback_handler.run_epoch(callbacks.ValidationReward(val_reward))
        if early_stopper.early_stop(val_reward):             
            break

    # load best 
    if os.path.exists(path:=os.path.join(checkpoint_dir, "actor.pt")):
        model_ = utils.get_inner_model(actor)
        model_.load_state_dict(torch.load(path, device))


def backward_comp_load_state_dict(model_params: opt.ModelParams, model_path: str, device: torch.device):
    """the structure of the models has been slightly changed, with a separate 'decoder' module
    which defines shelf- and item-decoders, rather then shelf- and item-decoders being
    defined on top level. In order to load models prior to this change (and avoid retraining)
    this module makes the required changes to the state_dict to be loaded"""
    state_dict = torch.load(model_path, device)
    shelf_decoder = [key for key in state_dict.keys() if key.startswith("shelf_decoder")]
    item_decoder = [key for key in state_dict.keys() if key.startswith("item_decoder")]

    if len(shelf_decoder+item_decoder) > 0:
        model_params.w_context_bias = True

    for key in shelf_decoder+item_decoder:
        if "Wq_last" in key:  # this weight was removed quite some time ago and wasnt used ever since
            state_dict.pop(key)
        else:
            state_dict[f"decoder.{key}"] = state_dict.pop(key)
    if "decoder.shelf_decoder.project_context_step.weight" in state_dict:
        model_params.project_context_step = True
    actor = model_params.get_model(models=models).to(device)
    actor.load_state_dict(state_dict)
    return actor

def train_or_fetch_actor(
    instance_params: opt.InstanceParams,
    model_params: opt.ModelParams,
    train_params: opt.TrainingParams,
    neptune_client: neptune.Run,
    device: torch.device,
    run_dir: str
):

    if train_params.checkpoint is not None:
        # fetch
        model_path = os.path.join(train_params.checkpoint, 'actor.pt')
        # define model with given config and load weights /state
        actor = backward_comp_load_state_dict(model_params, model_path, device)
        log.info("Loaded pretrained model from %s" % model_path)

    else:
        # train
        log.info("prepare model of type %s for training..." % model_params.model_type)
        actor = model_params.get_model(models=models).to(device)
        assert train_params.train, "train model or provide pretrained model"

    if train_params.train:

        log.info("Start training pipeline...")
        # define training dataset
        train_data = MixedShelvesDataset(
            train_params.train_size, train_params.batch_size, instance_params)
        
        train_loader = DataLoader(train_data, batch_size=None, sampler=None)

        # validation dataset
        val_data = MixedShelvesDataset(
            train_params.val_size, train_params.batch_size_valid, instance_params)
        
        val_loader = DataLoader(val_data, batch_size=None, sampler=None)

        # arguments for baseline
        baseline_args = BaselineArguments(
            instance_params=instance_params,
            model_params=model_params,
            train_params=train_params,
            actor=actor,
            device=device,
        )

        log.info("initialize baseline: %s" % train_params.baseline)
        baseline = Baseline.initialize(train_params.baseline, **baseline_args.asdict())

        if train_params.bl_warmup_epochs > 0:
            log.info("...and warming it up for %s epochs" % train_params.bl_warmup_epochs)
            baseline = WarmupBaseline(baseline, train_params)

        train_actor(actor, 
                    train_loader, 
                    val_loader, 
                    baseline, 
                    train_params, 
                    model_params, 
                    neptune_client, 
                    device=device,
                    run_dir=run_dir)

    return actor


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def train_and_test(cfg: DictConfig):

    instance_params = opt.InstanceParams(**cfg.instance)
    test_instance_params = opt.InstanceParams(**getattr(getattr(cfg, "test_instance", None), "instance", None) or cfg.instance)
    model_params = opt.ModelParams(instance=instance_params, **cfg.model)
    train_params = opt.TrainingParams(**cfg.train)
    test_params = opt.TestParams(**cfg.test)

    device = get_device(train_params)

    hc = HydraConfig.get()
    run_dir = hc.runtime.output_dir
    instance_dir = run_dir.split(f"{os.sep}runs")[0]
    torch.manual_seed(instance_params.seed)
            
    if instance_params.debug:
        neptune_client = neptune.init_run(mode="debug")
    else:
        neptune_project = utils.get_neptune_project_id(instance_params)
        neptune_client = neptune.init_run(project=neptune_project, with_id=cfg.neptune_id)

    neptune_client["status/code"] = 0
    neptune_client["monitoring/device"] = device.__str__()
    neptune_client["parameters/model"] = utils.make_serializable(model_params)
    neptune_client["parameters/training"] = utils.make_serializable(train_params)
    neptune_client["parameters/test"] = utils.make_serializable(test_params)

    log.info("start training with params:")
    log.info(OmegaConf.to_yaml(utils.make_serializable(model_params)))
    log.info(OmegaConf.to_yaml(utils.make_serializable(train_params)))

    try:
        actor = train_or_fetch_actor(instance_params, model_params, train_params, neptune_client, device, run_dir)

        val_handler = ValidationHandler(test_params=test_params,
                                        instance_params=test_instance_params,
                                        actor=actor,
                                        instance_dir=instance_dir,
                                        neptune_client=neptune_client,
                                        device=device)
        
        assert not test_instance_params.is_multi_instance, "testing only with fixed instance types"
        val_handler.evaluate()

        neptune_client[f"{model_params.model_type}/actor.pt"].upload(os.path.join(os.path.join(run_dir, "checkpoints"), "actor.pt"))
        neptune_client[f"{model_params.model_type}/params.yaml"].upload(hydra_get_conf_path_simple())
        neptune_client["status/logs"].upload(hc.job_logging.handlers.file.filename)
        neptune_client.stop()

    except Exception as e:
        finish_from_exception(e, neptune_client)


def main():
    utils.check_restart_option()
    train_and_test()


if __name__ == "__main__":
    main()