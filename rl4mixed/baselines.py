import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
# from scipy.stats import ttest_rel
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass, fields
import inspect
import logging
from rl4mixed.models.attention_model import BaseAttnModel
from rl4mixed.problems.dataset import MixedShelvesDataset
from rl4mixed.problems.vrp import StateMSVRP
from rl4mixed.settings import *
from rl4mixed.models.critic import Critic
from rl4mixed.problems.vrp import calc_reward
from rl4mixed.utils import set_decode_type, get_inner_model

log = logging.getLogger(__name__)


def greedy_rollout(model: BaseAttnModel, dataset: MixedShelvesDataset):

    model.eval()
    set_decode_type(model, "greedy")
    
    def eval_model_bat(bat):
        with torch.no_grad():
            _, _, state = model(bat.to(model.device))
            cost = state.get_final_cost()
        return cost.mean().item()

    rewards = [
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=None, sampler=None))
    ]

    return np.mean(rewards)


def beam_search_rollout(model: BaseAttnModel,
                        beam_width: int,
                        dataset: MixedShelvesDataset) -> float:
    
    # Put in beam search evaluation mode (do select_best=False in order to not select the best 
    # beam per instance!)
    model.eval()
    set_decode_type(model, "beam_search", beam_width, select_best=False)

    def eval_model_bat(bat):
        with torch.no_grad():
            _, _, state = model(bat.to(model.device))
            costs = state.get_final_cost()
            beam_costs = torch.cat(costs.unsqueeze(1).split(dataset.batch_size), 1)
            baseline_costs, _ = torch.min(beam_costs, dim=1)

        return baseline_costs.mean().item()

    rewards = [
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=None, sampler=None))
    ]

    return np.mean(rewards)


@dataclass
class BaselineArguments:

    # args for critic baseline
    instance_params: InstanceParams
    model_params: ModelParams
    train_params: TrainingParams

    # args for rollout baseline
    actor: torch.nn.Module
    device: str

    @classmethod
    def from_dict(cls, kwargs):      
        return cls(**{
            k: v for k, v in kwargs.items() 
            if k in inspect.signature(cls).parameters
        })
    
    def asdict(self):
        result = []
        for f in fields(self):
            value = getattr(self, f.name)
            result.append((f.name, value))
        return dict(result)


class Baseline(object):
    _Mapping = {}

    @classmethod
    def register(cls):
        cls._Mapping[cls.name] = cls
    
    @classmethod
    def initialize(cls, baseline: str, *args, **kwargs):
        try:
            BL = cls._Mapping[baseline]
            return BL(*args, **kwargs)
        except KeyError:
            raise KeyError("%s not registered as baseline, only %s are registered" % 
                           (baseline, ",".join([key.__repr__() for key in cls._Mapping.keys()])))

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class WarmupBaseline(Baseline):

    def __init__(self, baseline=None, train_params: TrainingParams = None, **kwargs):
        super(WarmupBaseline, self).__init__()
        assert baseline is not None
        self.baseline = baseline
        self.warmup_baseline = ExponentialBaseline(train_params)
        self.alpha = 0
        self.n_epochs = train_params.bl_warmup_epochs
        assert self.n_epochs > 0, "n_epochs to warmup must be positive"
        log.info("using exponential warmup of the baseline for %s epochs" % self.n_epochs)

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):
        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha * lw)

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        self.alpha = min(epoch + 1, self.n_epochs) / float(self.n_epochs)
        log.info("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):

    name = None

    def __init__(self, **kwargs) -> None:
        super(NoBaseline, self).__init__()

    def eval(self, x, c):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):

    name = "exponential"

    def __init__(self, 
                 training_params: TrainingParams = None,
                 **kwargs):
        super(ExponentialBaseline, self).__init__()

        self.beta = training_params.exp_beta
        self.v = None

    def eval(self, x, c):

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']


class CriticBaseline(Baseline):

    name = "critic"

    def __init__(self,     
                 actor: BaseAttnModel = None,  
                 model_params: ModelParams = None,            
                 device="cpu",
                 **kwargs):
        
        super(CriticBaseline, self).__init__()
        self.actor = get_inner_model(actor)
        self.critic = Critic(model_params).to(device)

    def eval(self, x, c):
        state = StateMSVRP.initialize(x)
        v = self.critic(state)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v.squeeze(1), c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class RolloutBaseline(Baseline):

    name = "rollout"

    def __init__(self, 
                 instance_params: InstanceParams = None,
                 train_params: TrainingParams = None,
                 model_params: ModelParams = None,
                 device="cpu", 
                 **kwargs):
        
        super(RolloutBaseline, self).__init__()
        import rl4mixed.models as all_models
        self.eval_set_size = train_params.val_size
        self.eval_set_bs = train_params.batch_size_valid
        self.dataset_params = instance_params
        self.mean = torch.inf
        self.epoch = 0
        self.dataset = MixedShelvesDataset(self.eval_set_size, self.eval_set_bs, self.dataset_params)
        self.actor = model_params.get_model(models=all_models).to(device)


    def _update_model(self, model, epoch):
        self.actor.load_state_dict(copy.deepcopy(model.state_dict()))
        log.info("Evaluating baseline model on evaluation dataset")
        self.mean = greedy_rollout(self.actor, self.dataset)
        self.epoch = epoch

    def eval(self, x, c):
        set_decode_type(self.actor, "greedy")
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            trajectories, _, state = self.actor(x)
            v = calc_reward(state, trajectories.shelf)
            assert torch.isclose(state.get_final_cost(), v).all()
        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        log.info("Evaluating candidate model on evaluation dataset")
        candidate_mean = greedy_rollout(model, self.dataset)

        log.info("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        
        if candidate_mean - self.mean < 0:
            log.info('Update baseline')
            self._update_model(model, epoch)



class BeamSearchBaseline(Baseline):

    name = "bs_rollout"

    def __init__(self,  
                 instance_params: InstanceParams = None, 
                 model_params: ModelParams = None,
                 train_params: TrainingParams = None,
                 **kwargs):
        
        super(BeamSearchBaseline, self).__init__()
        # need kwargs for logic to work, so test here whether all params are set properly
        self.eval_set_size = train_params.val_size
        self.eval_set_bs = train_params.batch_size_valid
        self.dataset_params = instance_params
        self.beam_width = model_params.baseline_beam_width
        self.mean = torch.inf
        self.epoch = 0
        self.dataset = MixedShelvesDataset(self.eval_set_size, self.eval_set_bs, self.dataset_params)


    def _update_model(self, model, epoch):
        try:
            self.actor = copy.deepcopy(model)
        except:
            self.actor.load_state_dict(get_inner_model(model).state_dict())

        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        self.dataset = MixedShelvesDataset(self.eval_set_size, self.eval_set_bs, self.dataset_params)

        log.info("Evaluating baseline model on evaluation dataset")
        # rollout ensures self.actor is in correct decoding mode
        # get mean reward over all batches in self.dataset
        self.mean = beam_search_rollout(self.actor, self.beam_width, self.dataset) 
        self.epoch = epoch

    def eval(self, x, c):
        # assure actor is in beam_search decoding mode
        set_decode_type(self.actor, "beam_search", self.beam_width, select_best=False)
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            trajectories, _, state = self.actor(x)
            costs = calc_reward(state, trajectories.shelf)

            aug_batch_size = costs.size(0)
            batch_size = aug_batch_size // self.beam_width

            beam_costs = torch.cat(costs.unsqueeze(1).split(batch_size), 1)
            baseline_costs = torch.mean(beam_costs, dim=1)

        # There is no loss
        return baseline_costs, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        log.info("Evaluating candidate model on evaluation dataset")
        candidate_mean = beam_search_rollout(self.actor, self.beam_width, self.dataset) 

        log.info("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        
        if candidate_mean - self.mean < 0:
            log.info('Update baseline')
            self._update_model(model, epoch)



class POMOBaseline(Baseline):

    name = "pomo"

    def __init__(self, 
                 train_params: TrainingParams = None, 
                 **kwargs):
        
        self.batch_size = train_params.batch_size

    
    def eval(self, x, c):
        # c: [bs*POMO]
        aug_batch_size = c.size(0)  # num nodes (with depot)
        pomo_size = aug_batch_size // self.batch_size

        # [bs]
        bl_val = torch.cat(c.unsqueeze(1).split(self.batch_size), 1).mean(1)
        bl_val = bl_val.repeat(pomo_size)

        return bl_val, 0
    


NoBaseline.register()
ExponentialBaseline.register()
RolloutBaseline.register()
CriticBaseline.register()
BeamSearchBaseline.register()
POMOBaseline.register()