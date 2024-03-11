import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import time
import os
from abc import ABC
import logging
from collections import defaultdict
from typing import List, Iterable
from rl4mixed.utils import get_inner_model


__all__ = [
    "CallbackHandler", 
    "LoggerCallback", 
    "SaveModelCallback",
    "ActorReward", 
    "ValidationReward",
    "TensorboardCallback",
    "ActorLoss", 
    "CriticLoss",
    "NeptuneCallback",
    "EarlyStopper"
]


# A logger for this file
log = logging.getLogger(__name__)



class NeptuneCallbackModule:

    def __init__(self, name, *args, **kwargs) -> None:
        self.name = name

    def get_log_info(self):
        raise NotImplementedError


class TrackGradientStatistics(NeptuneCallbackModule):
    def __init__(self, 
                 name,
                 optimizer) -> None:
        
        super().__init__(name)

        self.optimizer = optimizer

    def get_log_info(self):
            grad_vars = []
            for param in self.optimizer.state.values():
                if "exp_avg_sq" in list(param.keys()):
                    var = (param["exp_avg_sq"] - param["exp_avg"] ** 2).mean().item()
                    grad_vars.append(var)
            return np.mean(grad_vars)
    


# METRIC CONTAINER
class MetricStorage(ABC):
    path: str = NotImplemented
    def __init__(self, value) -> None:
        self.value = self._aggregate_values(value)

    def _aggregate_values(self, value):
        if isinstance(value, float) or isinstance(value, int):
            return value
        elif isinstance(value, torch.Tensor):
            if np.prod(value.size())>1:
                return torch.mean(value.detach()).item()
            else:
                return value.detach().item()
        elif isinstance(value, Iterable):
            return np.mean(value)
        else:
            raise ValueError(f"cannot handle data of type {type(value)}.")


class GradientVariance(MetricStorage):
    name = "gradient_variance"
    def __init__(self, value) -> None:
        super().__init__(value)

class ActorReward(MetricStorage):
    name = "actor_reward"
    def __init__(self, value) -> None:
        super().__init__(value)

class ValidationReward(MetricStorage):
    name = "val_reward"
    def __init__(self, value) -> None:
        super().__init__(value)

class ActorLoss(MetricStorage):
    name = "actor_loss"
    def __init__(self, value) -> None:
        super().__init__(value)

class CriticLoss(MetricStorage):
    name = "critic_loss"
    def __init__(self, value) -> None:
        super().__init__( value)


class InformationContainer:

    METRICS = [ActorReward.name, ActorLoss.name, CriticLoss.name]

    def __init__(self) -> None:
        self.information = defaultdict(list)

    def get_mean(self, attribute: str, lookback_interval=0):
        return np.mean(self.information[attribute][-lookback_interval:])
    
    def get_var(self, attribute: str, lookback_interval=0):
        return np.var(self.information[attribute][-lookback_interval:])
    
    def get_latest(self, attribute: str):
        return self.information[attribute][-1]

    def append_information(self, args: List[MetricStorage]):
        for metric in args:
            self.information[metric.name].append(metric.value)

    def list_all_information(self):
        return list(self.info.keys())
    
    def reset(self):
        self.information = defaultdict(list)


# CALLBACK CONTAINER
class Callback:
    def __init__(self, batch_step_size=None, epoch_step_size=None) -> None:
        self.batch_step_size = batch_step_size or float("inf")
        self.epoch_step_size = epoch_step_size or float("inf")
        
    def _batch_step(self, information: InformationContainer):
        pass

    def _epoch_step(self, information: InformationContainer):
        pass

    def _train_end_step(self, information):
        pass


class NeptuneCallback(Callback):
    def __init__(self, run, batch_step_size=None, epoch_step_size=None,
                 modules: List[NeptuneCallbackModule] = None) -> None:
        super().__init__(batch_step_size, epoch_step_size)
        self.batch_step_count = 0
        self.epoch = 0
        self.run = run
        self.modules = modules or []
        self.runtimes = []
        self.epoch_start = time.time()

    def _batch_step(self, information: InformationContainer):
        self.batch_step_count += self.batch_step_size
        mean_reward = information.get_mean(ActorReward.name, self.batch_step_size)
        var_loss = information.get_var(ActorLoss.name, self.batch_step_size)
        self.run["train/reward"].append(mean_reward, step=self.batch_step_count)
        self.run["train/loss_var"].append(var_loss, step=self.batch_step_count)
        for module in self.modules:
            metric = module.get_log_info()
            self.run[f"train/{module.name}"].append(metric, step=self.batch_step_count)


    def _epoch_step(self, information: InformationContainer):
        self.epoch += self.epoch_step_size
        curr_val_reward = information.get_latest(ValidationReward.name)
        self.run["validate/reward"].append(curr_val_reward, step=self.epoch)
        # track epoch duration
        epoch_runtime = time.time() - self.epoch_start
        self.runtimes.append(epoch_runtime)
        self.run["train/avg_epoch_duration"] = np.mean(self.runtimes)
        self.epoch_start = time.time()


class LoggerCallback(Callback):
    def __init__(self, total_steps, batch_step_size=None, epoch_step_size=None) -> None:
        super().__init__(batch_step_size, epoch_step_size)
        self.total_steps = total_steps
        self.start_batch = time.time()
        self.start_epoch = time.time()
        self.batch_step_count = 0
        self.epoch = 0
        

    def _batch_step(self, information: InformationContainer):
        end = time.time()
        duration = end - self.start_batch
        self.start_batch = time.time()
        self.batch_step_count += self.batch_step_size
        log.info('Epoch %s: Batch %d/%d, reward: %2.3f, loss: %2.4f, critic: %2.4f, took: %2.4fs' %
              (self.epoch, self.batch_step_count, self.total_steps, 
               information.get_mean(ActorReward.name, self.batch_step_size), 
               information.get_mean(ActorLoss.name, self.batch_step_size), 
               information.get_mean(CriticLoss.name, self.batch_step_size), 
               duration))
        
    def _epoch_step(self, information: InformationContainer):
        end = time.time()
        duration = end - self.start_epoch
        self.start_epoch = time.time()
        log.info("End of Epoch %s. Train Reward: %2.3f // Validation Reward: %2.3f; took %2.3fm" % (
            self.epoch, 
            information.get_mean(ActorReward.name, self.total_steps) ,
            information.get_latest(ValidationReward.name), 
            duration / 60))
        self.epoch += self.epoch_step_size
        self.batch_step_count = 0


class SaveModelCallback(Callback):

    def __init__(self, actor, critic, batch_step_size=None, epoch_step_size=None, model_path=None) -> None:
        if epoch_step_size is not None:
            # avoid interference of batch and epoch save callback
            batch_step_size = None
        self.actor = actor
        self.critic = critic if isinstance(critic, torch.nn.Module) else None
        self.best_lookback_reward = np.inf
        self.best_val_reward = np.inf
        self.model_path = model_path or "checkpoints"

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        super().__init__(batch_step_size, epoch_step_size)


    def _save_critic(self):
        if self.critic is not None:
            save_path = os.path.join(self.model_path, "critic.pt")
            torch.save(self.critic.state_dict(), save_path)

    def _save_actor(self, reward):
        log.info("save actor with reward %s" % reward)
        save_path = os.path.join(self.model_path, "actor.pt")
        # retrieve module in case of multiple gpus
        model = get_inner_model(self.actor)
        torch.save(model.state_dict(), save_path)

    def _batch_step(self, information):
        curr_reward = information.get_mean(ActorReward.name, self.batch_step_size)
        if curr_reward < self.best_lookback_reward:
            self.best_lookback_reward = curr_reward
            self._save_actor(curr_reward)
            self._save_critic()

    def _epoch_step(self, information):
        curr_val_reward = information.get_latest(ValidationReward.name)
        if curr_val_reward < self.best_val_reward:
            self.best_val_reward = curr_val_reward
            self._save_actor(curr_val_reward)
            self._save_critic()


def LRCallback(LRScheduler: _LRScheduler,
               optimizer: torch.optim.Optimizer,
               batch_step_size: int = None,
               epoch_step_size: int = None,
               verbose: bool = True,
               **kwargs):

    class LRCallback(Callback, LRScheduler):
        def __init__(self, 
                     optimizer, 
                     batch_step_size, 
                     epoch_step_size, 
                     verbose=False, 
                     **kwargs) -> None:
            
            Callback.__init__(self, batch_step_size, epoch_step_size)
            LRScheduler.__init__(self, optimizer=optimizer, verbose=verbose, **kwargs)

        def _batch_step(self, *_) -> None:
            LRScheduler.step(self)
        
        def _epoch_step(self, *_) -> None:
            LRScheduler.step(self)

    return LRCallback(optimizer, batch_step_size, epoch_step_size, verbose, **kwargs)




class ReduceLRonPlateau(Callback):
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 batch_step_size=None, 
                 epoch_step_size=None,
                 **kwargs) -> None:
        super().__init__(batch_step_size, epoch_step_size)

        self.scheduler = ReduceLROnPlateau(optimizer, **kwargs)

    def _batch_step(self, *_) -> None:
        ...
    
    def _epoch_step(self, information: InformationContainer) -> None:
        curr_val_reward = information.get_latest(ValidationReward.name)
        self.scheduler.step(curr_val_reward)


class CallbackHandler:

    def __init__(self, *callbacks: List[Callback]) -> None:

        self.callback_list: List[Callback] = callbacks
        self.batch_step = 0
        self.epoch_step = 0
        self.info = InformationContainer()


    def run_batch(self, *args): # actor_reward, actor_loss, critic_loss):
        self.info.append_information(args)
        self.batch_step += 1
        for callback in self.callback_list:
            if self.batch_step % callback.batch_step_size == 0:
                callback._batch_step(self.info)
        

    def run_epoch(self, *args) -> bool:
        self.batch_step = 0
        self.info.append_information(args)
        self.epoch_step += 1
        for callback in self.callback_list:
            if self.epoch_step % callback.epoch_step_size == 0:
                callback._epoch_step(self.info)

    def run_eot(self, *args):

        self.info.append_information(args)
        self.epoch_step += 1
        for callback in self.callback_list:
            if self.epoch_step % callback.epoch_step_size == 0:
                callback._epoch_step(self.info)
        

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                log.info("Early Stopping! Will terminate the Training...")
                return True
        return False