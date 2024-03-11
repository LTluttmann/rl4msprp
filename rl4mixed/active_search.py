import torch
import torch.optim as optim
import logging
import time
from rl4mixed.utils import set_decode_type
from rl4mixed.settings import TestParams
from rl4mixed.problems.dataset import BatchedInstances
from rl4mixed.problems.vrp import calc_reward
from rl4mixed.models.attention_model import BaseAttnModel


log = logging.getLogger(__name__)


def active_search(actor: BaseAttnModel, instance: BatchedInstances, test_params: TestParams):
    assert instance.bs == 1

    as_instance = instance.clone()
    # setup actor
    actor.train()
    set_decode_type(actor, "pomo", test_params.beam_width)
    # augment data
    actor.augment = test_params.augment_8_fold

    optimizer = optim.Adam(actor.parameters(), lr=test_params.active_search_lr, weight_decay=1e-6)

    for iter in range(test_params.active_search_iterations):

        trajectories, tour_logp, state = actor(as_instance)
        reward = calc_reward(state, trajectories.shelf)
        mean_reward = reward.mean(dim=None, keepdim=True)

        if iter % 20 == 0:
            log.info("Mean reward in active search iteration %s is %s" % (iter, mean_reward.squeeze().item()))
        
        loss = torch.mean((reward - mean_reward) * tour_logp)

        optimizer.zero_grad()
        loss.backward()

        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
        optimizer.step()

        iter += 1
    # revert actor state
    actor.augment = False
    actor.eval()
    return actor