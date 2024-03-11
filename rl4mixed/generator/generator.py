import torch
import torch.nn as nn

import warnings
from typing import Tuple
from collections import defaultdict
from dataclasses import dataclass, fields, replace


from rl4mixed.problems.vrp import StateMSVRP

__all__ = [
    "Generator",
    "Greedy",
    "Sampling",
    "BeamSearch",
    "POMO",
    "POMONew"
]


class BeamTooLargeError(Exception):
    """Raised when the beam width is too large, leading to 
    infeasible solutions entering the final beam"""
    pass


@dataclass
class Trajectories:

    shelf: torch.Tensor = None
    sku: torch.Tensor = None

    def append_for_key(self, val, key):
        # [BS, 1]
        if len(val.shape) == 1:
            val = val[:, None]
        # [BS, curr_seq_len]
        curr_key_val = getattr(self, key)
        if curr_key_val is None:
            setattr(self, key, val)
        else:
            setattr(self, key, torch.hstack((curr_key_val, val)))


    @property
    def trajectories(self):
        traj = []
        for field in fields(self):
            val = getattr(self, field.name)
            if val is not None:
                traj.append(val)
        return traj

    def to_tensor(self):
        # [BS, seq_len, num_keys]
        combined = torch.stack(self.trajectories, dim=2)
        if combined.size(2) == 1:
            combined = combined.squeeze(2)
        return combined
    
    @classmethod
    def from_tensor(cls, vals):
        if len(vals.shape) == 2:
            return cls(shelf=vals)
        elif vals.size(2) == 1:
            return cls(shelf=vals[...,0])
        else:
            return cls(shelf=vals[..., 0], sku=vals[...,1])
    
    def flatten(self):
        return torch.flatten(self.to_tensor(), 1)

    def __len__(self):
        if self.shelf is None:
            return 0
        return self.shelfs.size(1)

    def __getitem__(self, key):
        shelf = None if self.shelf is None else self.shelf[key]
        sku = None if self.sku is None else self.sku[key]
        return replace(self,shelf=shelf,sku=sku)
    

class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.tours = Trajectories()
        self.probs = Trajectories()
        self.training = ...

    def setup_state(self, state: StateMSVRP):
        return state

    def reset(self, training: bool):
        # reset
        self.tours = Trajectories()
        self.probs = Trajectories()

        self.training = training


    def finalize(self, state) -> Tuple[torch.Tensor, Trajectories, StateMSVRP]:        
        # keys refers to the number of dimensions in the action space, e.g. 2 for shelves and items
        # [BS, steps, keys]
        probs = self.probs.to_tensor()
        tours = self.tours

        return probs, tours, state

    def _step(self, probs, state, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def _pre_step_hook(self, probs, state, key, **kwargs):
        return probs, state

    def _post_step_hook(self, selected, state, key, **kwargs):
        return selected, state
    
    def step(self, 
             probs: torch.Tensor, 
             state: StateMSVRP, 
             key: str = None, 
             **kwargs) -> Tuple[torch.Tensor, torch.Tensor, StateMSVRP]:

        probs, state = self._pre_step_hook(probs, state, key, **kwargs)        
        selected_nodes, selected_probs, state = self._step(probs, state, **kwargs)
        selected_nodes, state = self._post_step_hook(selected_nodes, state, key, **kwargs) 
        
        if key is not None:
            self.tours.append_for_key(selected_nodes, key)
            self.probs.append_for_key(selected_probs, key)

        return selected_nodes, selected_probs, state 

class Greedy(Generator):

    type = "greedy"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


    def _step(self, probs: torch.Tensor, state: StateMSVRP, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, StateMSVRP]:
        # [BS], [BS]
        probs_selected, selected = probs.max(1)

        return selected, probs_selected, state


class Sampling(Generator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


    def _step(self, probs: torch.Tensor, state: StateMSVRP, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, StateMSVRP]:
        selected = probs.multinomial(1).squeeze(1)
        probs_selected = probs.gather(1, selected[:, None]).squeeze(1)

        return selected, probs_selected, state


class POMO(Generator):

    type = "pomo"

    def __init__(self, pomo_size, *args, **kwargs) -> None:
        super().__init__()
        self.pomo_size = pomo_size

    def _setup_flat_state(self, state: StateMSVRP):
        """in POMO, we solve the same instance with different starting points. Therefore,
        to we setup the state by picking these starting points. In the flat case, where 
        item-shelf combinations are considered, we simply pick pomo_size different of these
        item-shelf combinations per instance. 

        :param StateMSVRP state: _description_
        :return _type_: _description_
        """
        # state: [BS, ...]
        batch_size = state.loc.size(0)
        num_nodes = state.loc.size(1)
        pomo_size =  min(num_nodes, self.pomo_size)
        # state: [BS * pomo, ...]
        state =  state.repeat(pomo_size)
        
        # [BS, pomo]
        _, selected = torch.topk(
            torch.rand((batch_size, num_nodes), device=state.loc.device), 
            pomo_size
        )
        # [BS * pomo]
        selected = torch.hstack(torch.unbind(selected, 1))
        # increment selected ids by one to omit depot
        selected_shelves = selected + 1
        # [BS * pomo]
        selected_items = state.item_ids.gather(1, selected[:, None]).squeeze(1)
        # [BS * pomo]
        prob = torch.ones_like(selected_shelves)

        self.tours.append_for_key(selected_shelves, "shelf")
        self.probs.append_for_key(prob, "shelf")

        state.update(shelf=selected_shelves, sku=selected_items)

        return state

    def _setup_nested_state(self, state: StateMSVRP):
        """Like above, but for a nested action space, where items and shelves are considered
        two distinct sets. 

        :param StateMSVRP state: _description_
        :return _type_: _description_
        """
        batch_size = state.loc.size(0)

        # [BS, num_valid_nodes, 3(batch, shelf and item)]
        valid_entries = torch.stack(state.supply.nonzero().split(batch_size),1)
        num_valid_nodes = valid_entries.size(1)

        pomo_size = min(num_valid_nodes, self.pomo_size)

        _, selected = torch.topk(
            torch.rand((batch_size, num_valid_nodes), device=state.loc.device), 
            pomo_size
        )
        # selected = torch.arange(start=0, end=self.pomo_size, device=state.loc.device)[None, :].expand(batch_size, self.pomo_size)

        # [BS, pomo, 3]
        selected_entries = valid_entries.gather(1, selected[..., None].expand(batch_size, -1 , 3))
        # [BS, pomo, 3] -> [BS * pomo] (get 2nd dim for shelves)
        selected_shelves = torch.hstack(selected_entries[...,1].unbind(1))
        # add one to account for depot
        selected_shelves += 1
        # [BS, pomo, 3] -> [BS * pomo] (get 3rd dim for items)
        selected_items = torch.hstack(selected_entries[...,2].unbind(1))
        # [BS * pomo]
        prob = torch.ones_like(selected_items)

        self.tours.append_for_key(selected_shelves, "shelf")
        self.probs.append_for_key(prob, "shelf")

        self.tours.append_for_key(selected_items, "sku")
        self.probs.append_for_key(prob, "sku")

        state =  state.repeat(pomo_size)
        state.update(shelf=selected_shelves, sku=selected_items)

        return state

    def setup_state(self, state: StateMSVRP):
        
        if state.flat:
            state = self._setup_flat_state(state)
        else:
            state = self._setup_nested_state(state)

        return state
    

    def _step(self,  probs: torch.Tensor, state: StateMSVRP, **kwargs):

        selected = probs.multinomial(1).squeeze(1)
        probs_selected = probs.gather(1, selected[:, None]).squeeze(1)

        return selected, probs_selected, state


class POMONew(Generator):

    type = "pomo_new"

    def __init__(self, pomo_size, *args, **kwargs) -> None:
        super().__init__()
        self.pomo_size = pomo_size
        self.aug_factor = 3 # TODO what is a good value?
        self.step_num = 0

    def reset(self, training):
        super().reset(training)
        self.step_num = 0

    def setup_state(self, state: StateMSVRP):
        return state.repeat(self.pomo_size) # state.rand_augment_xy_data(num_folds=self.aug_factor) # 


    def _step(self,  probs: torch.Tensor, state: StateMSVRP, **kwargs):

        LARGE_NUM = 1000

        aug_batch_size, num_nodes = probs.shape  # num nodes (with depot)
        batch_size = aug_batch_size // self.pomo_size
        batch_pomo_sequence = torch.arange(0, batch_size).repeat(self.pomo_size).to(probs.device)

        if self.step_num == 0:

            # [BS*POMO, num_nodes]
            logp = probs.clone().log()

            penalty_factor = torch.repeat_interleave(torch.arange(self.pomo_size) * LARGE_NUM, batch_size).to(probs.device)
            
            # we add a penalty factor to repeated instances to only use them if there
            # are not enough valid nodes in the instance
            logp_pen = logp - penalty_factor[:, None]

            logp_hstacked = torch.cat(logp_pen.split(batch_size), dim=1)
            topk_logp, topk_ind = torch.topk(logp_hstacked, self.pomo_size, dim=1, sorted=True)

            assert topk_logp.isinf().sum() == 0, "decrease pomo size"

            # [BS*POMO, 1]
            topk_ind = torch.hstack(torch.unbind(topk_ind, 1)) 

            selected = topk_ind % num_nodes  # determine node index
            
            parent = (topk_ind // num_nodes).int()
            batch_idx = batch_pomo_sequence + parent * batch_size

            probs_selected = probs[batch_idx].gather(1, selected[:,None])

            state = state[batch_idx] 

        else:

            selected = probs.multinomial(1).squeeze(1)
            probs_selected = probs.gather(1, selected[:, None]).squeeze(1)

        self.step_num += 1

        return selected, probs_selected, state


class BeamSearch(Generator):

    type = "beam_search"

    def __init__(self, beam_width, select_best=True, *args, **kwargs) -> None:
        super().__init__()
        self.beam_width = beam_width
        self.select_best = select_best
        if beam_width <= 1:
            warnings.warn("Beam width is <= 1 in Beam search. This might not be what you want")

    def reset(self, training):
        super().reset(training)
        self.step_num = 0
        self.log_beam_probs = []
        self.beam_path = []

    def setup_state(self, state: StateMSVRP):
        return state.repeat(self.beam_width)

    def _step(self, 
             probs: torch.Tensor, 
             state: StateMSVRP, 
             ignore_in_beam=False, 
             penalty=0.0,
             **kwargs):
        
        if ignore_in_beam:
            # when ignoring the respective key in the beam, we select only the MAP per beam parent. 
            # As a consequece, the beam parents for every batch instance are [0,1,2,...,BW-1]. Since
            # we have batches of beams, the indices are [0,0,0...,1,1,1,.....,BW-1,BW-1,BW-1...]
            aug_batch_size = probs.size(0) 
            batch_size = aug_batch_size // self.beam_width
            beam_parent = torch.arange(self.beam_width).repeat_interleave(batch_size)
            self.beam_path.append(beam_parent.to(probs.device))
            probs_selected, selected = probs.max(1)

        else:
            
            if self.training:
                selected, probs_selected, batch_beam_idx = self._make_stochastic_beam_step(probs)
            else:
                selected, probs_selected, batch_beam_idx = self._make_beam_step(probs, penalty)
            # first select the correct state representation according to beam parent
            state = state[batch_beam_idx] 

        self.step_num += 1

        return selected, probs_selected, state

    def _backtrack(self):

        # [BS*BW, seq_len*num_targets]
        tour_idx = self.tours.flatten()
        # [BS*BW, seq_len*num_targets]
        tour_p = self.probs.flatten()
        assert tour_idx.size(1) == len(self.beam_path), "tour idx shape and beam path shape dont match"

        num_targets = len(self.tours.trajectories)
        seq_len = tour_idx.size(1) // num_targets

        # [BS*BW]
        cur_parent = self.beam_path[-1]
        # [BS*BW]
        reversed_aligned_sequences = [tour_idx[:, -1]]
        reversed_aligned_probs = [tour_p[:, -1]]

        aug_batch_size = tour_idx.size(0)
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = torch.arange(0, batch_size).repeat(self.beam_width).to(tour_idx.device)

        for k in reversed(range(len(self.beam_path)-1)):

            batch_beam_idx = batch_beam_sequence + cur_parent * batch_size 

            reversed_aligned_sequences.append(tour_idx[batch_beam_idx, k])
            reversed_aligned_probs.append(tour_p[batch_beam_idx, k])
            cur_parent = self.beam_path[k][batch_beam_idx]

        # [BS*BW, seq_len*num_targets]
        tour_idx = torch.stack(list(reversed(reversed_aligned_sequences)), dim=1)
        tour_p = torch.stack(list(reversed(reversed_aligned_probs)), dim=1)

        # unflatten and squeeze if num_target dimension does not exists
        if num_targets > 1:
            # [BS*BW, seq_len, num_targets]
            tour_idx = tour_idx.unflatten(-1, (seq_len, num_targets))
            tour_p = tour_p.unflatten(-1, (seq_len, num_targets))

        return tour_idx, tour_p
    

    def _select_best_beam(self, probs, sequences, state: StateMSVRP):

        aug_batch_size = probs.size(0)  # num nodes (with depot)
        batch_size = aug_batch_size // self.beam_width

        costs = state.get_final_cost()
        _, idx = torch.cat(costs.unsqueeze(1).split(batch_size), 1).min(1)
        flat_idx = torch.arange(batch_size, device=costs.device) + idx * batch_size
        return probs[flat_idx], sequences[flat_idx], state[flat_idx]
    

    def finalize(self, state):
        # [BS*BW, seq_len, num_targets]
        aligned_sequences, aligned_probs = self._backtrack()
        aligned_sequences = Trajectories.from_tensor(aligned_sequences)
        if not aligned_probs.gt(-10_000).data.all():
            raise BeamTooLargeError
        assert torch.equal(aligned_sequences.shelf, state.visited_[:,1:]), "state and backtracked sequences do not match"
        if aligned_sequences.sku is not None:
            # remove dummy sku
            assert torch.equal(aligned_sequences.sku, state.sku_[:,1:]), "state and backtracked sequences do not match"
        if self.select_best and not self.training:
            return self._select_best_beam(aligned_probs, aligned_sequences, state)
        else:
            return aligned_probs, aligned_sequences, state
        

    def _fill_up_beams(self, topk_ind, topk_logp, log_beam_prob):
        """There may be cases where there are less valid options than the specified beam width. This might not be a problem at 
        the start of the beam search, since a few valid options can quickly grow to a lot more options  (if each valid option
        splits up in two more options we have 2^x growth). However, there may also be cases in small instances where simply
        too few options exist. We define these cases when every beam parent has only one valid child and the sum of valid child
        nodes is less than the beam width. In these cases we will the missing child nodes by duplicating the valid ones.
        
        Moreover, in early phases of the algorithm we may choose invalid nodes to fill the beam. We hardcode these options to
        remain in the depot. These options get filtered out in later phases of the beam search since they have a logprob of -inf

        params:
        - topk_ind
        - topk_logp
        -log_beam_prob_hat [BS, num_nodes * beam_width]
        """
        if self.step_num > 0:

            bs = topk_ind.size(0)
            # [BS, num_nodes, beam_width]
            avail_opt_per_beam = torch.stack(log_beam_prob.split(bs), -1).gt(-torch.inf).sum(1)

            invalid = torch.logical_and(avail_opt_per_beam.le(1).all(1), avail_opt_per_beam.sum(1) < self.beam_width)
            if invalid.any():
                mask = topk_logp[invalid].isinf()
                new_prob, new_ind = topk_logp[invalid].max(1)
                new_prob_exp = new_prob[:,None].expand(-1, self.beam_width)
                new_ind_exp = topk_ind[invalid, new_ind][:,None].expand(-1, self.beam_width)
                topk_logp[invalid] = torch.where(mask, new_prob_exp, topk_logp[invalid])
                topk_ind[invalid] = torch.where(mask, new_ind_exp, topk_ind[invalid])

        # infeasible beam may remain in depot. Beam will be discarded anyway in next round
        topk_ind[topk_logp.eq(-torch.inf)] = 0

        return topk_ind, topk_logp


    def _make_beam_step(self, probs: torch.Tensor, penalty=0.0):

        aug_batch_size, num_nodes = probs.shape  # num nodes (with depot)
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = torch.arange(0, batch_size).repeat(self.beam_width).to(probs.device)

        # do log transform in order to avoid that impossible actions are chosen in the beam
        # [BS*BW, num_nodes]
        logp = probs.clone().log()

        if self.step_num == 0:
            # [BS, num_nodes]
            log_beam_prob_hat = logp
            log_beam_prob_hstacked = log_beam_prob_hat[:batch_size]

            if num_nodes < self.beam_width:
                # pack some artificial nodes onto logp
                dummy = torch.full((batch_size, (self.beam_width-num_nodes)), -torch.inf, device=probs.device)
                log_beam_prob_hstacked = torch.hstack((log_beam_prob_hstacked, dummy))

            # [BS, BW]
            topk_logp, topk_ind = torch.topk(log_beam_prob_hstacked, self.beam_width, dim=1, sorted=True)

        else:
            # determine the rank of every action per beam (descending order)
            ranks = torch.argsort(torch.argsort(logp, dim=1, descending=True), dim=1)
            # use the rank as penalty so as to promote the best option per beam
            # [BS*BW, num_nodes] + [BS*BW, 1] -> [BS*BW, num_nodes]
            log_beam_prob = logp + self.log_beam_probs[-1].unsqueeze(1) 
            log_beam_prob_hat = log_beam_prob - torch.nan_to_num(penalty * ranks, posinf=torch.inf, neginf=torch.inf)
            # [BS, num_nodes * BW]
            log_beam_prob_hstacked = torch.cat(log_beam_prob_hat.split(batch_size), dim=1)
            # [BS, BW]
            # _, topk_ind = torch.topk(log_beam_prob_hstacked, self.beam_width, dim=1)
            # NOTE: for testing purposes
            topk_ind = torch.topk(log_beam_prob_hstacked, self.beam_width, dim=1, sorted=True)[1].sort(1)[0]
            # we do not want to keep track of the penalty value, therefore discard it here
            topk_logp = torch.cat(log_beam_prob.split(batch_size), dim=1).gather(1, topk_ind)

        topk_ind, topk_logp = self._fill_up_beams(topk_ind, topk_logp, log_beam_prob_hat)

        # [BS*BW, 1]
        logp_selected = torch.hstack(torch.unbind(topk_logp,1))

        # [BS*BW, 1]
        topk_ind = torch.hstack(torch.unbind(topk_ind,1)) 

        # since we stack the logprobs from the distinct branches, the indices in 
        # topk dont correspond to node indices directly and need to be translated
        selected = topk_ind % num_nodes  # determine node index

        # calc parent this branch comes from
        beam_parent = (topk_ind // num_nodes).int()

        # extract the correct representations from augmented mini-batch
        batch_beam_idx = batch_beam_sequence + beam_parent * batch_size

        # ignore logp of MAP estimates
        if penalty < torch.inf:
            self.log_beam_probs.append(logp_selected)

        self.beam_path.append(beam_parent)
        probs_selected = probs[batch_beam_idx].gather(1, selected[:,None])

        return selected, probs_selected, batch_beam_idx



    def _make_stochastic_beam_step(self, probs: torch.Tensor):

        aug_batch_size, num_nodes = probs.shape  # num nodes (with depot)
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = torch.arange(0, batch_size).repeat(self.beam_width).to(probs.device)

        # do log transform in order to avoid that impossible actions are chosen in the beam
        # [BS*BW, num_nodes]
        logp = probs.clone().log()

        if self.step_num == 0:
            # [BS, num_nodes]
            log_beam_prob = logp
            log_beam_prob_hstacked = log_beam_prob[:batch_size]
            if num_nodes < self.beam_width:
                # pack some artificial nodes onto logp
                dummy = torch.full((batch_size, (self.beam_width-num_nodes)), -torch.inf, device=probs.device)
                log_beam_prob_hstacked = torch.hstack((log_beam_prob_hstacked, dummy))

        else:
            # [BS*BW, num_nodes] + [BS*BW, 1] -> [BS*BW, num_nodes]
            log_beam_prob = logp + self.log_beam_probs[-1].unsqueeze(1)
            # [BS, num_nodes * BW]
            log_beam_prob_hstacked = torch.cat(log_beam_prob.split(batch_size), dim=1)

        # [BS, BW]
        topk_ind = torch.multinomial(log_beam_prob_hstacked.exp(), self.beam_width, replacement=False)
        topk_logp = log_beam_prob_hstacked.gather(1, topk_ind)
        topk_ind, topk_logp = self._fill_up_beams(topk_ind, topk_logp, log_beam_prob)
        # [BS*BW, 1]
        logp_selected = torch.hstack(torch.unbind(topk_logp,1))
        # [BS*BW, 1]
        topk_ind = torch.hstack(torch.unbind(topk_ind,1)) 
        # since we stack the logprobs from the distinct branches, the indices in 
        # topk dont correspond to node indices directly and need to be translated
        selected = topk_ind % num_nodes  # determine node index
        # calc parent this branch comes from
        # [BS*BW, 1]
        beam_parent = (topk_ind // num_nodes).int()
        batch_beam_idx = batch_beam_sequence + beam_parent * batch_size


        self.log_beam_probs.append(logp_selected)
        self.beam_path.append(beam_parent)
        probs_selected = probs[batch_beam_idx].gather(1, selected[:,None])

        return selected, probs_selected, batch_beam_idx
