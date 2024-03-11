from __future__ import annotations
from dataclasses import dataclass, replace
from copy import copy,deepcopy
import torch
import logging
from typing import Callable, Union, Optional, List, Tuple, TypedDict, Dict
from rl4mixed.problems.dataset import BatchedInstances
from rl4mixed.utils import add_depot, add_item
from rl4mixed.settings import known_operators

log = logging.getLogger(__name__)
InstanceSolutionSignature = List[Tuple[int, float]]

@dataclass
class InstanceSolution:
    reward: float = None
    tour_and_units: Optional[InstanceSolutionSignature] = None
    runtime: float = None



SignatureSolveFn = Callable[[BatchedInstances, TypedDict], InstanceSolution]


class SolutionCache:

    def __init__(self, fn: SignatureSolveFn, cache=None) -> None:
        self.solve_fn = fn
        self.cache: Dict[int, InstanceSolution] = cache or dict()

    def solve_or_fetch(self, data: BatchedInstances, **kwargs) -> InstanceSolution:
        """solve a single instance from a batch if it has not been solved before and 
        fetch its solution otherwise"""

        data_hash = data.__hash__()

        try:
            solution = self.cache[data_hash]
            print("Found cached gurobi solution for problem. Retrieving it...")

        except KeyError:
            solution = self.solve_fn(data, **kwargs)
            self.cache[data_hash] = deepcopy(solution)

        return solution


def calc_reward(batch: Union[dataclass, StateMSVRP], tour_indices, beam_width=1, ret_type="tensor"):
    """
    Euclidean distance between all cities / nodes given by tour_indices
    """
    loc = batch.loc_w_depot.repeat((beam_width, 1, 1))
    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(2).expand(-1, -1, loc.size(2))
    tour = torch.gather(loc.data, 1, idx)

    # Ensure we're always returning to the depot - note the extra concat
    # won't add any extra loss, as the euclidean distance between consecutive
    # points is 0
    start = batch.depot.repeat((beam_width, 1, 1))
    y = torch.cat((start, tour, start), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    reward = tour_len.sum(1)
    
    if ret_type == "numpy":
        return reward.cpu().numpy()
    elif ret_type == "list":
        return reward.cpu().numpy().tolist()
    else:
        return reward


@dataclass
class StateMSVRP:
    # Fixed input
    loc: torch.Tensor  # Depot + loc
    depot: torch.Tensor
    demand: torch.Tensor
    supply: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    item_ids: torch.Tensor # Keeps track of original Sku
    shelf_ids: torch.Tensor # Keeps track of original shelf

    # State
    curr_node: torch.Tensor
    remaining_load: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    sku_: torch.Tensor  # Keeps track of skus that have been taken
    taken_units: torch.Tensor  
    tour_length: torch.Tensor
    curr_coord: torch.Tensor
    capacity: int
    num_nodes: int
    num_items: int
    flat: bool

    delta_supply: torch.Tensor
    delta_demand: torch.Tensor

    # VEHICLE_CAPACITY = 1.0  # Hardcoded
    max_item_slots: int = None
    _items_ohe_rand: torch.Tensor = None


    @classmethod
    def initialize(cls, input: BatchedInstances, augment=False, max_item_slots=None):

        # supply, demand, loc = cls.transform_input(input, is_supply_scalar)
        batch_size, n_loc, n_item = input.supply.size()
        assert n_loc == input.num_storage_locations or n_loc == input.num_shelves
        assert n_item == input.num_items

        loc = input.loc  # [BS, num_nodes, 2]
        depot = input.depot  # [BS, 1, 2]
        demand = input.demand  # [BS, num_items]
        supply = input.supply  # [BS, num_nodes, num_items]
        ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None]

        state = cls(
            loc=loc,
            depot=depot,
            demand=demand,
            supply=supply,
            ids=ids, 
            item_ids=input.item_ids,
            shelf_ids=input.shelf_ids,
            curr_node=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            remaining_load=torch.full((batch_size, 1), input.capacity, dtype=demand.dtype, device=loc.device),
            visited_=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            sku_=torch.full((batch_size, 1), fill_value=-1, dtype=torch.long, device=loc.device),
            taken_units=torch.zeros_like(supply, device=loc.device),
            tour_length=torch.zeros(batch_size, 1, device=loc.device),
            curr_coord=depot,  # Add step dimension
            capacity=input.capacity,
            num_nodes=n_loc,
            num_items=n_item,
            flat=input.is_flattened,
            delta_supply=torch.zeros((batch_size, n_loc+1, n_item), device=loc.device),
            delta_demand=torch.zeros_like(demand, device=loc.device),
            max_item_slots=max_item_slots,
        )

        if augment:
            state = state.augment_xy_data_by_8_fold()
        
        return state
    
    def __post_init__(self):
        self.max_item_slots = self.max_item_slots or self.num_items
    
    def repeat(self, beam_width):
        return replace(
            self,
            loc=self.loc.repeat((beam_width, 1, 1)),
            depot=self.depot.repeat((beam_width, 1, 1)),
            demand=self.demand.repeat((beam_width, 1)),
            supply=self.supply.repeat((beam_width, 1, 1)),
            ids=self.ids.repeat((beam_width, 1)),
            item_ids=self.item_ids.repeat((beam_width, 1)),
            shelf_ids=self.shelf_ids.repeat((beam_width, 1)),
            taken_units=self.taken_units.repeat((beam_width, 1, 1)),
            curr_coord=self.curr_coord.repeat((beam_width, 1, 1)),
            curr_node=self.curr_node.repeat((beam_width, 1)),
            remaining_load=self.remaining_load.repeat((beam_width, 1)),
            visited_=self.visited_.repeat((beam_width, 1)),
            sku_=self.sku_.repeat((beam_width, 1)),
            tour_length=self.tour_length.repeat((beam_width, 1)),
            delta_supply=self.delta_supply.repeat((beam_width, 1, 1)),
            delta_demand=self.delta_demand.repeat((beam_width, 1)),
        )
    
    @staticmethod
    def _augment_xy_data_by_8_fold(xy_data: torch.Tensor):
        # xy_data.shape = (batch_s, problem, 2)
        xy_data = xy_data.clone()

        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]
        # x,y shape = (batch, problem, 1)
        dat1 = torch.cat((x, y), dim=2)
        dat2 = torch.cat((1 - x, y), dim=2)
        dat3 = torch.cat((x, 1 - y), dim=2)
        dat4 = torch.cat((1 - x, 1 - y), dim=2)
        dat5 = torch.cat((y, x), dim=2)
        dat6 = torch.cat((1 - y, x), dim=2)
        dat7 = torch.cat((y, 1 - x), dim=2)
        dat8 = torch.cat((1 - y, 1 - x), dim=2)

        data_augmented = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        # shape = (8*batch, problem, 2)

        return data_augmented
    
    def augment_xy_data_by_8_fold(self):

        
        node_aug = self._augment_xy_data_by_8_fold(self.loc)
        depot_aug = self._augment_xy_data_by_8_fold(self.depot)

        new_state = self.repeat(8)
        new_state.loc = node_aug
        new_state.depot = depot_aug

        return new_state
    
    def rand_augment_xy_data(self, num_folds):
        assert num_folds <= 8 
        bs = self.loc.size(0)
        node_aug = self._augment_xy_data_by_8_fold(self.loc)
        depot_aug = self._augment_xy_data_by_8_fold(self.depot)

        # we always want to keep the original data, the other 7 are shuffled
        rand_idx = torch.cat((torch.zeros((1,)), torch.randperm(7)+1), 0).to(torch.long)[:num_folds]
        aug_idx = torch.cat((bs * rand_idx[None,:] + torch.arange(bs)[:,None]).unbind(1), 0)
        
        node_aug = node_aug[aug_idx]
        depot_aug = depot_aug[aug_idx]

        new_state = self.repeat(num_folds)
        new_state.loc = node_aug
        new_state.depot = depot_aug

        return new_state


    def clone(self, detach=False):
        batch_copy = copy(self)
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                if detach:
                    setattr(batch_copy, field, value.detach().clone())
                else:
                    setattr(batch_copy, field, value.clone())
        return batch_copy   

  
    @property
    def supply_w_depot(self):
        return add_depot(self.supply)
    

    @property
    def loc_w_depot(self):
        return torch.cat((
            self.depot,
            self.loc,
        ), dim=1)
    
    
    @property
    def items_ohe(self):
        device = self.demand.device
        bs, num_items = self.demand.shape
        return torch.eye(num_items, device=device)[None,...].expand(bs, num_items, num_items)

    @property
    def items_ohe_rand(self):
        def get_rand_ohe():
            device = self.demand.device
            bs, num_items = self.demand.shape
            # shape: (batch, max_slots, max_slots)
            ohe = torch.zeros(size=(bs, num_items, self.max_item_slots), device=device)
            
            # shape: (batch, max_slots)
            rand = torch.rand(bs, self.max_item_slots)
            rand_idx = rand.argsort(dim=1)[:, :num_items]
            # shape: (batch, nodes)
            b_idx = torch.arange(bs)[:, None].expand(bs, num_items)
            # shape: (batch, nodes)
            n_idx = torch.arange(num_items)[None, :].expand(bs, num_items)
            # shape: (batch, nodes, nodes)
            ohe[b_idx, n_idx, rand_idx] = 1
            return ohe
        
        if self._items_ohe_rand is None:
            self._items_ohe_rand = get_rand_ohe()

        return self._items_ohe_rand

    @property
    def demand_ohe(self):
        demand = self.demand.clone()[:,None,:]
        ohe_demand = torch.eye(self.num_items, device=demand.device)[None,...] * demand
        return ohe_demand
    
    @property
    def demand_rand_ohe(self):
        demand = self.demand.clone()[:,None,:]
        ohe_demand = self.items_ohe_rand * demand
        return ohe_demand
    
    @property
    def demand_column_vec(self):
        return self.demand.unsqueeze(-1)
    
    @property
    def taken_items(self):
        # [bs, num_skus]
        return self.taken_units.sum(1)
     
    @property
    def dist(self):
        return (self.loc_w_depot[:, :, None, :] - self.loc_w_depot[:, None, :, :]).norm(p=2, dim=-1)
    
    def parse_feature(self, feature: str) -> torch.Tensor:

        feat = getattr(self, feature, None)
        if feat is not None:
            return feat
        
        splits = feature.split("_")
        op_indx = [i for i, split in enumerate(splits) if split in list(known_operators.keys())]
        used_ops = [v for k, v in known_operators.items() if k in splits]
        used_ops = sorted(used_ops, key=lambda x: x[0])

        assert len(op_indx) > 0

        feat_str = "_".join(splits[:min(op_indx)])
        feat = getattr(self, feat_str).clone()

        for _, fn in used_ops:
            feat = fn(feat)

        return feat



    def __getitem__(self, key):
        # assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return replace(
            self,
            loc=self.loc[key],
            depot=self.depot[key],
            demand=self.demand[key],
            supply=self.supply[key],
            ids=self.ids[key],
            curr_node=self.curr_node[key],
            remaining_load=self.remaining_load[key],
            visited_=self.visited_[key],
            sku_=self.sku_[key],
            tour_length=self.tour_length[key],
            curr_coord=self.curr_coord[key],
            taken_units=self.taken_units[key],
            delta_demand=self.delta_demand[key],
            delta_supply=self.delta_supply[key]
        )


    def get_final_cost(self) -> torch.Tensor:

        #assert self.is_done()
        reward =  self.tour_length + (self.depot - self.curr_coord).norm(p=2, dim=-1)
        return reward.squeeze(1)


    def _update_from_shelf(self, chosen_shelf):
        """
        :param chosen_shelf: [BS]
        """
        # [BS, 1, 2]
        curr_coord = self.loc_w_depot.gather(1, chosen_shelf[:,None,None].expand(-1,1,2))

        step_dist = (curr_coord - self.curr_coord).norm(p=2, dim=2)  # (batch_dim, 1)

        self.tour_length += step_dist

        # [BS, 1]
        self.curr_node = chosen_shelf[:,None].clone()
        self.curr_coord = curr_coord
        self.visited_ = torch.hstack((self.visited_, self.curr_node))


    def _update_from_sku(self, chosen_sku):
        """
        :param chosen_sku: [BS]
        """
        # Update the dynamic elements differently for if we visit depot vs. a city
        # NOTE: Need to use the curr_node property instead of the shelves determined by the actor, since
        # during beam search different shelves might occur as beam parents for the sku child nodes
        chosen_node = self.get_current_node().squeeze(1)
        visit = chosen_node.ne(0) # [BS]
        depot = chosen_node.eq(0) # [BS]

        self.sku_ = torch.hstack((self.sku_, chosen_sku[:,None]))

        if not visit.sum() == chosen_sku.size(0):
            #assert torch.ge(chosen_sku[visit],0).all()
            chosen_sku = chosen_sku[visit]
        else:
            #assert torch.ge(chosen_sku, 0).all()
            chosen_sku = chosen_sku.clone()

        # in beam search it can happen that we have infeasible item shelf combinations selected. Usually, 
        # these get discarded in later levels of the beam tree, so simply ignore them here. We will notice 
        # infeasible solutions in _calc_log_likelihoods of the actor if they occur.
        # TODO this should not be necessary anymore! Infeasible beams "remain in the depot". Therefore, they
        # are not in the chosen_sku tensor anyway! 
        if not torch.ge(chosen_sku, 0).all():
            visit[chosen_sku.lt(0)] = False
            chosen_sku = chosen_sku[visit] 

        # [BS, 1]
        node_minus1 = chosen_node[visit, None] - 1
        # [num_visited, 1, num_items]
        selected_supply = self.supply[visit].gather(1, node_minus1[:, None].expand(-1,1,self.num_items))
        # [num_visited, num_items]
        selected_demand = self.demand[visit].gather(1, chosen_sku[:,None])

        selected_demand = torch.scatter(torch.zeros_like(selected_supply),
                                        dim=2,
                                        index=chosen_sku[:,None,None],
                                        src=selected_demand[...,None])

        # as much demand as possible
        # taken_units = torch.zeros_like(selected_demand)

        taken_units = torch.min(selected_demand, selected_supply) # [BS, 1, P]
        taken_units = torch.min(taken_units, self.remaining_load[visit,None])

        # keep track of changes
        old_supply = self.supply.clone()
        self.supply[visit] = self.supply[visit].scatter(1, 
                                                        node_minus1[:,None].expand(-1,1,self.num_items), 
                                                        selected_supply - taken_units) # .round(decimals=4)
        self.delta_supply = self.supply - old_supply

        self.remaining_load[visit] -= taken_units.sum(-1)

        # keep track of changes
        old_demand = self.demand.clone()
        self.demand[visit] -= taken_units.squeeze(1)
        self.delta_demand = self.demand - old_demand

        # update global record of units taken
        # perform a scatter_add operation
        self.taken_units[visit] = self.taken_units[visit].scatter(
            1, 
            node_minus1[:,None].expand(-1,1,self.num_items), 
            taken_units + self.taken_units[visit].gather(1, node_minus1[:,None].expand(-1,1,self.num_items))
        )

        # reset remaining load when at depot
        self.remaining_load[depot] = self.capacity


    def update(self, shelf: torch.Tensor = None, sku: torch.Tensor = None) -> StateMSVRP:
        """updates the state given the model output

        :param torch.Tensor chosen_node: tensor [BS] with index of selected nodes 
        :param torch.Tensor chosen_sku: tensor [num_visited] with index of selected sku 
        :return StateMSVRP: updated state
        """
        if shelf is not None:
            self._update_from_shelf(shelf)
        if sku is not None:
            self._update_from_sku(sku)

        return self


    def is_done(self):
        if self.visited_.size(-1) > 2000: # TODO self.num_nodes * self.demand.sum(1).max().ceil():
            log.warn("Already in iteration %s for this batch" % self.visited_.size(-1))
            action = input("What should i do ('continue', 'stop')?")
            if action == "continue":
                pass
            else:
                torch.save(self, "./state.pth")
                raise KeyboardInterrupt()
            
        return torch.logical_and(self.demand.le(1e-5).all(), self.curr_node.eq(0).all())

    def get_current_node(self):
        # return current nodes in shape [BS, 1]
        return self.curr_node.clone()
    
    def get_node_minus1(self):
        chosen_node = self.get_current_node() # [BS, 1]
        visit = chosen_node.ne(0) # [BS, 1]
        node_minus1 = chosen_node[visit, None] - 1  # [BS, 1]
        return node_minus1
    

    def get_node_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        chosen_idx = self.curr_node.clone()  
        
        if self.is_done():
            return self.demand * 0.

        # Otherwise, we can choose to go anywhere where demand is > 0
        # and demand is lower or equal to remaining capacity (no splits)
        mask_loc = torch.all(self.demand[:,None,:] * self.supply < 1e-5, 2)

        # We should avoid traveling to the depot back-to-back
        at_home = chosen_idx.eq(0)
        mask_depot = torch.where(at_home, 1, 0)
        
        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = self.remaining_load.squeeze(1).eq(0)
        # we get precision related issues if we check for exactly 0, therefore do approx check here
        has_no_demand = torch.isclose(self.demand, torch.zeros_like(self.demand), atol=1e-5).all(1) # self.demand.eq(0).all(1)

        combined = has_no_load + has_no_demand
        if combined.any():
            mask_depot[combined] = 0
            mask_loc[combined] = 1

        return torch.cat((mask_depot, mask_loc), 1).bool()
    

    def get_item_mask_from_node(self, chosen_node=None, add_dummy=True):
        """
        Gets a (batch_size, n_items+1) mask with the feasible actions depending on item supply at visited 
        shelf and item demand. 0 = feasible, 1 = infeasible
        NOTE that this returns mask of shape (bs, num_items+1) for one additional dummy item in case no 
        item is feasible (ie when at the depot). This would raise issues with softmax
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return: mask (batch_size, n_items) with 0 = feasible, 1 = infeasible
        """
        # [BS]
        # lets assume that, after selecting a shelf, we first update the state and then
        # select an item. This is much cleaner i think
        if chosen_node is None:
            chosen_node = self.get_current_node().squeeze(1)
        if len(chosen_node.shape) > 1:
            chosen_node = chosen_node.squeeze(1)
        visit = chosen_node.ne(0) # [BS]
        depot = chosen_node.eq(0) # [BS]
        # [BS, 1]
        node_minus1 = chosen_node[visit, None] - 1

        supply_at_chosen_node = self.supply[visit].gather(
            1, node_minus1[...,None].expand(-1,1,self.num_items)
        ).squeeze(1)

        supply_mask = supply_at_chosen_node.eq(0)
        demand_mask = self.demand[visit].eq(0)

        mask = torch.zeros_like(self.demand, dtype=torch.int32)
        mask[depot] = 1
        mask[visit] = torch.logical_or(supply_mask, demand_mask).int()
        if add_dummy:
            # add one dimension for a dummy item which can be chosen when at the depot
            mask = torch.cat((visit[:,None].int(), mask), dim=1)

        return mask.bool()

