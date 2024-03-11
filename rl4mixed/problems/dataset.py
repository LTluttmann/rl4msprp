import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, InitVar
from copy import copy
from typing import Optional
import pandas as pd
from rl4mixed.settings import InstanceParams
from rl4mixed.utils import infer_num_storage_locations


@dataclass
class BatchedInstances:
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    depot: torch.Tensor
    supply: torch.Tensor
    demand: torch.Tensor
    loc: torch.Tensor
    capacity: int
    normalize: InitVar[bool] = False
    item_ids: torch.Tensor = None
    shelf_ids: torch.Tensor = None
    num_items: int = None
    num_shelves: int = None
    num_storage_locations: int = None
    _normalized: bool = False
    _flattened: bool = False
    _original_capacity: int = None
    exact_obj: Optional[torch.Tensor] = None # field(init=False)
    heur_obj: Optional[torch.Tensor] = None # field(init=False)
    
    @property
    def is_batched(self):
        return len(self.loc.size()) == 3
    
    @property
    def bs(self):
        if self.is_batched:
            return self.depot.size(0)
        else:
            return 0
    
    @property
    def is_normalized(self):
        return self._normalized
    
    @property
    def is_flattened(self):
        return self._flattened
    
    @property
    def original_capacity(self):
        return self._original_capacity or self.capacity
    
    @property
    def supply_w_depot(self):
        batch_slice = slice(self.bs) if self.bs > 0 else []
        slices = batch_slice + [slice(1)]
        dim = int(self.is_batched)
        return torch.cat((
            torch.zeros_like(self.supply[slices]),
            self.supply,
        ), dim=dim)
    
    @property
    def device(self):
        return self.depot.device 
    
    @property
    def loc_w_depot(self):
        dim = int(self.is_batched)
        return torch.cat((
            self.depot,
            self.loc,
        ), dim=dim)

        
    def __post_init__(self, normalize):
        _attrs = ["item_ids", "num_items", "num_shelves", "num_storage_locations"]
        if any([getattr(self, x) is None for x in _attrs]):
            self.item_ids = self._get_item_ids()
            self.num_items = self.demand.size(-1)
            self.num_shelves = self.supply.size(1)
            self.shelf_ids = torch.arange(self.num_shelves+1).repeat(self.bs).view(self.bs, self.num_shelves+1)
            self.num_storage_locations = torch.sum(
                self.supply.flatten(1,2) > 0
            ).item() // self.bs
        if normalize:   
            self.normalize_batch()
        # self.exact_obj = [None for _ in range(len(self))] # torch.full((len(self), ), float("nan"))
        # self.heur_obj = [None for _ in range(len(self))] # torch.full((len(self), ), float("nan"))

    def normalize_batch(self):
        if self.is_normalized:
            return self
        
        self._original_capacity = self.capacity
        self.supply = self.supply.float() / self.original_capacity
        self.demand = self.demand.float() / self.original_capacity
        self.capacity = 1.0
        self._normalized = True

        return self
    
    def unnormalize_batch(self):
        if not self.is_normalized:
            return self
        
        self.supply = (self.supply * self.original_capacity).int()
        self.demand = (self.demand * self.original_capacity).int()
        self.capacity = self.original_capacity
        self._normalized = False

        return self
    
    def _get_item_ids(self):
        """This function helps us to keep track, which node corresponds to which SKU
        Each node in the graph represents a sku/shelf combination, where invalid nodes
        (those combinations that do not exist in the warehouse) are filtered out. To be 
        able to backtrack, which node holds which sku, this function generates ids
        """
        assert not self.is_flattened
        batch_size, n_shelves, n_items = self.supply.size()
        valid_nodes = self.supply.permute(0,2,1).flatten(1,2) > 0
        item_ids = torch.arange(n_items).repeat_interleave(n_shelves)
        item_ids_per_batch = item_ids.unsqueeze(0).expand(batch_size,-1)
        item_ids_per_batch = item_ids_per_batch[valid_nodes].view(batch_size,-1)
        return item_ids_per_batch
    
    def __getitem__(self, key):
        # assert torch.is_tensor(key) or isinstance(key, slice)
        instance_copy = copy(self)
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(instance_copy, field, value[key])
        return instance_copy
    
    def to(self, device):
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))
        return self
    
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
        

    def _flatten_batch(self):
        batch_size, n_shelves, n_items = self.supply.size()

        # to get indices of valid nodes, flatten the item and shelf dimensions here: [BS, P*S]
        # identify invalid nodes
        valid_nodes = self.supply.permute(0,2,1).flatten(1,2) > 0

        # create one node for each sku/shelf combination and filter out those with zero units
        # supply is represented as one-hot vector, with each entry corresponding to an sku
        # [BS, num_items*num_shelves, num_items]
        helper = torch.eye(n_items, 
                           n_items).unsqueeze(0).repeat_interleave(n_shelves, dim=1).to(self.device)
        # [BS, num_items*num_shelves, num_items]
        supply_flat_and_tiled = helper * self.supply.repeat(1, n_items, 1)
        # [BS, num_storage_locations, num_items]
        supply = supply_flat_and_tiled[valid_nodes].view(batch_size, self.num_storage_locations, n_items)

        assert torch.all(supply.nonzero()[:,2].view(-1,self.num_storage_locations) == self.item_ids)

        loc = self.loc.unsqueeze(1).expand(batch_size, n_items,-1,-1)
        loc = loc.flatten(1,2)[valid_nodes].view(batch_size, -1, 2)

        return supply, loc

    def _flatten_instance(self):
        n_shelves, n_items = self.supply.size()
        valid_nodes = self.supply.permute(1,0).flatten() > 0
        helper = torch.eye(n_items, 
                           n_items).repeat_interleave(n_shelves, dim=0).to(self.device)
        supply_flat_and_tiled = helper * self.supply.repeat(n_items, 1)
        supply = supply_flat_and_tiled[valid_nodes].view(self.num_storage_locations, n_items)

        loc = self.loc.unsqueeze(0).expand(n_items,-1,-1)
        loc = loc.flatten(0,1)[valid_nodes].view(-1, 2)

        return supply,loc

    def flatten(self):
        if self.is_flattened:
            return self.clone()
        if self.is_batched:
            supply, loc = self._flatten_batch()
        else:
            supply, loc = self._flatten_instance()

        transformed_input = self.clone()
        transformed_input.supply = supply
        transformed_input.loc = loc
        transformed_input._flattened = True
        transformed_input.shelf_ids = match_orig_shelf_idx_to_flattened(self, transformed_input)
        return transformed_input
    
    def __hash__(self):
        assert not self.is_batched

        if self.is_normalized:
            return self.clone().unnormalize_batch().__hash__()
        supply_hash = "".join([str(i) for i in self.supply_w_depot.flatten().numpy()]).strip("0")
        demand_hash = "".join([str(i+1) for i in self.demand.flatten().numpy()])
        return int(supply_hash+demand_hash)
    
    def __len__(self):
        return self.bs

        
        
def match_orig_shelf_idx_to_flattened(batch: BatchedInstances, batch_flat: BatchedInstances):
    if batch.is_batched:
        bs, num_nodes, _ = batch_flat.loc_w_depot.shape
        batch_idx, old_idx, new_idx = torch.eq(torch.cdist(batch.loc_w_depot, batch_flat.loc_w_depot), 0).nonzero(as_tuple=True)
        helper_idx = batch_idx * num_nodes + new_idx
        idx_map = torch.empty((bs * num_nodes), device=old_idx.device)
        idx_map = idx_map.scatter_(0, helper_idx, old_idx.float()).view(bs, num_nodes)
    else:
        old_idx, new_idx = old_idx, new_idx = torch.eq(batch.loc_w_depot[...,None,:], batch_flat.loc_w_depot).all(-1).nonzero(as_tuple=True)
        idx_map = torch.empty((batch_flat.loc_w_depot.size(0)), device=old_idx.device)
        idx_map = idx_map.scatter_(0, new_idx, old_idx.float())
    return idx_map


def get_flat_ids(node, sku, instance: BatchedInstances):
    assert not instance.is_flattened
    assert not instance.is_batched

    instance = instance.clone().to("cpu")
    flat_instance = instance.clone().flatten()

    # check if every node in the flattened graph represents only a single item shelf combination
    assert (pd.DataFrame(flat_instance.supply.nonzero()).groupby(0).count() == 1).all().item()
    # make sure there are no duplicates in terms of item / shelf combinations
    assert not pd.DataFrame(torch.hstack((flat_instance.supply, flat_instance.loc))).duplicated().any()

    matched_idx = match_orig_shelf_idx_to_flattened(instance, flat_instance)

    if node == 0:
        return 0
    else:
        sku_ = (flat_instance.item_ids == sku).nonzero().squeeze()
        # discard first idx to ignore depot
        assert matched_idx[0].eq(0)
        shelf_ = (matched_idx[1:] == node).nonzero().squeeze()
        flat_shelf_idx = np.intersect1d(shelf_, sku_)

        assert flat_shelf_idx.size == 1
        assert flat_instance.supply[flat_shelf_idx, sku].item() == instance.supply[node-1, sku].item()
        
        # add one to include depot again
        return flat_shelf_idx[0] + 1
    


class MixedShelvesDataset(Dataset):

    def __init__(self,
                 num_samples: int,
                 batch_size: int,
                 instance_params: InstanceParams):

        super(MixedShelvesDataset, self).__init__()

        np.random.seed(instance_params.seed)
        torch.manual_seed(instance_params.seed)

        self.num_samples = num_samples
        self.batch_size = batch_size

        self.num_skus = instance_params.num_skus
        self.num_shelves = instance_params.num_shelves
        self.size_tuple = (self.num_shelves, self.num_skus)


        self.num_storage_locations = instance_params.num_storage_locations
        print(f"Instance size is {self.num_storage_locations} physical items")

        self.max_load = instance_params.capacity[self.num_skus]
        print(f"vehicle capacity is {self.max_load}")

        self.num_nodes = self.num_shelves + 1

        self.normalize = instance_params.normalize

        # get demand params
        self.min_demand = instance_params.min_demand
        self.max_demand = instance_params.max_demand

        # calc supply params
        # -------- old -----------
        # self.lamda = instance_params.lam
        # self.min_supply = instance_params.min_supply
        # ------------------------
        self.min_supply = instance_params.min_supply
        if instance_params.max_supply is None:
            avg_demand_per_sku = (self.max_demand + self.min_demand) / 2
            avg_supply_to_demand_ratio = instance_params.avg_supply_to_demand_ratio
            avg_total_supply_per_sku = avg_supply_to_demand_ratio * avg_demand_per_sku
            avg_storgage_locations_per_sku = self.num_storage_locations / self.num_skus
            avg_supply_per_storage_loc = max(avg_total_supply_per_sku / avg_storgage_locations_per_sku, 1)
            self.max_supply = np.ceil(avg_supply_per_storage_loc * 2 - self.min_supply).astype("int")
        else:
            self.max_supply = instance_params.max_supply

    def _simulate_batch(self):
        # simulate supply [BS, P, S]
        # -------- old -----------
        # supply = torch.poisson(torch.full((self.batch_size, *self.size_tuple), self.lamda))
        # supply = torch.maximum(supply, torch.full_like(supply, self.min_supply))
        # ------------------------
        supply = torch.randint(low=self.min_supply, high=self.max_supply+1, size=(self.batch_size, *self.size_tuple)).to(torch.float32)
        # simulate demand [BS, P]; add 1 since max of randint is 'high'-1
        demand = torch.randint(low=self.min_demand, high=self.max_demand+1, size=(self.batch_size, self.num_skus))
        # simulate shelf locations as x,y coordinates in a unit circle [BS, S, 2]
        shelf_locations = torch.rand((self.batch_size, self.num_shelves, 2))

        # simulate for each batch a series of indices which correspond to the item/shelf combinations for 
        # which supply is available: [BS, PS], where PS is the number of desired physical items in the warehouse
        idx = torch.argsort(torch.rand(self.batch_size, np.prod(self.size_tuple)))[:,:self.num_storage_locations]
        # in order to select only those supply nodes which were sampled in idx, flatten the supply tensor [BS, P*S]. 
        # Select only entries from supply which were sampled in idx and use entries of zeros tensor otherwise. 
        # In the end reshape to [BS, P, S]
        supply = torch.scatter(torch.zeros(self.batch_size, np.prod(self.size_tuple)), 
                               dim=1, 
                               index=idx, 
                               src=supply.view(self.batch_size, -1)
                               ).view(self.batch_size, *self.size_tuple)
        assert torch.all((supply>0).sum((1,2)).eq(self.num_storage_locations))

        # make instance feasible by reducing demand to sum of supply for this sku
        demand = torch.minimum(demand, supply.sum(1))
        assert not demand.eq(0).all()

        depot = torch.rand((self.batch_size, 1, 2))

        data = BatchedInstances(
            depot, # [BS, 2]
            supply, # [BS, num_shelves, num_items]
            demand, # [BS, num_items]
            shelf_locations,  # [BS, num_shelves, 2]
            self.max_load,   # [BS, 1]
            normalize=self.normalize
        )
        
        return data
    
    def __len__(self):
        return self.num_samples // self.batch_size

    def __getitem__(self, _) -> BatchedInstances:
        while True:
            try:
                batch = self._simulate_batch()
                return batch
            except AssertionError:
                continue




# class MixedShelvesDataset(Dataset):

#     def __init__(self,
#                  num_samples: int,
#                  batch_size: int,
#                  instance_params: InstanceParams):

#         super(MixedShelvesDataset, self).__init__()

#         np.random.seed(instance_params.seed)
#         torch.manual_seed(instance_params.seed)
#         print(f"Using multi-instance dataset for training")
#         self.instance_params = instance_params

#         self.num_samples = num_samples
#         self.batch_size = batch_size

#         self.normalize = instance_params.normalize

#         # get demand params
#         self.min_demand = instance_params.min_demand
#         self.max_demand = instance_params.max_demand
#         self.min_supply = instance_params.min_supply

#         if not instance_params.is_multi_instance:
#             self.num_skus = self.instance_params.num_skus
#             self.num_shelves = self.instance_params.num_shelves
#             self.num_storage_locations = self.instance_params.num_storage_locations
#             self._set_instance_params()


#     def _set_instance_params(self):
#         self.size_tuple = (self.num_shelves, self.num_skus)
#         self.max_load = self.instance_params.capacity[self.num_skus]
#         if self.instance_params.max_supply is None:
#             avg_demand_per_sku = (self.max_demand + self.min_demand) / 2
#             avg_supply_to_demand_ratio = self.instance_params.avg_supply_to_demand_ratio
#             avg_total_supply_per_sku = avg_supply_to_demand_ratio * avg_demand_per_sku
#             avg_storgage_locations_per_sku = self.num_storage_locations / self.num_skus
#             avg_supply_per_storage_loc = max(avg_total_supply_per_sku / avg_storgage_locations_per_sku, 1)
#             self.max_supply = np.ceil(avg_supply_per_storage_loc * 2 - self.min_supply).astype("int")
#         else:
#             self.max_supply = self.instance_params.max_supply


#     def _set_batch_params(self):
#         assert len(self.instance_params.num_skus) == 2
#         assert len(self.instance_params.num_shelves) == 2
#         self.num_skus = np.random.randint(*self.instance_params.num_skus)
#         self.num_shelves = np.random.randint(*self.instance_params.num_shelves)
#         self.num_storage_locations = infer_num_storage_locations(self.num_skus, self.num_shelves, pi=self.instance_params.pi)
#         self._set_instance_params()



#     def _simulate_batch(self):

#         supply = torch.randint(low=self.min_supply, high=self.max_supply+1, size=(self.batch_size, *self.size_tuple)).to(torch.float32)
#         # simulate demand [BS, P]; add 1 since max of randint is 'high'-1
#         demand = torch.randint(low=self.min_demand, high=self.max_demand+1, size=(self.batch_size, self.num_skus))
#         # simulate shelf locations as x,y coordinates in a unit circle [BS, S, 2]
#         shelf_locations = torch.rand((self.batch_size, self.num_shelves, 2))

#         # simulate for each batch a series of indices which correspond to the item/shelf combinations for 
#         # which supply is available: [BS, PS], where PS is the number of desired physical items in the warehouse
#         idx = torch.argsort(torch.rand(self.batch_size, np.prod(self.size_tuple)))[:,:self.num_storage_locations]
#         # in order to select only those supply nodes which were sampled in idx, flatten the supply tensor [BS, P*S]. 
#         # Select only entries from supply which were sampled in idx and use entries of zeros tensor otherwise. 
#         # In the end reshape to [BS, P, S]
#         supply = torch.scatter(torch.zeros(self.batch_size, np.prod(self.size_tuple)), 
#                                dim=1, 
#                                index=idx, 
#                                src=supply.view(self.batch_size, -1)
#                                ).view(self.batch_size, *self.size_tuple)
#         assert torch.all((supply>0).sum((1,2)).eq(self.num_storage_locations))

#         # make instance feasible by reducing demand to sum of supply for this sku
#         demand = torch.minimum(demand, supply.sum(1))
#         assert not demand.eq(0).all()

#         depot = torch.rand((self.batch_size, 1, 2))

#         data = BatchedInstances(
#             depot, # [BS, 2]
#             supply, # [BS, num_shelves, num_items]
#             demand, # [BS, num_items]
#             shelf_locations,  # [BS, num_shelves, 2]
#             self.max_load,   # [BS, 1]
#             normalize=self.normalize
#         )
        
#         return data
    
#     def __len__(self):
#         return self.num_samples // self.batch_size

#     def __getitem__(self, _) -> BatchedInstances:
#         while True:
#             if self.instance_params.is_multi_instance:
#                 self._set_batch_params()
#             try:
#                 batch = self._simulate_batch()
#                 return batch
#             except AssertionError:
#                 continue