import torch
from torch import nn
from typing import Optional, Tuple
from omegaconf import DictConfig
from rl4mixed.models.decoder.layer import ModelParams
from rl4mixed.problems.vrp import StateMSVRP
from rl4mixed.settings import ModelParams
from rl4mixed.generator.generator import Generator
from rl4mixed.models.decoder.layer import *


__all__ = [
    "BaseDecoder",
    "FlatDecoder",
    "HetFlatDecoder",
    "HybridDecoder",
    "HierarchicalAttnDecoder",
]

def subtract_dummy_from_item_hook(selected, state, key, **kwargs):
    if key == "sku":
        # the softmax item decoder adds an additional artificial node for depot case 
        # (no item can be selected): ignore the dummy sku here
        selected = torch.subtract(selected, torch.ones_like(selected))
    return selected, state


class BaseDecoder(nn.Module):

    def __init__(self, model_params: ModelParams) -> None:
        super().__init__()
        self.model_params = model_params
        self.generator: Generator = None
        self.shelf_decoder = None
        self.item_decoder = None


    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """function to determine the next SKU to be picked from the current shelf (:param node:)
        Logic must be implemented in Subclass 

        :param torch.Tensor embeddings: embeddings if needed for inference
        :param StateMSVRP state: state instance NOTE not updated with chosen node yet
        :param torch.Tensor node: pointer to current shelf idx
        :raises NotImplementedError: must be implemented by subclass
        :return Tuple[torch.Tensor, Optional[torch.Tensor]]: return the pointers of the SKU to be picked.
            NOTE that the shape is not [BS] but [num_visited], i.e. returns an index only for those batch 
            instances that actually visit a node but not the depot. Optionally, the logits returned by 
            the method can be returned 
        """
        raise NotImplementedError("Implement item pointer in subclass!")


    def _compute_kv(self, shelf_emb, item_emb=None):
        if hasattr(self.shelf_decoder, "_compute_kv"):
            self.shelf_decoder._compute_kv(shelf_emb)
        if item_emb is not None and hasattr(self.item_decoder, "_compute_kv"):
            self.item_decoder._compute_kv(item_emb)

    def post_generator_hook(self, selected, state, key, **kwargs):
        return selected, state


    def setup(self, state):
        assert self.generator is not None, "ensure generator is setup using set_decode_type"
        # reset generator to erase all past information stored there
        self.generator.reset(self.training)
        state = self.generator.setup_state(state)
        self.generator._post_step_hook = self.post_generator_hook
        return state



class FlatDecoder(BaseDecoder):
    """for the homogeneous graph model"""

    def __init__(self, model_params: ModelParams) -> None:
        super().__init__(model_params)

        self.shelf_decoder = AttentionDecoder(model_params, 
                                              context_features=model_params.decoder_context_features,
                                              node_update_features=model_params.decoder_dynamic_features,
                                              mask_fn=lambda state: state.get_node_mask())

        self.item_decoder = FlattenedItemDecoder()


    def forward(self, embeddings, state: StateMSVRP):
        assert state.flat, "Flat model requires a flattened state"

        shelf_probs = self.shelf_decoder(embeddings, state)

        selected_node, _, state = self.generator.step(shelf_probs, state, key="shelf")
        state = state.update(shelf=selected_node)

        selected_sku = self.item_decoder(embeddings, state)
        state = state.update(sku=selected_sku)

        return state
    
    
class HetFlatDecoder(BaseDecoder):
    """decoder for the heterogenous graph formulation, but with flattened action space at the end
    as used for example by Song et al. (2022)"""

    def __init__(self, model_params: ModelParams) -> None:
        super().__init__(model_params)

        def _mask_fn(state: StateMSVRP):
            mask_all_comb = state.demand[:,None,:] * state.supply < 1e-5
            shelf_mask = state.get_node_mask()
            depot_mask = shelf_mask[:,[0]]
            mask = shelf_mask[:,1:,None].expand_as(mask_all_comb) + mask_all_comb
            alt = torch.stack((mask_all_comb, shelf_mask[:,1:,None].expand_as(mask_all_comb)),-1).eq(True).any(-1)
            assert (mask==alt).all()
            mask = mask.flatten(1,2)
            mask = torch.cat((depot_mask, mask), 1)
            assert mask.eq(False).any(1).all()
            return mask

        self.shelf_decoder = AttentionDecoder(model_params,
                                              context_features=model_params.decoder_context_features,
                                              node_update_features=model_params.decoder_dynamic_features,
                                              mask_fn=_mask_fn)
        
        self.num_skus = model_params.instance.num_skus


    def forward(self, comb_embs, state: StateMSVRP):
        

        probs = self.shelf_decoder(comb_embs, state)

        selected_node, probs_selected, state = self.generator.step(probs, state)

        with torch.no_grad():

            node_minus_one = selected_node - 1
            shelf = torch.floor(node_minus_one / self.num_skus).long() + 1
            sku = node_minus_one % self.num_skus + 1
            sku[node_minus_one.lt(0)] = 0


        self.generator.tours.append_for_key(shelf, key="shelf")
        self.generator.probs.append_for_key(probs_selected, key="shelf")
        assert len(self.generator.tours.trajectories) == 1
        assert len(self.generator.probs.trajectories) == 1

        state = state.update(shelf=shelf, sku=sku-1)

        return state
    


class HybridDecoder(BaseDecoder):
    """hybrid model where the shelf is picked by attention decoder and the sku by a heuristic"""
    def __init__(self, model_params: ModelParams) -> None:
        super().__init__(model_params)

        self.edge_update_features = model_params.decoder_dynamic_features.get("edge", None)
        if self.edge_update_features is not None:
            self.edge_update_features_trans = [x + "_trans" for x in self.edge_update_features]
        else:
            self.edge_update_features_trans = None
    
        self.shelf_decoder = AttentionDecoder(model_params, 
                                              context_features=model_params.decoder_context_features.get("shelf", None),
                                              node_update_features=model_params.decoder_dynamic_features.get("shelf", None),
                                              other_update_features=model_params.decoder_dynamic_features.get("item", None),
                                              edge_update_features=self.edge_update_features,
                                              mask_fn=lambda state: state.get_node_mask())

        self.item_decoder = HeuristicItemDecoder()


    def forward(self, shelf_embeddings, sku_embeddings, state: StateMSVRP):

        shelf_probs = self.shelf_decoder(shelf_embeddings, state, other_emb=sku_embeddings)
        selected_shelf, _, state = self.generator.step(shelf_probs, state, key="shelf")
        state = state.update(shelf=selected_shelf)

        # sku selection with heuristic
        probs_sku = self.item_decoder(state)
        chosen_sku, _, state = self.generator.step(probs_sku, state, key="sku", ignore_in_beam=True)
        state = state.update(sku=chosen_sku)

        return state



class HierarchicalAttnDecoder(BaseDecoder):
    """the proposed hierarchical decoding strategy"""
    def __init__(self, model_params: ModelParams) -> None:
        super().__init__(model_params)

        self.edge_update_features = model_params.decoder_dynamic_features.get("edge", None)
        if self.edge_update_features is not None:
            self.edge_update_features_trans = [x + "_trans" for x in self.edge_update_features]
        else:
            self.edge_update_features_trans = None

        self.shelf_decoder = AttentionDecoder(model_params, 
                                              context_features=model_params.decoder_context_features.get("shelf", None),
                                              node_update_features=model_params.decoder_dynamic_features.get("shelf", None),
                                              other_update_features=model_params.decoder_dynamic_features.get("item", None),
                                              edge_update_features=self.edge_update_features,
                                              mask_fn=lambda state: state.get_node_mask())
        
        self.item_decoder = AttentionDecoder(model_params, 
                                             context_features=model_params.decoder_context_features.get("item", None),
                                             node_update_features=model_params.decoder_dynamic_features.get("item", None),
                                             other_update_features=model_params.decoder_dynamic_features.get("shelf", None),
                                             edge_update_features=self.edge_update_features_trans,
                                             mask_fn=lambda state: state.get_item_mask_from_node())


    def forward(self, shelf_emb, item_emb, state: StateMSVRP):
        shelf_probs = self.shelf_decoder(shelf_emb, state, other_emb=item_emb)
        selected_shelf, _, state = self.generator.step(shelf_probs, state, key="shelf")
        state = state.update(shelf=selected_shelf)

        sku_probs = self.item_decoder(shelf_emb, state, other_emb=shelf_emb)
        selected_sku, _, state = self.generator.step(sku_probs, state, key="sku", ignore_in_beam=True)
        state = state.update(sku=selected_sku)

        return state


    def post_generator_hook(self, selected, state, key, **kwargs):
        return subtract_dummy_from_item_hook(selected, state, key, **kwargs)
    