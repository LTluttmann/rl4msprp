import torch
from torch import nn
from typing import Tuple
import logging
from rl4mixed.problems.dataset import BatchedInstances
from rl4mixed.problems.vrp import StateMSVRP
from rl4mixed.generator.generator import Trajectories
from rl4mixed.settings import ModelParams
from rl4mixed.utils import get_inner_model
from rl4mixed.generator.generator import *
from rl4mixed.models.decoder import *
from rl4mixed.models.encoder import *



__all__ = [
    "BaseAttnModel",
    "FlattenedAttnModel",
    "HetEncoderFlatDecoderModel",
    "HybridAttnModel",
    "MatNet",
    "HAN",
    "HeteGCN",
    "CombNet",
]


log = logging.getLogger(__name__)
LARGE_NEGATIVE_NUMBER = -1e9



class BaseAttnModel(nn.Module):

    decode_startegy_map = {
        "pomo": POMO,
        "pomo_new": POMONew,
        "greedy": Greedy,
        "sampling": Sampling,
        "beam_search": BeamSearch
    }

    def __init__(self, model_params: ModelParams):
        
        super(BaseAttnModel, self).__init__()  
        self.model_params: ModelParams = model_params
        # self.encoder = self.define_encoder(model_params)
        # self.shelf_decoder, self.item_decoder = self.define_decoder(model_params)
        self.decode_type: str = None
        self.augment = False
        self.encoder = None
        self.decoder: BaseDecoder = None


    @property
    def device(self):
        return next(self.parameters()).device


    def set_decode_type(self, decode_type: str, *args, **kwargs):
        mode = "training" if self.training else "eval"
        log.info("setting decode type to %s in %s mode" % (decode_type, mode))
        StrategyCls = self.decode_startegy_map[decode_type]
        self.decoder.generator = StrategyCls(*args, **kwargs)
        self.decode_type = decode_type

    
    def _calc_log_likelihood(self, probs: torch.Tensor, mask=None):
        # shape: (bs, seq_len) or (bs, seq_len, num_targets)
        log_p = probs.log()

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert log_p.gt(LARGE_NEGATIVE_NUMBER).data.all(), log_p # "Logprobs should not be -inf, check sampling procedure!"

        if len(log_p.shape) == 3:
            log_p = log_p.sum(2)

        # Calculate log_likelihood
        return log_p.sum(1)
    

    def forward(self, input: BatchedInstances) -> Tuple[Trajectories, torch.Tensor, StateMSVRP]:
        
        state = self.setup(input)
            
        embeddings = self.encoder(state)

        # get K and V matrices for decoder
        get_inner_model(self.decoder)._compute_kv(*embeddings)
        
        while not state.is_done():

            state = self.decoder(*embeddings, state)

        probs, sequences, state = self.decoder.generator.finalize(state)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(probs)
        return sequences, ll, state
    
    
    @staticmethod
    def _transform(input: BatchedInstances):
        return input
    
    def setup(self, input: BatchedInstances) -> StateMSVRP:
        with torch.no_grad():
            # transform input
            input = self._transform(input.clone())
            # setup state
            state = StateMSVRP.initialize(input, augment=self.augment, max_item_slots=self.model_params.max_item_slots)
            # setup decoder
            state = self.decoder.setup(state)

        return state
    
    def grad_norm(self):
        total_norm = 0
        parameters = [p for p in self.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    


class FlattenedAttnModel(BaseAttnModel):
    """attention model (AM) on homogeneous graph MSPRP"""

    def __init__(self, model_params: ModelParams):
        super(FlattenedAttnModel, self).__init__(model_params)  
        self.encoder = MHSAEncoder(
            model_params,
            node_feature_list=model_params.encoder_features,
            context_feature_list=model_params.encoder_context_features,
            with_depot=True
        )

        self.decoder = FlatDecoder(model_params)

    @staticmethod
    def _transform(input: BatchedInstances):
        return input.flatten()
    


class HetEncoderFlatDecoderModel(BaseAttnModel):
    """HAM encoder on heterogeneous graph MSPRP with flattened action space"""
    def __init__(self, model_params: ModelParams):
        super(HetEncoderFlatDecoderModel, self).__init__(model_params)  
        self.encoder = CombEncoder(model_params)
        self.linear = nn.Linear(2*model_params.embedding_dim, model_params.embedding_dim)
        self.decoder = HetFlatDecoder(model_params)


    def _combine_embeddings(self, shelf_emb, item_emb):
        bs, shelves, emb_dim = shelf_emb.shape
        skus = item_emb.size(1)
        shelf_emb = shelf_emb[:,1:,None].expand(bs, shelves-1, skus-1, emb_dim)
        item_emb = item_emb[:,None,1:].expand_as(shelf_emb)
        comb_emb = torch.cat((shelf_emb, item_emb),dim=-1).flatten(1,2) # [bs, num_shelves * num_skus, 2*emb_dim]
        depot = torch.cat((shelf_emb[:, [0]], item_emb[:, [0]]), -1) # [bs, 1, 2*emb_dim]
        comb_emb = torch.cat((depot, comb_emb), 1) # [bs, num_shelves * num_skus + 1, 2*emb_dim]
        comb_emb = self.linear(comb_emb) # [bs, num_shelves * num_skus + 1, emb_dim]
        return (comb_emb,)
    

    def forward(self, input: BatchedInstances) -> Tuple[Trajectories, torch.Tensor, StateMSVRP]:
        
        state = self.setup(input)
            
        embeddings = self.encoder(state)
        comb_embs = self._combine_embeddings(*embeddings)

        # get K and V matrices for decoder
        get_inner_model(self.decoder)._compute_kv(*comb_embs)
        
        while not state.is_done():

            state = self.decoder(*comb_embs, state)

        probs, sequences, state = self.decoder.generator.finalize(state)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(probs)
        return sequences, ll, state
    

    def setup(self, input: BatchedInstances) -> StateMSVRP:
        assert self.decode_type != "pomo", "Normal pomo not yet supported for this model. Use pomo_new"
        return super().setup(input)



class HybridAttnModel(BaseAttnModel):
    """HAM encoder with hybrid decoder: Shelf decoder is AM decoder, 
    SKU decoder is a simple heuristic
    """
    def __init__(self, model_params: ModelParams):
        super(HybridAttnModel, self).__init__(model_params)  
        self.encoder = CombEncoder(model_params)
        self.decoder = HybridDecoder(model_params)



class CombNet(BaseAttnModel):
    """HAM Encoder with hierarchical AM decoder"""

    def __init__(self, model_params: ModelParams):
        super(CombNet, self).__init__(model_params)  
        self.encoder = CombEncoder(model_params)
        self.decoder = HierarchicalAttnDecoder(model_params)


class HeteGCN(BaseAttnModel):
    """HGCN encoder with hierarchical AM decoder"""
    def __init__(self, model_params: ModelParams):
        super(HeteGCN, self).__init__(model_params)  
        self.encoder = GcnEncoder(model_params)
        self.decoder = HierarchicalAttnDecoder(model_params)
    

class HAN(BaseAttnModel):
    """HAN encoder with hierarchical AM decoder"""
    def __init__(self, model_params: ModelParams):
        super(HAN, self).__init__(model_params)  
        self.encoder = HANEncoder(model_params)
        self.decoder = HierarchicalAttnDecoder(model_params)


class MatNet(BaseAttnModel):
    """MatNet encoder with hierarchical AM decoder"""
    def __init__(self, model_params: ModelParams):
        super(MatNet, self).__init__(model_params)  
        self.encoder = MatNetEncoder(model_params)
        self.decoder = HierarchicalAttnDecoder(model_params)