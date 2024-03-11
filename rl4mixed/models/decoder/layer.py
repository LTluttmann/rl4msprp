import torch
from torch import nn
import torch.nn.functional as F
import math
from rl4mixed.problems.vrp import StateMSVRP
from rl4mixed.settings import ModelParams
from rl4mixed.models.utils import cat_state_features, ensure_node_dim_exists, get_curr_node_embed



__all__ = [
    "HeuristicItemDecoder",
    "FlattenedItemDecoder",
    "SoftmaxItemDecoder",
    "AttentionDecoder",
    "SoftmaxDecoder"
]



class HeuristicItemDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, state: StateMSVRP):
        # shape: (bs, 1)
        curr_node = state.get_current_node()

        curr_node_supply = get_curr_node_embed(state.supply_w_depot, curr_node).squeeze(1)
        potential_amount = torch.minimum(curr_node_supply, state.remaining_load)
        potential_amount = torch.min(potential_amount, state.demand)
        probs = torch.zeros_like(potential_amount).scatter(1, potential_amount.max(dim=1)[1][:,None], 1)
        assert torch.isclose(probs.sum(1), torch.ones_like(probs[:,0])).all()

        return probs


class FlattenedItemDecoder(nn.Module):

    def __init__(self) -> None:
        super(FlattenedItemDecoder, self).__init__()

    def forward(self, _: torch.Tensor, state: StateMSVRP):
        # shape: [BS]
        node = state.get_current_node().squeeze(1)
        visit = node.ne(0) 

        node_minus1 = node[visit, None] - 1  # [BS, 1]
        # [num_visited]
        chosen_sku = state.item_ids[visit].gather(1, node_minus1).squeeze(1)
        # [BS]
        chosen_sku_and_dummy = torch.full_like(visit, -1, dtype=torch.int64)
        chosen_sku_and_dummy[visit] = chosen_sku
        # return shape: [BS]
        return chosen_sku_and_dummy


class SoftmaxItemDecoder(nn.Module):

    def __init__(self, model_params: ModelParams) -> None:
        super(SoftmaxItemDecoder, self).__init__()
        num_items = model_params.instance.num_skus
        emb_dim = model_params.embedding_dim
        cntxt_size = num_items + num_items + 1 # (supply, demand, load) 
        self.emb_to_logits = nn.Linear(emb_dim+cntxt_size, num_items+1)
        
        

    def forward(self, embeddings: torch.Tensor, state: StateMSVRP):
        # shape: (bs, 1)
        node = state.get_current_node()
        
        # shape: (bs, 1, emb)
        curr_node_embed = get_curr_node_embed(embeddings, node)
        curr_node_supply = get_curr_node_embed(state.supply_w_depot, node)

        X =  torch.cat((curr_node_embed.squeeze(1), 
                        curr_node_supply.squeeze(1),
                        state.demand, 
                        state.remaining_load), -1)
        
        mask = state.get_item_mask_from_node(node)
        logits = self.emb_to_logits(X)

        logits[mask] = -math.inf

        probs = F.softmax(logits, -1)

        return probs
    
    
class SoftmaxDecoder(nn.Module):
    "as proposed by Song et al"
    def __init__(self, model_params: ModelParams):
        """num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        device: which device to use
        """

        super(SoftmaxDecoder, self).__init__()

        self.linear = nn.Linear(model_params.embedding_dim, 1)
                  

    def forward(self, x, state: StateMSVRP):


        logits = self.linear(x).squeeze(-1)
        
        
        mask_all_comb = state.demand[:,None,:] * state.supply < 1e-5
        shelf_mask = state.get_node_mask()
        depot_mask = shelf_mask[:,[0]]
        mask = shelf_mask[:,1:,None].expand_as(mask_all_comb) + mask_all_comb
        mask = mask.flatten(1,2)
        mask = torch.cat((depot_mask, mask), 1)

        logits[mask] = - torch.inf
        probs = F.softmax(logits, 1)
        return probs


class AttentionDecoder(nn.Module):


    def __init__(self, 
                 model_params: ModelParams, 
                 context_features: list = None,
                 node_update_features: list = None,
                 other_update_features: list = None,
                 edge_update_features: list = None,
                 mask_fn=None,):
        
        super(AttentionDecoder, self).__init__()

        self.num_heads = model_params.num_heads
        embedding_dim = model_params.embedding_dim

        self.tanh_clipping = model_params.tanh_clipping
        self.mask_fn = mask_fn

        self.context_features = context_features or []
        context_dynamic_size = model_params.infer_feature_size(context_features)
        self.has_cntxt = int(context_dynamic_size > 0)
        if self.has_cntxt:
            self.cntxt_emb = nn.Linear(context_dynamic_size, embedding_dim, bias=model_params.w_context_bias)


        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq = nn.Linear((1+self.has_cntxt)*embedding_dim, embedding_dim, bias=False)
        self.Wout = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.use_graph_emb_in_decoder = model_params.use_graph_emb_in_decoder
        if self.use_graph_emb_in_decoder: 
            self.fixed_graph_embed_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)


        # define the update layers
        self.node_update_features = node_update_features
        node_dynamic_size = model_params.infer_feature_size(node_update_features) # returns 0 if None
        if node_dynamic_size > 0:
            self.project_node_step = nn.Linear(node_dynamic_size, 3 * embedding_dim, bias=False)
        else:
            self.project_node_step = None
            

        other_dynamic_size = model_params.infer_feature_size(other_update_features) # returns 0 if None
        if other_dynamic_size > 0:
            self.other_update_features = other_update_features
        else:
            self.other_update_features = None


        self.edge_update_features = edge_update_features
        edge_dynamic_size = model_params.infer_feature_size(edge_update_features) # returns 0 if None
        if edge_dynamic_size > 0 and model_params.size_agn_updates:
            self.project_edge_step = nn.Linear(edge_dynamic_size, 3, bias=False)
        else:
            self.project_edge_step = None


    @property
    def device(self):
        return next(self.parameters()).device


    def _compute_kv(self, enc_embeddings):
        # shape: (bs, nodes, emb)
        self.enc_embeddings = enc_embeddings

        # shape: (bs, 1, emb)
        if self.use_graph_emb_in_decoder:
            self.graph_embed = self.fixed_graph_embed_proj(enc_embeddings.mean(1))[:,None,:]

        # shape: (bs, head_num, n, key_dim)
        self.K = self.reshape_heads(self.Wk(enc_embeddings))
        # shape: (bs, head_num, n, key_dim)
        self.V = self.reshape_heads(self.Wv(enc_embeddings))

        self.single_head_key = self.Wk2(enc_embeddings)


    def _get_context_proj(self, shelf_emb, state):
        current_node = state.get_current_node()
        # shape: (bs, 1, emb)
        X = get_curr_node_embed(shelf_emb, current_node)
        if self.has_cntxt:
            cat_features = cat_state_features(state, self.context_features)
            cntx = self.cntxt_emb(cat_features)
            X = torch.cat((X, ensure_node_dim_exists(cntx)), -1)
        # shape: (bs, 1, emb)
        return self.Wq(X)
    

    def reshape_heads(self, qkv):
        # q.shape: (bs, num_nodes, head_num*key_dim)
        bs, n, _ = qkv.shape

        qkv_reshaped = qkv.reshape(bs, n, self.num_heads, -1)
        # shape: (bs, n, head_num, key_dim)

        qkv_transposed = qkv_reshaped.transpose(1, 2)
        # shape: (bs, head_num, n, key_dim)

        return qkv_transposed
    
    
    def _update_w_dynamic_node_vec(self, state: StateMSVRP, other_emb=None):
        bs, num_nodes, emb_dim = self.single_head_key.shape
        # (bs, nodes, 3*emb)
        updates = torch.zeros((bs, num_nodes, 3*emb_dim), device=self.single_head_key.device)

        if self.project_node_step is not None:     
            cat_features = cat_state_features(state, self.node_update_features)
            updates = updates + self.project_node_step(cat_features)

        if self.project_edge_step is not None:
            edge_feat = self.project_edge_step(cat_state_features(state, self.edge_update_features)) # bs, nodes, other_nodes, 3
            edge_upd = torch.einsum('ijkl,ikm->ijlm', edge_feat, other_emb).view(bs, num_nodes, 3*emb_dim) # bs, nodes, 3*emb
            updates = updates + edge_upd

        # (bs, nodes, emb)
        glimpse_key_upd, glimpse_val_upd, logit_key_upd = updates.chunk(3, dim=-1)

        # Projection of concatenation is equivalent to addition of projections but this is more efficient
        return (
            self.K + self.reshape_heads(glimpse_key_upd),  # shape: (bs, heads, nodes, key_dim)
            self.V + self.reshape_heads(glimpse_val_upd),  # shape: (bs, heads, nodes, key_dim)
            self.single_head_key + logit_key_upd,           # shape: (bs, nodes, emb_dim) 
        )
    
    
    def _apply_updates(self, updates):
        if updates is None:
            updates = torch.zeros_like(self.single_head_key).repeat(1,1,3)

        K_update, V_update, single_head_update = updates.chunk(3, dim=-1)

        return (
            self.K + self.reshape_heads(K_update),
            self.V + self.reshape_heads(V_update),
            self.single_head_key + single_head_update
        )
        
    def forward(self, shelf_emb, state: StateMSVRP, updates=None, other_emb=None, mask=None):
    
        # embeddings shape: (bs, num_nodes, embedding_dim)
        bs = shelf_emb.size(0)
        emb_dim = shelf_emb.size(2)
        # (bs, 1, emb)
        glimpse_Q = self._get_context_proj(shelf_emb, state)

        if self.use_graph_emb_in_decoder:
            # (bs, 1, emb)
            glimpse_Q = self.graph_embed + glimpse_Q

        if self.other_update_features is not None:
            weighted_embs = torch.einsum("ijk,ijl->ikl", cat_state_features(state, self.other_update_features), other_emb).mean(1, keepdims=True)
            glimpse_Q = weighted_embs + glimpse_Q

        # (bs, heads, 1, key_dim)
        glimpse_Q_heads = self.reshape_heads(glimpse_Q)

        # (bs, heads, nodes, key_dim) | (bs, heads, nodes, key_dim)  |  (bs, nodes, emb_dim) 
        glimpse_K, glimpse_V, logit_K = self._update_w_dynamic_node_vec(state, other_emb)

        # (bs, heads, 1, nodes)
        compatibility = torch.matmul(glimpse_Q_heads, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q_heads.size(-1))

        if mask is not None:
            compatibility[mask] = -math.inf
            
        # (bs, heads, 1, key_dim)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        # shape: (batch, 1, emb_dim)
        glimpse = self.Wout(heads.transpose(1, 2).reshape(bs, 1, emb_dim))

        if self.other_update_features is not None:  # skip connection
            glimpse = glimpse.add_(glimpse_Q).div_(2.0)

        # shape: (bs, nodes)
        logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(1) / math.sqrt(glimpse.size(-1))

        mask = self.mask_fn(state)

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        logits[mask] = -math.inf

        probs = F.softmax(logits, -1)

        assert (probs == probs).all(), "Probs should not contain any nans"

        return probs
