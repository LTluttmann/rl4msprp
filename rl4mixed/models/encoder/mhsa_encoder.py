import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from rl4mixed.problems.vrp import StateMSVRP
from rl4mixed.settings import ModelParams
from rl4mixed.models.encoder.graph_encoder import AttentionEncoderBlock, reshape_by_heads
from rl4mixed.models.utils import cat_state_features


class MultiHeadAttention(nn.Module):
    def __init__(self, model_params: ModelParams):
        
        super(MultiHeadAttention, self).__init__()


        self.num_heads = model_params.num_heads
        self.embed_dim = model_params.embedding_dim
        self.qkv_dim = model_params.qkv_dim

        self.norm_factor = 1 / math.sqrt(self.qkv_dim)  # See Attention is all you need

        self.Wq = nn.Linear(self.embed_dim, self.num_heads * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embed_dim, self.num_heads * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embed_dim, self.num_heads * self.qkv_dim, bias=False)



    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.embed_dim, "Wrong embedding dimension of input"

        qflat = q.contiguous().view(-1, input_dim)
        h_flat = h.contiguous().view(-1, input_dim)

        # q shape: (batch, head_num, row_cnt, qkv_dim)
        Q = reshape_by_heads(self.Wq(q), head_num=self.num_heads)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)
        K = reshape_by_heads(self.Wk(h), head_num=self.num_heads)
        V = reshape_by_heads(self.Wv(h), head_num=self.num_heads)

        # shp = (batch_size, self.num_heads , graph_size, -1)
        # shp_q = (batch_size, self.num_heads, n_query, -1)
        # # Calculate queries, (num_heads, n_query, graph_size, key/val_size)
        # Q = torch.matmul(qflat, self.Wq).view(shp_q)
        # # Calculate keys and values (num_heads, batch_size, graph_size, key/val_size)
        # K = torch.matmul(h_flat, self.Wk).view(shp)
        # V = torch.matmul(h_flat, self.Wv).view(shp)

        # Calculate compatibility (num_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(batch_size, 1, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -math.inf

        attn = torch.softmax(compatibility, dim=-1)

        # shape: (batch, head_num, nodes, qkv_dim)
        heads = torch.matmul(attn, V)

        # shape: (batch, nodes, head_num, qkv_dim)
        heads_transposed = heads.transpose(1, 2)
        # shape: (batch, nodes, head_num*qkv_dim)
        heads_concat = heads_transposed.reshape(batch_size, graph_size, self.num_heads * self.qkv_dim)
        
        return heads_concat
    

class Encoder(nn.Module):

    def __init__(self, 
                 model_params: ModelParams, 
                 node_feature_list: List[str] = None,
                 context_feature_list: List[str] = None,
                 with_depot: bool = False) -> None:
        
        super().__init__()

        embedding_dim = model_params.embedding_dim
        # Special embedding projection for depot node        
        self.init_depot = nn.Linear(2, embedding_dim)
        self.node_features = node_feature_list or model_params.encoder_features
        node_dim = model_params.infer_feature_size(self.node_features) # x, y, task specific node information
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.layers = nn.ModuleList([
            AttentionEncoderBlock(model_params, MultiHeadAttention) 
            for _ in range(model_params.num_mhsa_layers)
        ])

        self.context_features = context_feature_list or model_params.encoder_context_features

        if len(self.context_features) > 0:
            self.context_encoder = nn.Linear(
                model_params.infer_feature_size(self.context_features), embedding_dim
            )
            self.node_and_cntxt_proj = nn.Linear(2*embedding_dim, embedding_dim)
    
        if with_depot:
            self.init_depot = nn.Linear(2, model_params.embedding_dim)


    def forward(self, state: StateMSVRP):
        node_emb = self._init_node_embed(state)

        for layer in self.layers:
            node_emb = layer(node_emb)

        if len(self.context_features) > 0:
            cntxt_embeds = self._context_embed(state)

            bs, n_nodes, emd_dim = node_emb.shape
            node_and_cntxt = torch.cat((node_emb, cntxt_embeds[:,None,:].expand(bs, n_nodes, emd_dim)), 2)

            node_emb = self.node_and_cntxt_proj(node_and_cntxt)

        return (node_emb,)


    def _init_node_embed(self, state: StateMSVRP):

        features = cat_state_features(state, self.node_features)
        embeddings = self.init_embed(features)
        if hasattr(self, "init_depot"):
            embeddings = torch.cat((self.init_depot(state.depot), embeddings), 1)

        return embeddings

    def _context_embed(self, state: StateMSVRP):

        features = cat_state_features(state, self.context_features)
        embeddings = self.context_encoder(features)
        return embeddings
