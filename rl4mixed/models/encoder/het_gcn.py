import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
from rl4mixed.problems.vrp import StateMSVRP
from rl4mixed.settings import ModelParams
from rl4mixed.models.encoder.graph_encoder import AddAndNormalization
from rl4mixed.models.utils import cat_state_features, add_dummy_item


def row_normalize_adjacency(adjacency_matrix):
    # Calculate the row sums (out-degrees for each node)
    row_sums = torch.sum(adjacency_matrix, dim=2, keepdim=False)

    # Handle potential division by zero for nodes with no outgoing edges
    row_sums[row_sums == 0] = 1.0

    diag = torch.diag_embed(row_sums.squeeze(-1), offset=0, dim1=-2, dim2=-1)

    diag_inv = torch.inverse(diag)

    normalized_adjacency = diag_inv @ adjacency_matrix

    return normalized_adjacency


class HetGCNLayer(nn.Module):
    def __init__(self, model_params: ModelParams):
        super().__init__()
        # self.W_b = nn.Linear(model_params.embedding_dim, model_params.embedding_dim)
        # self.W_a = nn.Linear(model_params.embedding_dim, model_params.embedding_dim)

        self.W_q = nn.Linear(model_params.embedding_dim, model_params.embedding_dim, bias=False)
        self.W_k = nn.Linear(model_params.embedding_dim, model_params.embedding_dim, bias=False)
        self.W_v = nn.Linear(2*model_params.embedding_dim, 1, bias=False)

        # self.emb = nn.Linear(2*model_params.embedding_dim, model_params.embedding_dim)
        self.skipcon_and_norm = AddAndNormalization(model_params)
        

    def forward(self, x_a, x_b, adj):
        # x_a shape: (bs, nodes_a, emb)
        # x_b shape: (bs, nodes_b, emb)
        # adj shape: (bs, nodes_a, nodes_b)
        adj_norm = row_normalize_adjacency(adj)
        # [BS, col_size, emb]
        # x_b_out = self.W_b(x_b) 
        # [BS, row_size, emb]
        compat = torch.matmul(adj_norm, x_b)

        Q = self.W_q(x_a)
        K_self = self.W_k(x_a)
        K_neigh = self.W_k(compat)

        # [BS, row_size, 1]
        compat_self = F.elu(self.W_v(torch.cat((K_self, Q), -1)))
        # [BS, row_size, 1]
        compat_neigh = F.elu(self.W_v(torch.cat((K_neigh, Q), -1)))
        # [BS, row_size, 2]
        weights = F.softmax(torch.cat((compat_self, compat_neigh), -1), -1)
        # [BS, row_size, 1, emb]
        x_a_out = weights[...,None,:] @ torch.cat((x_a[...,None], compat[...,None]),-1).transpose(3,2)
        x_a_out = x_a_out.squeeze(2)

        return x_a_out
    

class EncoderLayer(nn.Module):
    def __init__(self, model_params: ModelParams, activate=True):
        super().__init__()
        self.shelf_emb = HetGCNLayer(model_params)
        self.item_emb = HetGCNLayer(model_params)
        self.activate = activate

    def forward(self, shelf_embedding, item_embedding, cost_mat):

        shelf_embedding_out = self.shelf_emb(shelf_embedding, item_embedding, cost_mat)
        item_embedding_out = self.item_emb(item_embedding, shelf_embedding, cost_mat.transpose(1, 2))

        if self.activate:
            shelf_embedding_out = F.relu(shelf_embedding_out)
            item_embedding_out = F.relu(item_embedding_out)

        return shelf_embedding_out, item_embedding_out
    


class Encoder(nn.Module):
    def __init__(self, model_params: ModelParams):
        super().__init__()
        self.shelf_features = model_params.encoder_features["shelf"]
        self.item_features = model_params.encoder_features["item"]

        self.init_embed_depot = nn.Linear(2, model_params.embedding_dim)
        self.init_shelf_embed = nn.Linear(model_params.infer_feature_size(self.shelf_features), model_params.embedding_dim)
        self.init_item_embed = nn.Linear(model_params.infer_feature_size(self.item_features), model_params.embedding_dim)
        # self.init_item_embed = SimpleItemEncoder(model_params)

        encoder_layer_num = model_params.num_mhsa_layers
        layers = [EncoderLayer(model_params, activate=True) for _ in range(encoder_layer_num)]
        # layers += [EncoderLayer(model_params, activate=False)]
        self.layers = nn.ModuleList(layers)

    def forward(self, state: StateMSVRP):

        shelf_embed = self._init_shelf_embed(state)
        item_embed = self._init_item_embed(state)
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        for layer in self.layers:
            shelf_embed, item_embed = layer(shelf_embed, item_embed, state.supply_w_depot)

        item_embed = add_dummy_item(item_embed)

        return shelf_embed, item_embed
    
    def _init_item_embed(self, input: StateMSVRP):

        features = cat_state_features(input, self.item_features)
        item_embeddings = self.init_item_embed(features)

        return item_embeddings

    def _init_shelf_embed(self, input: StateMSVRP):

        features = cat_state_features(input, self.shelf_features)
        embeddings = torch.cat((self.init_embed_depot(input.depot), self.init_shelf_embed(features)), 1)
        
        return embeddings