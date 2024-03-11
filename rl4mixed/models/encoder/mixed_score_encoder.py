import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rl4mixed.problems.vrp import StateMSVRP
from rl4mixed.settings import ModelParams
from rl4mixed.models.encoder.graph_encoder import AttentionEncoderBlock, reshape_by_heads
from rl4mixed.models.utils import cat_state_features, add_dummy_item


class MixedScoreFF(nn.Module):
    def __init__(self, model_params: ModelParams) -> None:
        super().__init__()

        head_num = model_params.num_heads
        ms_hidden_dim = model_params.ms_hidden_dim
        mix1_init = (1/2)**(1/2)
        mix2_init = (1/16)**(1/2)

        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)


    def forward(self, dot_product_score, cost_mat_score):
        # dot_product_score shape: (batch, head_num, row_cnt, col_cnt)
        # cost_mat_score shape: (batch, head_num, row_cnt, col_cnt)

        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)

        two_scores_transposed = two_scores.transpose(1,2)
        # shape: (batch, row_cnt, head_num, col_cnt, 2)

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        mixed_scores = ms2.transpose(1,2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)
        return mixed_scores


class MixedScoreMultiHeadAttention(nn.Module):
    def __init__(self, model_params: ModelParams):
        super().__init__()

        embedding_dim = model_params.embedding_dim
        head_num = model_params.num_heads
        qkv_dim = model_params.qkv_dim

        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.norm_factor = 1 / math.sqrt(qkv_dim)

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        # self.init_parameters()

        self.mixed_scores_layer = MixedScoreFF(model_params) if model_params.ms_hidden_dim > 0 else None


    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, row_emb, col_emb, cost_mat):

        # q shape: (batch, head_num, row_cnt, qkv_dim)
        q = reshape_by_heads(self.Wq(row_emb), head_num=self.head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=self.head_num)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        dot_product_score = self.norm_factor * torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)

        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, self.head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt)

        if self.mixed_scores_layer is not None:
            mixed_scores = self.mixed_scores_layer(dot_product_score, cost_mat_score)
        else:
            mixed_scores = dot_product_score * cost_mat_score

        weights = nn.Softmax(dim=3)(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, row_cnt, self.head_num * self.qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat


class EncoderLayer(nn.Module):
    def __init__(self, model_params: ModelParams):
        super().__init__()
        self.row_encoding_block = AttentionEncoderBlock(model_params, MixedScoreMultiHeadAttention)
        self.col_encoding_block = AttentionEncoderBlock(model_params, MixedScoreMultiHeadAttention)

    def forward(self, shelf_emb, item_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        shelf_emb_out = self.row_encoding_block(shelf_emb, item_emb, cost_mat)
        item_emb_out = self.col_encoding_block(item_emb, shelf_emb, cost_mat.transpose(1, 2))

        return shelf_emb_out, item_emb_out
   



class Encoder(nn.Module):
    def __init__(self, model_params: ModelParams):
        super().__init__()
        # Special embedding projection for depot node
        self.shelf_features = model_params.encoder_features["shelf"]
        self.item_features = model_params.encoder_features["item"]

        self.init_embed_depot = nn.Linear(2, model_params.embedding_dim)
        self.init_shelf_embed = nn.Linear(model_params.infer_feature_size(self.shelf_features), model_params.embedding_dim)
        self.init_item_embed = nn.Linear(model_params.infer_feature_size(self.item_features), model_params.embedding_dim)

        encoder_layer_num = model_params.num_mhsa_layers

        self.layers = nn.ModuleList([EncoderLayer(model_params) for _ in range(encoder_layer_num)])

        
    def forward(self, state: StateMSVRP):
        # [BS, num_shelves, emb]
        shelf_embed = self._init_shelf_embed(state)
        # [BS, num_skus+1, emb]
        item_embed = self._init_item_embed(state)
        # [BS, num_shelves+num_skus+1, emb]

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