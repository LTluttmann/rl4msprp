import torch
import torch.nn as nn
from rl4mixed.problems.vrp import StateMSVRP
from rl4mixed.settings import ModelParams
from rl4mixed.models.encoder.graph_encoder import AddAndNormalization, FeedForward
from rl4mixed.models.encoder.mhsa_encoder import MultiHeadAttention
from rl4mixed.models.encoder.mixed_score_encoder import MixedScoreMultiHeadAttention
from rl4mixed.models.utils import cat_state_features, add_dummy_item

    


class SelfAndCrossAttnLayer(nn.Module):

    def __init__(self, model_params: ModelParams):
        super(SelfAndCrossAttnLayer, self).__init__()

        embedding_dim = model_params.embedding_dim
        head_num = model_params.num_heads
        qkv_dim = model_params.qkv_dim
        self.head_num = head_num

        self.self_attn = MultiHeadAttention(model_params)
        self.cross_attn = MixedScoreMultiHeadAttention(model_params)

        self.multi_head_combine1 = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.multi_head_combine2 = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndNormalization(model_params)
        self.add_n_normalization_2 = AddAndNormalization(model_params)
        self.add_n_normalization_3 = AddAndNormalization(model_params)

        self.feed_forward = FeedForward(model_params)


    def forward(self, row_emb, col_emb, cost_mat):

         # shape: (batch, row_cnt, head_num*qkv_dim)
        mhsa_heads = self.self_attn(row_emb)
        # shape: (batch, row_cnt, embedding)
        mhsa_out = self.multi_head_combine1(mhsa_heads)
        out_sa = self.add_n_normalization_1(row_emb, mhsa_out)

        msmha_heads = self.cross_attn(out_sa, col_emb, cost_mat)
        msmha_out = self.multi_head_combine2(msmha_heads)

        out_ca = self.add_n_normalization_2(out_sa, msmha_out)

        out1 = self.feed_forward(out_ca)
        out2 = self.add_n_normalization_3(out_ca, out1)

        return out2
    

class EncoderLayer(nn.Module):
    def __init__(self, model_params: ModelParams):
        super().__init__()
        self.shelf_encoding_block = SelfAndCrossAttnLayer(model_params)
        self.item_encoding_block = SelfAndCrossAttnLayer(model_params)

    def forward(self, shelf_emb, item_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        shelf_emb_out = self.shelf_encoding_block(shelf_emb, item_emb, cost_mat)
        item_emb_out = self.item_encoding_block(item_emb, shelf_emb, cost_mat.transpose(1, 2))

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
    
