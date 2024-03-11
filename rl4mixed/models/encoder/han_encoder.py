import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rl4mixed.problems.vrp import StateMSVRP
from rl4mixed.settings import ModelParams
from rl4mixed.models.utils import cat_state_features, add_dummy_item


class HANLayer(nn.Module):
    def __init__(self, model_params: ModelParams):
        super().__init__()
        self.attention_mapping = nn.Linear(2*model_params.embedding_dim, 1, bias=False)

    def forward(self, row_emb, col_emb, cost_mat):
        
        bs,num_rows,emb = row_emb.shape
        num_cols = col_emb.size(1)

        row_emb_ext = row_emb[:,:,None,:].expand(bs, num_rows, num_cols, emb)
        col_embed_ext = col_emb[:,None].expand_as(row_emb_ext)
        # [BS, rows, cols, 2*emb_dim]
        concat_emb = torch.cat((row_emb_ext, col_embed_ext), -1)
        # [BS, rows, cols]
        attn_logits = self.attention_mapping(concat_emb).squeeze(-1)
        attn_logits *= cost_mat
        attn_logits_activated = F.leaky_relu(attn_logits)
        # in HAN attn scores are only computed for first-order neighbors
        # mask = cost_mat.eq(0)
        # attn_logits_activated[mask] = -math.inf 
        # [BS, rows, 1]
        self_logit = self.attention_mapping(torch.cat((row_emb, row_emb), -1))
        self_logit_activated = F.leaky_relu(self_logit)
        # [BS, rows, 1+cols]
        all_logits = torch.cat((self_logit_activated, attn_logits_activated), 2)
        # [BS, rows, 1+cols]
        attn_scores = F.softmax(all_logits, 2)
        # [BS, rows, 1+cols, emb_dim] 
        values = torch.cat((
            row_emb[:,:,None,:],
            col_emb[:,None].expand(bs, num_rows, num_cols, emb)
        ), 2)

        emb = attn_scores[:,:,None,:] @ values
        # [BS, rows, emb_dim] 
        return emb.squeeze(2)


class EncoderLayer(nn.Module):

    def __init__(self, model_params: ModelParams) -> None:
        super().__init__()
        self.han_shelf = HANLayer(model_params)
        self.han_item = HANLayer(model_params)

    def forward(self, shelf_emb, item_emb, cost_mat):
        shelf_emb_out = self.han_shelf(shelf_emb, item_emb, cost_mat)
        item_emb_out = self.han_item(item_emb, shelf_emb, cost_mat.transpose(1,2))

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
        # self.init_item_embed = SimpleItemEncoder(model_params)

        encoder_layer_num = model_params.num_mhsa_layers
        self.layers = nn.ModuleList([
            EncoderLayer(model_params)
            # AttentionEncoderBlock(model_params, MaskedMultiHeadSelfAttention) 
            for _ in range(encoder_layer_num)
        ])

        
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
    
