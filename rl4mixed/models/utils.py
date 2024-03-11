import torch
from typing import List


def get_curr_node_embed(embeddings, node_idx_to_pick):
    # encoded_nodes shape: (bs, nodes, embedding)
    # node_index_to_pick shape: (bs, 1)
    batch_size = embeddings.size(0)
    embedding_dim = embeddings.size(2)

    # shape: (bs, 1, emb)
    if len(node_idx_to_pick.shape) == 1:
        node_idx_to_pick = node_idx_to_pick[:, None]
    gathering_index = node_idx_to_pick[:, :, None].expand(batch_size, 1, embedding_dim)
    
    # shape: (bs, 1, embedding)
    picked_nodes = embeddings.gather(dim=1, index=gathering_index)
    
    # shape: (bs, 1, embedding)
    return picked_nodes



def cat_state_features(state, feature_list: List[str]):
    feature_list = [feature_list] if isinstance(feature_list, str) else feature_list
    features = [state.parse_feature(x) for x in feature_list]
    # check all features have same dimensions exept feature dimension
    assert all([feat.shape[:-1] == features[0].shape[:-1] for feat in features])
    return torch.cat(tuple(features), -1)



def ensure_node_dim_exists(vec: torch.Tensor, batched: bool=True):
    dims_assumed = 2 + int(batched)
    if len(vec.shape) < dims_assumed:
        return ensure_node_dim_exists(vec.unsqueeze(1), batched=batched)
    return vec


def add_dummy_item(seq_embeddings):
    if len(seq_embeddings.shape) == 3:
        bs, _, emb_dim = seq_embeddings.shape
        dummy = torch.zeros((bs, 1, emb_dim), device=seq_embeddings.device)
        # [BS, num_items+1, emb_dim]
        emb = torch.cat((dummy, seq_embeddings), 1)
    elif len(seq_embeddings.shape) == 2:
        bs, _ = seq_embeddings.shape
        dummy = torch.zeros((bs, 1), device=seq_embeddings.device)
        # [BS, num_items+1]
        emb = torch.cat((dummy, seq_embeddings), 1)
    else:
        raise ValueError("wrong shape")
    return emb

