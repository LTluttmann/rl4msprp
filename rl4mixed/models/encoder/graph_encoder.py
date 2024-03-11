import torch
from torch import nn
import torch.nn.functional as F
from rl4mixed.settings import ModelParams


    


class AttentionEncoderBlock(nn.Module):

    def __init__(self, model_params: ModelParams, mhsa_layer_class: nn.Module):
        super(AttentionEncoderBlock, self).__init__()

        embedding_dim = model_params.embedding_dim
        head_num = model_params.num_heads
        qkv_dim = model_params.qkv_dim
        self.head_num = head_num

        self.mhsa = mhsa_layer_class(model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndNormalization(model_params)
        self.feed_forward = FeedForward(model_params)
        self.add_n_normalization_2 = AddAndNormalization(model_params)

    def forward(self, x, *args, **kwargs):

         # shape: (batch, row_cnt, head_num*qkv_dim)
        mhsa_heads = self.mhsa(x, *args, **kwargs)
       
        # shape: (batch, row_cnt, embedding)
        mhsa_out = self.multi_head_combine(mhsa_heads)
        
        out1 = self.add_n_normalization_1(x, mhsa_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
    


class AddAndNormalization(nn.Module):
    def __init__(self, model_params: ModelParams, normalization=None):
        super().__init__()
        embedding_dim = model_params.embedding_dim
        normalization = normalization or model_params.normalization
        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization)

        self.normalizer = normalizer_class(embedding_dim, affine=True)


    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)
        x = input1 + input2

        if isinstance(self.normalizer, nn.BatchNorm1d):
            normalized = self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            normalized = self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            raise ValueError()

        return normalized


class FeedForward(nn.Module):
    def __init__(self, model_params: ModelParams):
        super().__init__()
        embedding_dim = model_params.embedding_dim
        ff_hidden_dim = model_params.feed_forward_hidden

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
    

def reshape_by_heads(qkv, head_num) -> torch.Tensor:
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed
