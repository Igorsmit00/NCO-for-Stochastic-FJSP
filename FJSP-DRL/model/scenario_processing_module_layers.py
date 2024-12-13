"""
This code is inspired by the paper 'Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks'
The original code can be found at https://github.com/juho-lee/set_transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim_Q, num_heads, batch_first=True)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):

        O = self.attention(Q, K, K, need_weights=False)[0] + Q
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SelfAttentionBlock, self).__init__()
        self.mab = MultiHeadAttentionBlock(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ScenarioProcessingModuleWithoutAggregation(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=True):
        super(ScenarioProcessingModuleWithoutAggregation, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MultiHeadAttentionBlock(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MultiHeadAttentionBlock(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)
