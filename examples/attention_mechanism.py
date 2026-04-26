"""
示例: 注意力机制详解
对应文档: docs/LLM Code/Char01.txt - 1.2.3节
功能: 实现Scaled Dot-Product Attention和多头注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        将 d_model 维度拆分为 num_heads 个 head，每个 head 的维度为 depth
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.split_heads(self.q_linear(Q), batch_size)
        K = self.split_heads(self.k_linear(K), batch_size)
        V = self.split_heads(self.v_linear(V), batch_size)
        attention, weights = ScaledDotProductAttention()(Q, K, V, mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.depth)
        output = self.fc_out(attention)
        return output, weights


if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    seq_len = 10
    batch_size = 2

    Q = torch.rand(batch_size, seq_len, d_model)
    K = torch.rand(batch_size, seq_len, d_model)
    V = torch.rand(batch_size, seq_len, d_model)

    multi_head_attention = MultiHeadAttention(d_model, num_heads)
    output, attention_weights = multi_head_attention(Q, K, V)

    print("多头注意力输出形状:", output.shape)
    print("注意力权重形状:", attention_weights.shape)