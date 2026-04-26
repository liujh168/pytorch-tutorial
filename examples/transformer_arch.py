"""
示例: Transformer架构解析
对应文档: docs/LLM Code/Char01.txt - 1.2.1节
功能: 实现Transformer编码器的一部分，包括多头注意力机制和前馈神经网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码模块，用于为输入序列添加位置信息"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        matmul_qk = torch.matmul(Q, K.transpose(-2, -1))
        dk = K.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        """将输入分割成多个注意力头"""
        return x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.q_linear(Q)
        K = self.k_linear(K)
        V = self.v_linear(V)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        attention, weights = self.scaled_dot_product_attention(Q, K, V, mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(attention)


class FeedForwardNetwork(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.mha(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout(ffn_output))
        return out2


if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    d_ff = 2048
    seq_len = 10
    batch_size = 2

    sample_input = torch.rand(batch_size, seq_len, d_model)
    mask = None

    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
    positional_encoding = PositionalEncoding(d_model)

    input_with_pos = positional_encoding(sample_input)
    output = encoder_layer(input_with_pos, mask)

    print("输入形状:", sample_input.shape)
    print("输出形状:", output.shape)