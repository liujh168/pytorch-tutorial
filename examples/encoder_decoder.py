"""
示例: 编码器-解码器
对应文档: docs/LLM Code/Char01.txt - 1.2.2节
功能: 实现基于PyTorch的简化版编码器-解码器模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)


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

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float))
        if mask is not None:
            scores += mask * -1e9
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.q_linear(Q)
        K = self.k_linear(K)
        V = self.v_linear(V)
        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        attention, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.depth)
        return self.fc_out(attention)


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class Encoder(nn.Module):
    """编码器"""
    def __init__(self, input_dim, d_model, num_heads, num_layers, d_ff, max_len):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(d_model, num_heads),
                nn.LayerNorm(d_model),
                FeedForward(d_model, d_ff),
                nn.LayerNorm(d_model)
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for mha, norm1, ffn, norm2 in self.layers:
            attn_output = mha(x, x, x, mask)
            x = norm1(x + attn_output)
            ffn_output = ffn(x)
            x = norm2(x + ffn_output)
        return x


class Decoder(nn.Module):
    """解码器"""
    def __init__(self, output_dim, d_model, num_heads, num_layers, d_ff, max_len):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(d_model, num_heads),
                nn.LayerNorm(d_model),
                MultiHeadAttention(d_model, num_heads),
                nn.LayerNorm(d_model),
                FeedForward(d_model, d_ff),
                nn.LayerNorm(d_model)
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for mha1, norm1, mha2, norm2, ffn, norm3 in self.layers:
            attn1 = mha1(x, x, x, tgt_mask)
            x = norm1(x + attn1)
            attn2 = mha2(x, enc_output, enc_output, src_mask)
            x = norm2(x + attn2)
            ffn_output = ffn(x)
            x = norm3(x + ffn_output)
        return x


if __name__ == "__main__":
    input_dim = 1000
    output_dim = 1000
    d_model = 512
    num_heads = 8
    num_layers = 2
    d_ff = 2048
    max_len = 100

    src_seq = torch.randint(0, input_dim, (2, 10))
    tgt_seq = torch.randint(0, output_dim, (2, 10))

    encoder = Encoder(input_dim, d_model, num_heads, num_layers, d_ff, max_len)
    decoder = Decoder(output_dim, d_model, num_heads, num_layers, d_ff, max_len)

    enc_output = encoder(src_seq)
    dec_output = decoder(tgt_seq, enc_output)

    print("编码器输出形状:", enc_output.shape)
    print("解码器输出形状:", dec_output.shape)