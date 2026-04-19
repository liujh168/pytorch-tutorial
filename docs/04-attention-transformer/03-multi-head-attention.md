# Multi-Head Attention 详解 (Multi-Head Attention)

## 概述 Overview

Multi-Head Attention 通过并行运行多个注意力头，让模型能够同时关注不同子空间的信息。

**难度级别**：⭐ 重点章节

## 代码实现 Implementation

### 1. 完整的 Multi-Head Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention 完整实现"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        """
        query: (batch, seq_q, d_model)
        key: (batch, seq_k, d_model)
        value: (batch, seq_k, d_model)
        mask: (batch, 1, seq_q, seq_k) 或 (batch, 1, 1, seq_k)
        """
        batch_size = query.size(0)

        # 1. 线性投影
        Q = self.W_q(query)  # (batch, seq_q, d_model)
        K = self.W_k(key)    # (batch, seq_k, d_model)
        V = self.W_v(value)  # (batch, seq_k, d_model)

        # 2. 分割成多头
        # (batch, seq, d_model) -> (batch, seq, n_heads, d_k) -> (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 3. 计算注意力
        # (batch, n_heads, seq_q, d_k) @ (batch, n_heads, d_k, seq_k)
        # -> (batch, n_heads, seq_q, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # (batch, n_heads, seq_q, seq_k) @ (batch, n_heads, seq_k, d_k)
        # -> (batch, n_heads, seq_q, d_k)
        context = torch.matmul(attention_weights, V)

        # 4. 合并多头
        # (batch, n_heads, seq_q, d_k) -> (batch, seq_q, n_heads, d_k) -> (batch, seq_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. 输出投影
        output = self.W_o(context)

        return output, attention_weights

# 测试
mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)
output, weights = mha(x, x, x)

print(f"Input: {x.shape}")
print(f"Output: {output.shape}")
print(f"Attention weights: {weights.shape}")  # (batch, n_heads, seq, seq)
```

### 2. 使用 PyTorch nn.MultiheadAttention

```python
import torch
import torch.nn as nn

# PyTorch 内置的 MultiheadAttention
mha = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True  # 重要：使用 batch first 格式
)

# 使用
query = torch.randn(2, 10, 512)  # (batch, seq, d_model)
key = torch.randn(2, 20, 512)
value = torch.randn(2, 20, 512)

# Self-attention
output, weights = mha(query, query, query)

# Cross-attention
output, weights = mha(query, key, value)

# 带 mask
# key_padding_mask: (batch, seq_k) - True 表示忽略
key_padding_mask = torch.zeros(2, 20).bool()
key_padding_mask[0, 15:] = True  # 第一个样本后5个位置是padding

# attn_mask: (seq_q, seq_k) 或 (batch*n_heads, seq_q, seq_k)
# 用于因果注意力
attn_mask = torch.triu(torch.ones(10, 10), diagonal=1).bool()

output, weights = mha(query, query, query, attn_mask=attn_mask)
```

### 3. 多头注意力的直觉理解

```python
# 为什么使用多头？

# 1. 单头注意力：一种"关注方式"
#    - 可能只关注语法关系，或只关注语义关系

# 2. 多头注意力：多种"关注方式"并行
#    - Head 1: 可能学习语法依赖
#    - Head 2: 可能学习语义相似
#    - Head 3: 可能学习位置关系
#    - ...

# 可视化不同头的注意力模式
import matplotlib.pyplot as plt

def visualize_heads(attention_weights, tokens):
    """可视化多头注意力"""
    n_heads = attention_weights.size(1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, ax in enumerate(axes.flat):
        if i < n_heads:
            ax.imshow(attention_weights[0, i].detach().numpy(), cmap='Blues')
            ax.set_title(f'Head {i+1}')
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45)
            ax.set_yticklabels(tokens)

    plt.tight_layout()
    # plt.show()
```

## 深入理解：自定义实现 vs PyTorch 内置对比

```python
import torch
import torch.nn as nn
import math, time

# 验证自定义实现与 PyTorch 内置的等价性
torch.manual_seed(42)
d_model, n_heads, seq_len, batch = 64, 4, 10, 2

# 自定义实现（来自上面的 MultiHeadAttention）
class MultiHeadAttentionCustom(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

    def forward(self, q, k, v):
        B = q.size(0)
        Q = self.W_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        out    = torch.matmul(torch.softmax(scores, dim=-1), V)
        out    = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.W_o(out)

custom_mha  = MultiHeadAttentionCustom(d_model, n_heads)
builtin_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)

x = torch.randn(batch, seq_len, d_model)
out_custom  = custom_mha(x, x, x)
out_builtin, _ = builtin_mha(x, x, x)

print(f"自定义 MHA 输出 shape:  {out_custom.shape}")
print(f"内置 MHA 输出 shape:    {out_builtin.shape}")
print("(注意：两者权重不同，输出数值不同，但形状相同)")

# 性能对比
N_ITER = 100
x_large = torch.randn(8, 128, d_model)

start = time.perf_counter()
for _ in range(N_ITER):
    with torch.no_grad():
        custom_mha(x_large, x_large, x_large)
print(f"自定义 MHA: {(time.perf_counter()-start)*1000/N_ITER:.2f} ms/iter")

start = time.perf_counter()
for _ in range(N_ITER):
    with torch.no_grad():
        builtin_mha(x_large, x_large, x_large)
print(f"内置 MHA:   {(time.perf_counter()-start)*1000/N_ITER:.2f} ms/iter")
print("(内置实现经过 CUDA kernel 优化，生产环境优先使用)")
```

## 深入理解：Cross-Attention（编码器-解码器注意力）

```python
import torch
import torch.nn as nn

# Cross-Attention: Q 来自 decoder，K/V 来自 encoder
# 用于 Seq2Seq 任务（机器翻译、摘要、语音识别）

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Q 投影：作用于 decoder 状态
        # K/V 投影：作用于 encoder 输出
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, decoder_hidden, encoder_output, memory_key_padding_mask=None):
        """
        decoder_hidden:          (batch, tgt_len, d_model) — 解码器的当前状态
        encoder_output:          (batch, src_len, d_model) — 编码器的全部输出
        memory_key_padding_mask: (batch, src_len)          — True 表示 padding
        """
        # Q = decoder_hidden, K = V = encoder_output
        out, attn_weights = self.attn(
            query=decoder_hidden,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=memory_key_padding_mask,
        )
        return out, attn_weights


# 演示：英译中场景
d_model, n_heads = 128, 4
cross_attn = CrossAttention(d_model, n_heads)

src_len, tgt_len = 20, 15   # 源语言 20 个词，目标语言 15 个词
encoder_out    = torch.randn(2, src_len, d_model)   # 编码器输出
decoder_hidden = torch.randn(2, tgt_len, d_model)   # 解码器当前状态

# 最后 3 个源词是 padding
pad_mask = torch.zeros(2, src_len, dtype=torch.bool)
pad_mask[:, -3:] = True

out, weights = cross_attn(decoder_hidden, encoder_out, memory_key_padding_mask=pad_mask)
print(f"Cross-Attention 输出 shape:  {out.shape}")       # (2, 15, 128)
print(f"注意力权重 shape:            {weights.shape}")   # (2, 15, 20)
print("(每个目标词对 20 个源词的注意力分布)")
```

## 小结 Summary

```python
# Multi-Head Attention 核心步骤：
# 1. 投影到 Q, K, V（各 n_heads 个子空间）
# 2. 每个头独立计算 Scaled Dot-Product Attention
# 3. 合并所有头的输出
# 4. 输出投影

# 三种使用模式:
# Self-Attention:  Q=K=V=x               (Encoder)
# Causal-Self:     Q=K=V=x + 因果掩码    (Decoder 自注意力)
# Cross-Attention: Q=decoder, K=V=encoder (Decoder 交叉注意力)
```

| 变体 | Q 来源 | K/V 来源 | 掩码 | 典型位置 |
|------|--------|---------|------|----------|
| Self-Attention | 自身 | 自身 | 无（或 padding） | Encoder |
| Causal Self-Attn | 自身 | 自身 | 因果掩码 | Decoder 第1层 |
| Cross-Attention | Decoder | Encoder | padding | Decoder 第2层 |

## 下一步 Next

[下一章：位置编码详解 →](./04-positional-encoding.md)
