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

## 小结 Summary

```python
# Multi-Head Attention 核心步骤：
# 1. 投影到 Q, K, V
# 2. 分割成 n_heads 个头
# 3. 每个头独立计算注意力
# 4. 合并所有头
# 5. 输出投影

# 优势：
# - 并行计算效率高
# - 捕获多种依赖关系
# - 参数量与单头相同
```

## 下一步 Next

[下一章：位置编码详解 →](./04-positional-encoding.md)
