# Transformer Block 完整实现 (Transformer Block)

## 概述 Overview

Transformer Block 是 Transformer 的基本构建单元。本章实现一个完整的 Transformer Block。

**难度级别**：⭐ 重点章节

## 代码实现 Implementation

### 1. 完整的 Transformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        return self.W_o(context)

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """标准 Transformer Block"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.pre_norm:
            # Pre-LN: Norm -> Attention -> Residual
            x = x + self.dropout1(self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask))
            x = x + self.dropout2(self.ff(self.norm2(x)))
        else:
            # Post-LN: Attention -> Residual -> Norm
            x = self.norm1(x + self.dropout1(self.attention(x, x, x, mask)))
            x = self.norm2(x + self.dropout2(self.ff(x)))

        return x

# 测试
block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
x = torch.randn(2, 100, 512)
output = block(x)
print(f"Input: {x.shape}")
print(f"Output: {output.shape}")
```

### 2. 堆叠多个 Block

```python
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    """Transformer Encoder - 堆叠多个 Block"""

    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1, max_len=512):
        super().__init__()

        self.d_model = d_model

        # 位置编码
        self.pos_encoding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # 堆叠 Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model) - 已经嵌入的输入
        """
        seq_len = x.size(1)

        # 添加位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_encoding(positions)
        x = self.dropout(x)

        # 通过所有层
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

# 测试
encoder = TransformerEncoder(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_layers=6
)

x = torch.randn(2, 100, 512)
output = encoder(x)
print(f"Encoder output: {output.shape}")
print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
```

### 3. 参数初始化

```python
def init_transformer_weights(module, init_std=0.02):
    """Transformer 权重初始化"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=init_std)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

# 应用初始化
encoder.apply(lambda m: init_transformer_weights(m, init_std=0.02))
```

## 架构变体

| 变体 | 特点 | 使用模型 |
|------|------|----------|
| Post-LN | 原始架构，训练较难 | 原始 Transformer |
| Pre-LN | 训练更稳定 | GPT-2, GPT-3 |
| Sandwich-LN | 结合两者优点 | 一些变体 |
| RMSNorm | 计算更高效 | LLaMA |

## 下一步 Next

[下一章：Encoder-Decoder 架构 →](./06-encoder-decoder.md)
