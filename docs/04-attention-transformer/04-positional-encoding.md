# 位置编码详解 (Positional Encoding)

## 概述 Overview

Transformer 本身不具有位置感知能力（因为注意力是置换不变的），需要通过位置编码注入位置信息。

**难度级别**：⭐ 重点章节

## 代码实现 Implementation

### 1. 正弦位置编码（原始 Transformer）

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码 - Transformer 原始论文"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数位置用 sin，奇数位置用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 测试
pe = SinusoidalPositionalEncoding(d_model=512)
x = torch.randn(2, 100, 512)
output = pe(x)
print(f"Output shape: {output.shape}")

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.imshow(pe.pe[0, :100, :].numpy(), aspect='auto', cmap='RdBu')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Sinusoidal Positional Encoding')
plt.colorbar()
# plt.show()
```

### 2. 可学习的位置编码（BERT/GPT 风格）

```python
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # 注册位置索引
        position_ids = torch.arange(max_len).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        return self.dropout(x)

# 测试
lpe = LearnedPositionalEncoding(d_model=512)
x = torch.randn(2, 100, 512)
output = lpe(x)
print(f"Learnable PE parameters: {lpe.position_embeddings.weight.shape}")
```

### 3. RoPE (Rotary Position Embedding)

```python
import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    """RoPE - 旋转位置编码 (用于 LLaMA, GPT-NeoX 等)"""

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 预计算 cos 和 sin
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, q, k, seq_len=None):
        """
        q, k: (batch, n_heads, seq_len, head_dim)
        返回旋转后的 q, k
        """
        if seq_len is None:
            seq_len = q.size(2)

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        q_embed = self._apply_rotary(q, cos, sin)
        k_embed = self._apply_rotary(k, cos, sin)

        return q_embed, k_embed

    def _apply_rotary(self, x, cos, sin):
        # x: (batch, n_heads, seq_len, head_dim)
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)

        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

        return x * cos + rotated * sin

# 测试
rope = RotaryPositionalEmbedding(dim=64)
q = torch.randn(2, 8, 100, 64)
k = torch.randn(2, 8, 100, 64)
q_rot, k_rot = rope(q, k)
print(f"Rotated Q shape: {q_rot.shape}")
```

### 4. ALiBi (Attention with Linear Biases)

```python
import torch
import torch.nn as nn

class ALiBi(nn.Module):
    """ALiBi - 注意力线性偏置"""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

        # 计算每个头的斜率
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', torch.tensor(slopes))

    def _get_slopes(self, n_heads):
        """生成几何级数的斜率"""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            return (
                get_slopes_power_of_2(closest_power_of_2) +
                self._get_slopes(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
            )

    def forward(self, attention_scores, seq_len):
        """
        给注意力分数添加位置偏置
        attention_scores: (batch, n_heads, seq_q, seq_k)
        """
        # 创建相对位置矩阵
        positions = torch.arange(seq_len, device=attention_scores.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions.abs().float()

        # 计算偏置
        alibi = self.slopes.unsqueeze(1).unsqueeze(1) * relative_positions
        alibi = -alibi.unsqueeze(0)  # (1, n_heads, seq, seq)

        return attention_scores + alibi

# ALiBi 的优势：
# 1. 不需要额外参数
# 2. 支持外推到更长序列
# 3. 训练时更稳定
```

## 位置编码对比

| 方法 | 优点 | 缺点 | 使用模型 |
|------|------|------|----------|
| Sinusoidal | 无需学习、可外推 | 表达能力有限 | 原始 Transformer |
| Learned | 表达能力强 | 无法外推 | BERT, GPT-2 |
| RoPE | 相对位置、可外推 | 实现复杂 | LLaMA, GPT-NeoX |
| ALiBi | 简单、长序列友好 | 可能略差 | BLOOM |

## 下一步 Next

[下一章：Transformer Block 完整实现 →](./05-transformer-block.md)
