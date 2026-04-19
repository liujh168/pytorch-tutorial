# 词嵌入与位置嵌入 (Embeddings)

## 概述 Overview

嵌入层将离散的 token 映射到连续的向量空间，是语言模型的基础组件。

## 代码实现 Implementation

### 1. Token Embedding

```python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 通常会乘以 sqrt(d_model) 进行缩放
        return self.embedding(x) * (self.d_model ** 0.5)

# 使用
vocab_size = 32000
d_model = 512
embed = TokenEmbedding(vocab_size, d_model)

tokens = torch.randint(0, vocab_size, (2, 10))
embeddings = embed(tokens)
print(f"Input: {tokens.shape}")
print(f"Embeddings: {embeddings.shape}")
```

### 2. 完整的嵌入层

```python
import torch
import torch.nn as nn
import math

class TransformerEmbedding(nn.Module):
    """Token + Position Embedding"""

    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        """
        x: (batch, seq_len) token ids
        """
        seq_len = x.size(1)

        # Token embedding
        tok_emb = self.token_embedding(x) * math.sqrt(self.d_model)

        # Position embedding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        # 相加
        embeddings = tok_emb + pos_emb

        return self.dropout(embeddings)

# 测试
embed_layer = TransformerEmbedding(vocab_size=32000, d_model=512)
tokens = torch.randint(0, 32000, (2, 100))
embeddings = embed_layer(tokens)
print(f"Output: {embeddings.shape}")  # (2, 100, 512)
```

### 3. 权重共享 (Weight Tying)

```python
class LMHead(nn.Module):
    """语言模型输出头，与 embedding 共享权重"""

    def __init__(self, d_model, vocab_size, embed_weight=None):
        super().__init__()
        if embed_weight is not None:
            # 共享 embedding 权重
            self.weight = embed_weight
        else:
            self.weight = nn.Parameter(torch.randn(vocab_size, d_model))

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        output: (batch, seq_len, vocab_size)
        """
        return x @ self.weight.T

# 使用权重共享
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True),
            n_layers
        )
        # 与 embedding 共享权重
        self.lm_head = LMHead(d_model, vocab_size, self.embedding.weight)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.lm_head(x)
```

## 深入理解：三种位置编码对比

```python
import torch
import torch.nn as nn
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── 1. 正弦位置编码（Sinusoidal，原始 Transformer）────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ─── 2. 可学习位置嵌入（Learnable，GPT-2 风格）────────────────
class LearnablePE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(pos)

# ─── 3. RoPE - Rotary Position Embedding（LLaMA 风格）──────────
def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple:
    """
    旋转位置编码：通过旋转 Q/K 向量编码位置信息
    q, k: (batch, n_heads, seq_len, d_head)
    """
    _, _, seq_len, d = q.shape
    # 生成旋转角度
    theta = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float) / d))
    pos   = torch.arange(seq_len, dtype=torch.float)
    freqs = torch.outer(pos, theta)             # (seq, d/2)
    cos   = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d/2)
    sin   = freqs.sin().unsqueeze(0).unsqueeze(0)

    def rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]    # 偶数/奇数维度
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos.repeat(1, 1, 1, 2) + rotated * sin.repeat(1, 1, 1, 2)

    return rotate(q), rotate(k)

# 对比三种位置编码的特性
d_model, seq_len = 64, 50
x = torch.randn(1, seq_len, d_model)

sin_pe  = SinusoidalPE(d_model)
learn_pe = LearnablePE(d_model)

print("=== 位置编码参数量对比 ===")
print(f"正弦编码 (无参数):   0")
print(f"可学习编码:          {sum(p.numel() for p in learn_pe.parameters()):,}")
print(f"RoPE (无额外参数):   0  (在注意力计算中动态生成)")

# 可视化正弦位置编码
pe_matrix = sin_pe.pe[0, :20, :32].numpy()
plt.figure(figsize=(10, 4))
plt.imshow(pe_matrix, cmap='RdBu', aspect='auto')
plt.colorbar()
plt.xlabel('Embedding 维度')
plt.ylabel('Token 位置')
plt.title('正弦位置编码可视化（前 20 个位置，前 32 个维度）')
plt.tight_layout()
plt.savefig('docs/05-llm/sinusoidal_pe.png', dpi=100)
print("\n位置编码可视化已保存到 docs/05-llm/sinusoidal_pe.png")
```

## 深入理解：嵌入空间分析

```python
import torch
import torch.nn as nn

# 词嵌入空间的核心属性
def analyze_embeddings(vocab_size=1000, d_model=64, n_samples=100):
    """分析随机初始化 vs 训练后嵌入的差异"""
    torch.manual_seed(42)
    emb = nn.Embedding(vocab_size, d_model)

    # 随机初始化（模拟训练前）
    tokens = torch.randint(0, vocab_size, (n_samples,))
    vecs   = emb(tokens)  # (n_samples, d_model)

    # 余弦相似度矩阵
    norms = vecs.norm(dim=1, keepdim=True)
    vecs_norm = vecs / norms.clamp(min=1e-8)
    sim_matrix = vecs_norm @ vecs_norm.T

    print(f"嵌入统计（随机初始化）:")
    print(f"  均值:         {vecs.mean():.4f}")
    print(f"  标准差:       {vecs.std():.4f}")
    print(f"  平均 L2 范数: {norms.mean():.4f}")
    print(f"  平均相似度:   {sim_matrix.mean():.4f}  (趋近 0 = 随机分散)")

    # 权重绑定（input embedding = output projection）的意义
    print(f"\n权重绑定的优点:")
    print(f"  - 节省参数: vocab_size × d_model = {vocab_size * d_model:,} 个参数")
    print(f"  - 语义一致: 输入和输出使用同一向量空间")
    print(f"  - 训练更稳定: 梯度同时更新两个方向的表示")


analyze_embeddings()
```

## 位置编码选择指南

| 类型 | 参数量 | 长度外推 | 相对位置 | 适用场景 |
|------|--------|----------|----------|----------|
| 正弦编码 | 0 | 有限 | 隐式 | 原始 Transformer |
| 可学习编码 | max_len × d | 需重训练 | 无 | GPT-2, BERT |
| RoPE | 0 | 良好 | 显式 | LLaMA, Mistral, Qwen |
| ALiBi | 0 | 极好 | 显式 | BLOOM, MPT |

## 下一步 Next

[下一章：语言模型目标函数 →](./03-language-modeling.md)
