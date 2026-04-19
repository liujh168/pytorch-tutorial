# Self-Attention 数学推导与实现 (Self-Attention)

## 概述 Overview

Self-Attention（自注意力）是 Transformer 的核心组件。本章从数学角度详细推导并实现自注意力机制。

**难度级别**：⭐ 重点章节

## 核心概念 Core Concepts

### 数学定义

给定输入序列 X ∈ ℝ^(n×d)，Self-Attention 的计算过程：

```
1. 线性投影：
   Q = X · W_Q,  W_Q ∈ ℝ^(d×d_k)
   K = X · W_K,  W_K ∈ ℝ^(d×d_k)
   V = X · W_V,  W_V ∈ ℝ^(d×d_v)

2. 注意力分数：
   A = softmax(Q · K^T / √d_k)

3. 加权求和：
   Output = A · V
```

## 代码实现 Implementation

### 1. 从零实现 Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """从零实现的 Self-Attention"""

    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model

        # 线性投影层
        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_v, bias=False)

        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len, seq_len) 或 (1, seq_len, seq_len)
        """
        # 线性投影
        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_v)

        # 计算注意力分数
        # (batch, seq_len, d_k) @ (batch, d_k, seq_len) -> (batch, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用 mask（用于因果注意力等）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax 归一化
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权求和
        output = torch.matmul(attention_weights, V)  # (batch, seq_len, d_v)

        return output, attention_weights

# 测试
batch_size, seq_len, d_model = 2, 10, 64
x = torch.randn(batch_size, seq_len, d_model)

attention = SelfAttention(d_model)
output, weights = attention(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Weights sum (should be 1): {weights[0, 0].sum():.4f}")
```

### 2. 因果 Self-Attention（用于语言模型）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """因果自注意力：只能看到当前及之前的位置"""

    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.scale = math.sqrt(d_model)

        # 注册因果 mask（下三角矩阵）
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('mask', mask)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用因果 mask
        causal_mask = self.mask[:seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Softmax
        weights = F.softmax(scores, dim=-1)

        # 输出
        output = torch.matmul(weights, V)
        output = self.W_o(output)

        return output, weights

# 测试
attention = CausalSelfAttention(d_model=64)
x = torch.randn(2, 10, 64)
output, weights = attention(x)

print(f"Causal mask:\n{attention.mask[:5, :5]}")
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
```

### 3. 使用 PyTorch 内置实现

```python
import torch
import torch.nn as nn

# PyTorch 提供的 Scaled Dot-Product Attention（PyTorch 2.0+）
def pytorch_attention(query, key, value, attn_mask=None, is_causal=False):
    """
    使用 F.scaled_dot_product_attention
    这个函数内部可能使用 Flash Attention 优化
    """
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        dropout_p=0.0
    )

# 示例
batch_size, seq_len, d_model = 2, 100, 64

Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

# 普通注意力
output = pytorch_attention(Q, K, V)

# 因果注意力
output_causal = pytorch_attention(Q, K, V, is_causal=True)
```

## 深入理解 Deep Dive

### 计算复杂度分析

```python
# Self-Attention 的复杂度：
# - 时间复杂度：O(n² · d)，n 是序列长度，d 是维度
# - 空间复杂度：O(n²)，需要存储 n×n 的注意力矩阵

# 这就是为什么长序列会很慢的原因
# 解决方案见后续章节：Flash Attention、稀疏注意力等
```

### 为什么 Self-Attention 有效？

```
1. 全局感受野：每个位置可以直接访问所有其他位置
2. 并行计算：不像 RNN 需要顺序计算
3. 灵活的依赖建模：可以学习任意复杂的依赖关系
4. 可解释性：注意力权重提供了一种解释模型关注的方式
```

## 练习题 Exercises

**练习 1（🟢 入门）**: 实现一个函数，接受注意力权重矩阵（shape `(seq, seq)`），用 `matplotlib` 绘制热力图，横/纵轴标注 token 文本（如 `["The", "cat", "sat"]`）。

<details>
<summary>参考答案</summary>

```python
import torch, matplotlib.pyplot as plt

def plot_attention(weights: torch.Tensor, tokens: list):
    """weights: (seq, seq) 注意力权重"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(weights.detach().cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=45)
    ax.set_yticks(range(len(tokens))); ax.set_yticklabels(tokens)
    plt.colorbar(im)
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.show()
```

</details>

---

**练习 2（🟡 进阶）**: 在 `scaled_dot_product_attention` 的基础上，添加 **Dropout** 支持（在 softmax 之后对注意力权重进行 dropout），并解释为什么 dropout 要在 softmax 之后、矩阵乘 V 之前施加。

<details>
<summary>参考答案</summary>

```python
import torch, torch.nn.functional as F, math

def attention_with_dropout(Q, K, V, dropout_p=0.1, training=True, mask=None):
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    # Dropout 在 softmax 后：随机丢弃某些 token 的注意力，迫使模型不过度依赖单一位置
    if training and dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    return torch.matmul(attn, V), attn
```

</details>

---

**练习 3（🔴 挑战）**: 实现 **相对位置编码的注意力**：注意力分数加上可学习的相对位置偏置 `bias[i-j]`，参数量应为 `(2*seq_len-1,)`。

<details>
<summary>提示</summary>

创建 shape `(2*max_len-1,)` 的可学习参数，用 `i-j+seq_len-1` 作为索引取出每对位置的偏置。

</details>

<details>
<summary>参考答案</summary>

```python
import torch, torch.nn as nn, math

class RelativeAttention(nn.Module):
    def __init__(self, d_k: int, max_len: int = 64):
        super().__init__()
        self.d_k     = d_k
        self.max_len = max_len
        self.rel_bias = nn.Parameter(torch.zeros(2 * max_len - 1))

    def forward(self, Q, K, V):
        S = Q.size(-2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        i = torch.arange(S, device=Q.device).unsqueeze(1)
        j = torch.arange(S, device=Q.device).unsqueeze(0)
        idx = i - j + self.max_len - 1
        scores = scores + self.rel_bias[idx]
        return torch.matmul(F.softmax(scores, dim=-1), V)
```

</details>

## 小结 Summary

```python
# Self-Attention 核心代码
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
output = softmax(Q @ K.T / sqrt(d_k)) @ V
```

## 下一步 Next

[下一章：Multi-Head Attention 详解 →](./03-multi-head-attention.md)
