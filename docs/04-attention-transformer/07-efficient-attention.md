# 高效注意力机制 (Efficient Attention)

## 概述 Overview

标准注意力的 O(n²) 复杂度限制了长序列处理。本章介绍各种高效注意力机制。

**难度级别**：⭐ 重点章节

## 代码实现 Implementation

### 1. Flash Attention（PyTorch 2.0+）

```python
import torch
import torch.nn.functional as F

# PyTorch 2.0 内置 Flash Attention
def flash_attention(query, key, value, is_causal=False):
    """
    使用 PyTorch 内置的高效注意力
    底层可能使用 Flash Attention
    """
    return F.scaled_dot_product_attention(
        query, key, value,
        is_causal=is_causal,
        dropout_p=0.0
    )

# 使用示例
batch_size, n_heads, seq_len, d_head = 2, 8, 4096, 64

Q = torch.randn(batch_size, n_heads, seq_len, d_head, device='cuda')
K = torch.randn(batch_size, n_heads, seq_len, d_head, device='cuda')
V = torch.randn(batch_size, n_heads, seq_len, d_head, device='cuda')

# 比标准实现快 2-4x，省内存 5-20x
output = flash_attention(Q, K, V, is_causal=True)
print(f"Output shape: {output.shape}")
```

### 2. 线性注意力

```python
import torch
import torch.nn as nn

class LinearAttention(nn.Module):
    """线性注意力 - O(n) 复杂度"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # 使用核函数近似 softmax
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        # 线性注意力: O(n*d) 而不是 O(n²)
        # KV = K^T @ V: (batch, heads, d_k, d_v)
        KV = torch.einsum('bshe,bshv->bhev', K, V)

        # Z = sum(K): 归一化项
        Z = K.sum(dim=1, keepdim=True)  # (batch, 1, heads, d_k)

        # output = Q @ KV / (Q @ Z)
        numerator = torch.einsum('bshd,bhdv->bshv', Q, KV)
        denominator = torch.einsum('bshd,bshd->bsh', Q, Z.expand_as(Q)).unsqueeze(-1)

        output = numerator / (denominator + 1e-6)
        output = output.view(batch_size, seq_len, -1)

        return self.W_o(output)
```

### 3. 稀疏注意力

```python
import torch
import torch.nn as nn

class SparseAttention(nn.Module):
    """局部 + 全局稀疏注意力"""

    def __init__(self, d_model, n_heads, window_size=256, global_tokens=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        self.global_tokens = global_tokens

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 简化示例：只实现滑动窗口注意力
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 分块处理（简化）
        outputs = []
        for i in range(0, seq_len, self.window_size):
            end = min(i + self.window_size, seq_len)
            q_chunk = Q[:, i:end]
            k_chunk = K[:, max(0, i-self.window_size):end]
            v_chunk = V[:, max(0, i-self.window_size):end]

            # 标准注意力在窗口内
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / (self.d_k ** 0.5)
            weights = F.softmax(scores, dim=-1)
            out = torch.matmul(weights, v_chunk)
            outputs.append(out)

        output = torch.cat(outputs, dim=1)
        return self.W_o(output)
```

### 4. 分组查询注意力 (GQA)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    """GQA: 多个 Q 头共享 KV 头（LLaMA 2 使用）"""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 扩展 KV 头
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        # 标准注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(output)

# GQA 参数效率：n_heads=32, n_kv_heads=8 → KV 参数减少 4x
```

## 高效注意力对比

| 方法 | 复杂度 | 优点 | 缺点 |
|------|--------|------|------|
| 标准注意力 | O(n²) | 准确 | 内存和计算瓶颈 |
| Flash Attention | O(n²) | IO 优化 | 需要特殊硬件 |
| 线性注意力 | O(n) | 真正线性 | 精度损失 |
| 稀疏注意力 | O(n·k) | 长序列友好 | 实现复杂 |
| GQA | O(n²) | 参数高效 | 质量略降 |

## 下一步 Next

恭喜完成 Attention 与 Transformer 篇！接下来学习 LLM。

[下一章：分词器原理 →](../05-llm/01-tokenization.md)
