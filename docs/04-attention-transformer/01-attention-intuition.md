# 注意力机制直觉理解 (Attention Intuition)

## 概述 Overview

注意力机制是现代深度学习最重要的突破之一，它让模型能够"关注"输入中最相关的部分。本章从直觉角度理解注意力机制。

**难度级别**：⭐ 重点章节

## 核心概念 Core Concepts

### 什么是注意力？

想象你在阅读一段文字来回答问题。你不会平等地关注每个词，而是会"注意"与问题最相关的部分。这就是注意力机制的核心思想。

```
问题："谁发明了电话？"
文本："亚历山大·贝尔在1876年发明了电话。他是一位著名的发明家。"
注意力：   ████████░░░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
          高              低               高              低
```

### 注意力的数学形式

```
Attention(Q, K, V) = softmax(Q · K^T / √d) · V

其中：
- Q (Query): 查询 - "我在找什么？"
- K (Key): 键 - "我有什么？"
- V (Value): 值 - "我要返回什么？"
```

## 代码实现 Implementation

### 1. 简单的注意力示例

```python
import torch
import torch.nn.functional as F

# === 直觉示例：寻找相似的词 ===
# 假设我们有词向量
vocabulary = {
    'cat': torch.tensor([1.0, 0.0, 0.0, 1.0]),    # 动物
    'dog': torch.tensor([1.0, 0.0, 0.0, 0.8]),    # 动物
    'car': torch.tensor([0.0, 1.0, 0.0, 0.0]),    # 交通工具
    'bike': torch.tensor([0.0, 0.8, 0.0, 0.0]),   # 交通工具
    'apple': torch.tensor([0.0, 0.0, 1.0, 0.0])   # 水果
}

query = vocabulary['cat']  # 查询：cat

# 计算与所有词的相似度（点积）
keys = torch.stack(list(vocabulary.values()))
similarities = query @ keys.T
print("Similarities:", dict(zip(vocabulary.keys(), similarities.tolist())))

# 应用 softmax 得到注意力权重
attention_weights = F.softmax(similarities, dim=0)
print("Attention weights:", dict(zip(vocabulary.keys(), attention_weights.tolist())))

# 结果：dog 的权重最高，因为它和 cat 最相似

# === 基本注意力函数 ===
def simple_attention(query, keys, values):
    """
    query: (d,) 查询向量
    keys: (n, d) 键向量
    values: (n, d_v) 值向量
    """
    # 计算注意力分数
    scores = query @ keys.T  # (n,)

    # 归一化得到权重
    weights = F.softmax(scores, dim=0)  # (n,)

    # 加权求和
    output = weights @ values  # (d_v,)

    return output, weights

# 使用
query = torch.randn(64)
keys = torch.randn(10, 64)
values = torch.randn(10, 128)

output, weights = simple_attention(query, keys, values)
print(f"Output shape: {output.shape}")  # (128,)
print(f"Weights sum: {weights.sum():.4f}")  # 1.0
```

### 2. 批量注意力

```python
import torch
import torch.nn.functional as F

def batch_attention(queries, keys, values, mask=None):
    """
    queries: (batch, seq_q, d)
    keys: (batch, seq_k, d)
    values: (batch, seq_k, d_v)
    mask: (batch, seq_q, seq_k) 或 None
    """
    d = queries.size(-1)

    # 计算注意力分数
    # (batch, seq_q, d) @ (batch, d, seq_k) -> (batch, seq_q, seq_k)
    scores = queries @ keys.transpose(-2, -1)

    # 缩放（防止 softmax 饱和）
    scores = scores / (d ** 0.5)

    # 应用 mask（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax
    weights = F.softmax(scores, dim=-1)

    # 加权求和
    # (batch, seq_q, seq_k) @ (batch, seq_k, d_v) -> (batch, seq_q, d_v)
    output = weights @ values

    return output, weights

# 示例
batch_size, seq_q, seq_k, d, d_v = 2, 5, 10, 64, 128

queries = torch.randn(batch_size, seq_q, d)
keys = torch.randn(batch_size, seq_k, d)
values = torch.randn(batch_size, seq_k, d_v)

output, weights = batch_attention(queries, keys, values)
print(f"Output shape: {output.shape}")  # (2, 5, 128)
print(f"Weights shape: {weights.shape}")  # (2, 5, 10)
```

### 3. 可视化注意力

```python
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

def visualize_attention(sentence1, sentence2, attention_weights):
    """
    可视化两个句子之间的注意力
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights.detach().numpy(),
        xticklabels=sentence2.split(),
        yticklabels=sentence1.split(),
        cmap='Blues',
        annot=True,
        fmt='.2f'
    )
    plt.xlabel('Keys (sentence 2)')
    plt.ylabel('Queries (sentence 1)')
    plt.title('Attention Weights')
    # plt.show()

# 模拟示例
sentence1 = "The cat sat"
sentence2 = "A cat is sitting on the mat"

# 模拟注意力权重
attention = torch.tensor([
    [0.1, 0.6, 0.1, 0.1, 0.0, 0.0, 0.1],  # The -> A, cat, is, sitting, on, the, mat
    [0.0, 0.8, 0.0, 0.1, 0.0, 0.0, 0.1],  # cat
    [0.0, 0.0, 0.0, 0.7, 0.1, 0.0, 0.2],  # sat
])

visualize_attention(sentence1, sentence2, attention)
```

## 深入理解 Deep Dive

### 为什么需要缩放？

```python
import torch
import torch.nn.functional as F

d = 512  # 维度

# 不缩放
q = torch.randn(d)
k = torch.randn(10, d)
scores_unscaled = q @ k.T
print(f"Unscaled scores std: {scores_unscaled.std():.2f}")  # 约 22

# 缩放后
scores_scaled = scores_unscaled / (d ** 0.5)
print(f"Scaled scores std: {scores_scaled.std():.2f}")  # 约 1

# 未缩放的 softmax 会过于尖锐
print(f"Unscaled softmax: {F.softmax(scores_unscaled, dim=0)}")  # 接近 one-hot
print(f"Scaled softmax: {F.softmax(scores_scaled, dim=0)}")  # 更平滑
```

### 不同类型的注意力

```
1. Self-Attention: Q=K=V 来自同一序列
   - 用于：理解序列内部关系

2. Cross-Attention: Q 来自一个序列，K=V 来自另一个
   - 用于：翻译、问答

3. Masked Attention: 使用 mask 隐藏某些位置
   - 用于：自回归生成（只看过去）
```

## 小结 Summary

- 注意力机制让模型动态关注输入的不同部分
- 核心公式：`Attention = softmax(QK^T/√d) × V`
- Q（查询）、K（键）、V（值）是三个核心组件
- 缩放因子 √d 防止 softmax 饱和

## 下一步 Next

[下一章：Self-Attention 数学推导与实现 →](./02-self-attention.md)
