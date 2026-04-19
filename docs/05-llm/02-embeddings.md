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

## 下一步 Next

[下一章：语言模型目标函数 →](./03-language-modeling.md)
