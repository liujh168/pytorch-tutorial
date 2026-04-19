# Encoder-Decoder 架构 (Encoder-Decoder Architecture)

## 概述 Overview

Encoder-Decoder 是 Transformer 的完整架构，用于序列到序列任务如机器翻译。

## 代码实现 Implementation

### 完整的 Transformer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    """完整的 Encoder-Decoder Transformer"""

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_encoder_layers=6,
        n_decoder_layers=6,
        dropout=0.1,
        max_len=512
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_len, d_model)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask=None):
        """编码源序列"""
        seq_len = src.size(1)
        positions = torch.arange(seq_len, device=src.device).unsqueeze(0)

        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = x + self.pos_encoding(positions)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return self.norm(x)

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """解码目标序列"""
        seq_len = tgt.size(1)
        positions = torch.arange(seq_len, device=tgt.device).unsqueeze(0)

        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = x + self.pos_encoding(positions)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)

        return self.norm(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """完整的前向传播"""
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask)
        return self.output_projection(output)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # Feed-forward with residual
        x = x + self.dropout2(self.ff(x))
        x = self.norm2(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # Masked self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention
        attn_out, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_mask)
        x = self.norm2(x + self.dropout(attn_out))

        # Feed-forward
        x = self.norm3(x + self.dropout(self.ff(x)))

        return x

# 测试
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    n_encoder_layers=6,
    n_decoder_layers=6
)

src = torch.randint(0, 10000, (2, 20))
tgt = torch.randint(0, 10000, (2, 15))

# 创建因果 mask
tgt_mask = torch.triu(torch.ones(15, 15), diagonal=1).bool()

output = model(src, tgt, tgt_mask=tgt_mask)
print(f"Output shape: {output.shape}")  # (2, 15, 10000)
```

## Encoder-Only vs Decoder-Only vs Encoder-Decoder

| 架构 | 任务类型 | 代表模型 |
|------|----------|----------|
| Encoder-Only | 分类、理解 | BERT |
| Decoder-Only | 生成 | GPT |
| Encoder-Decoder | 翻译、摘要 | T5, BART |

## 下一步 Next

[下一章：高效注意力机制 →](./07-efficient-attention.md)
