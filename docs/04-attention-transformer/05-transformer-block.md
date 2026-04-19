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

## 深入理解：Pre-LN vs Post-LN 训练稳定性对比

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pre-LN（现代 GPT 风格）vs Post-LN（原始 Transformer）
# Pre-LN 训练更稳定，原因：梯度可以通过残差路径绕过 LayerNorm 直接传播

class TransformerBlockPostLN(nn.Module):
    """Post-LN: 原始 Transformer 结构"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ff    = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention → Residual → Norm
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.drop(attn_out))
        # FFN → Residual → Norm
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class TransformerBlockPreLN(nn.Module):
    """Pre-LN: GPT-2 / GPT-3 风格（推荐）"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ff    = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Norm → Attention → Residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.drop(attn_out)
        # Norm → FFN → Residual
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# 比较两种架构的梯度范数（Pre-LN 梯度更稳定）
d_model, n_heads, d_ff = 64, 4, 256
x = torch.randn(4, 20, d_model, requires_grad=False)
y_target = torch.randn(4, 20, d_model)

for name, block in [('Post-LN', TransformerBlockPostLN(d_model, n_heads, d_ff)),
                    ('Pre-LN',  TransformerBlockPreLN(d_model, n_heads, d_ff))]:
    x_in = x.clone().detach().requires_grad_(True)
    out  = block(x_in)
    loss = F.mse_loss(out, y_target)
    loss.backward()
    grad_norm = x_in.grad.norm().item()
    print(f"{name}: 输入梯度范数 = {grad_norm:.4f}")
```

## 深入理解：完整的 GPT 风格解码器

```python
import torch
import torch.nn as nn
import math

class GPTDecoderBlock(nn.Module):
    """GPT 风格解码器块（仅 Self-Attention，使用因果掩码）"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        S = x.size(1)
        # 因果掩码：当前位置只能看到之前的位置
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    """极简 GPT 实现"""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=512, max_len=256, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.blocks    = nn.ModuleList([
            GPTDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重绑定：输出层与嵌入层共享权重（常见优化）
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, tokens):  # tokens: (B, S)
        B, S = tokens.shape
        pos  = torch.arange(S, device=tokens.device).unsqueeze(0)
        x    = self.token_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))  # (B, S, vocab_size)


# 测试
VOCAB, MAX_LEN = 5000, 64
gpt = MiniGPT(vocab_size=VOCAB, max_len=MAX_LEN)
tokens = torch.randint(0, VOCAB, (2, 32))
logits = gpt(tokens)
print(f"输入 tokens: {tokens.shape}")
print(f"输出 logits: {logits.shape}  (B, S, vocab_size)")
print(f"参数量: {sum(p.numel() for p in gpt.parameters()):,}")
```

## 架构变体

| 变体 | 特点 | 使用模型 |
|------|------|----------|
| Post-LN | 原始架构，训练较难 | 原始 Transformer |
| Pre-LN | 训练更稳定 | GPT-2, GPT-3 |
| Sandwich-LN | 结合两者优点 | 一些变体 |
| RMSNorm | 计算更高效，无减法归一化 | LLaMA, Mistral |
| SwiGLU FFN | 替换 GELU，性能更好 | LLaMA 2/3 |

## 下一步 Next

[下一章：Encoder-Decoder 架构 →](./06-encoder-decoder.md)
