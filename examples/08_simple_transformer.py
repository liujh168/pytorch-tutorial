"""
示例 08: 完整 Transformer Block（含位置编码）
对应文档: docs/04-attention-transformer/05-transformer-block.md
运行方式: python examples/08_simple_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"使用设备: {device}\n")


# ── 1. 正弦位置编码 (Sinusoidal Positional Encoding) ─────────
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # 注册为 buffer（不参与训练）
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── 2. 前馈网络 (Feed-Forward Network) ───────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),           # 现代 Transformer 偏好 GELU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 3. Transformer 编码器层 ───────────────────────────────────
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.ff        = FeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN 结构（更稳定的现代写法）
        # 自注意力 + 残差连接
        residual = x
        x_norm   = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm,
                                     key_padding_mask=src_key_padding_mask)
        x = residual + self.dropout(attn_out)

        # 前馈 + 残差连接
        x = x + self.ff(self.norm2(x))
        return x


# ── 4. 完整 Transformer 编码器 ────────────────────────────────
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int    = 128,
        n_heads: int    = 4,
        n_layers: int   = 3,
        d_ff: int       = 512,
        max_len: int    = 128,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm      = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,                   # (batch, seq_len) 整数 token id
        padding_mask: torch.Tensor | None = None,  # True 表示该位置是 padding
    ) -> torch.Tensor:
        x = self.pos_enc(self.embedding(tokens))
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        return self.norm(x)


# ── 5. 测试运行 ───────────────────────────────────────────────
print("=== Transformer 编码器测试 ===")
VOCAB  = 1000
BATCH  = 4
SEQ    = 20
D_MODEL = 128

model = TransformerEncoder(vocab_size=VOCAB, d_model=D_MODEL, n_heads=4, n_layers=3).to(device)

tokens  = torch.randint(1, VOCAB, (BATCH, SEQ)).to(device)
# 最后 3 个 token 是 padding（id=0）
tokens[:, -3:] = 0
pad_mask = (tokens == 0)   # True = padding 位置

output = model(tokens, padding_mask=pad_mask)
print(f"输入 tokens shape:  {tokens.shape}")
print(f"输出 hidden shape:  {output.shape}  (batch, seq, d_model)")

params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {params:,}")


# ── 6. 分类头 ─────────────────────────────────────────────────
print("\n=== 接分类头（文本分类）===")

class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 128):
        super().__init__()
        self.encoder    = TransformerEncoder(vocab_size, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(tokens)
        # 取 [CLS] token（第 0 位）的向量做分类
        cls_vec = hidden[:, 0]
        return self.classifier(cls_vec)


clf = TextClassifier(VOCAB, num_classes=5).to(device)
logits = clf(tokens)
print(f"分类 logits shape: {logits.shape}  (batch=4, classes=5)")

print("\n✓ 示例 08 运行完成")
