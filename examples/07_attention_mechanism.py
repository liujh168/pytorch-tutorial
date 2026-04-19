"""
示例 07: 注意力机制手动实现
对应文档: docs/04-attention-transformer/02-self-attention.md
运行方式: python examples/07_attention_mechanism.py
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


# ── 1. Scaled Dot-Product Attention ───────────────────────────
print("=== 1. Scaled Dot-Product Attention ===")

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q, K, V: (batch, seq_len, d_k)
    返回: (输出, 注意力权重)
    """
    d_k = Q.size(-1)
    # 计算注意力分数并缩放
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 可选的掩码（如因果掩码、padding 掩码）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output       = torch.matmul(attn_weights, V)
    return output, attn_weights


batch, seq_len, d_k = 2, 5, 16
Q = torch.randn(batch, seq_len, d_k).to(device)
K = torch.randn(batch, seq_len, d_k).to(device)
V = torch.randn(batch, seq_len, d_k).to(device)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Q/K/V shape: {Q.shape}")
print(f"输出 shape:  {output.shape}")
print(f"注意力权重 shape: {weights.shape}")
print(f"权重行和（应为1）: {weights[0, 0].sum().item():.6f}")


# ── 2. 因果掩码 (Causal Mask) ─────────────────────────────────
print("\n=== 2. 因果掩码（用于语言模型）===")

def make_causal_mask(seq_len: int) -> torch.Tensor:
    """下三角矩阵，位置 i 只能看到 j <= i 的位置"""
    return torch.tril(torch.ones(seq_len, seq_len))


mask = make_causal_mask(seq_len).to(device)
print(f"因果掩码 ({seq_len}×{seq_len}):")
print(mask.int())

# 使用因果掩码
output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=mask)
print(f"\n使用因果掩码后，未来位置权重（应趋近0）:")
print(f"  weights[0,0] = {weights_causal[0, 0].detach().cpu()}")


# ── 3. 多头注意力 (Multi-Head Attention) ─────────────────────
print("\n=== 3. 多头注意力 (Multi-Head Attention) ===")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须整除 n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape

        Q = self.split_heads(self.W_q(x))   # (B, H, S, d_k)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # 注意力（对每个头独立计算）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))

        # 合并多头
        out = torch.matmul(attn, V)                          # (B, H, S, d_k)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)  # (B, S, d_model)
        return self.W_o(out), attn


d_model, n_heads = 64, 8
mha = MultiHeadAttention(d_model, n_heads).to(device)

x = torch.randn(2, seq_len, d_model).to(device)
out, attn_weights = mha(x)
print(f"输入 shape:    {x.shape}")
print(f"输出 shape:    {out.shape}")
print(f"注意力权重 shape: {attn_weights.shape}  (batch, heads, seq, seq)")

# 参数量
params = sum(p.numel() for p in mha.parameters())
print(f"多头注意力参数量: {params:,}")


# ── 4. 与 PyTorch 内置对比 ───────────────────────────────────
print("\n=== 4. 与 nn.MultiheadAttention 对比 ===")
builtin_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True).to(device)
out_builtin, _ = builtin_mha(x, x, x)
print(f"内置 MHA 输出 shape: {out_builtin.shape}")
print(f"自定义 MHA 输出 shape: {out.shape}  (形状一致 ✓)")

print("\n✓ 示例 07 运行完成")
