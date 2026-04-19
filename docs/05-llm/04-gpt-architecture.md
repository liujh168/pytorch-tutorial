# GPT 架构详解 (GPT Architecture)

## 概述 Overview

GPT (Generative Pre-trained Transformer) 是最流行的自回归语言模型架构。

## 代码实现 Implementation

### 完整的 GPT 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTConfig:
    """GPT 配置"""
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        dropout=0.1
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果 mask
        mask = torch.tril(torch.ones(config.n_positions, config.n_positions))
        self.register_buffer('mask', mask.view(1, 1, config.n_positions, config.n_positions))

    def forward(self, x):
        B, T, C = x.size()

        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 分头
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享
        self.wte.weight = self.lm_head.weight

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.n_positions

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return logits, loss

# 测试
config = GPTConfig(vocab_size=50257, n_layer=6, n_head=6, n_embd=384)
model = GPT(config)

x = torch.randint(0, 50257, (2, 100))
logits, loss = model(x, x)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Logits: {logits.shape}")
```

## 深入理解：文本生成采样策略

```python
import torch
import torch.nn.functional as F

def greedy_decode(logits_last: torch.Tensor) -> int:
    """贪心解码：始终选择概率最高的 token"""
    return logits_last.argmax(dim=-1).item()


def temperature_sample(logits_last: torch.Tensor, temperature: float = 1.0) -> int:
    """温度采样：temperature < 1 更保守，> 1 更随机"""
    probs = F.softmax(logits_last / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


def top_k_sample(logits_last: torch.Tensor, k: int = 50) -> int:
    """Top-K 采样：只从概率最高的 k 个 token 中采样"""
    values, indices = torch.topk(logits_last, k)
    probs = F.softmax(values, dim=-1)
    chosen_idx = torch.multinomial(probs, 1).item()
    return indices[chosen_idx].item()


def top_p_sample(logits_last: torch.Tensor, p: float = 0.9) -> int:
    """Top-P (Nucleus) 采样：从概率累计超过 p 的最小 token 集中采样"""
    sorted_logits, sorted_indices = torch.sort(logits_last, descending=True)
    sorted_probs   = F.softmax(sorted_logits, dim=-1)
    cumulative_p   = sorted_probs.cumsum(dim=-1)
    # 保留累积概率超过 p 之前的 token
    cutoff_mask    = cumulative_p - sorted_probs > p
    sorted_logits[cutoff_mask] = float('-inf')
    probs  = F.softmax(sorted_logits, dim=-1)
    chosen = torch.multinomial(probs, 1).item()
    return sorted_indices[chosen].item()


# 对比演示（使用随机 logits）
torch.manual_seed(42)
vocab_size  = 100
fake_logits = torch.randn(vocab_size)
fake_logits[5]  = 5.0   # 模拟一个强烈偏好的 token
fake_logits[10] = 4.0

greedy_id    = greedy_decode(fake_logits)
temp_low_id  = temperature_sample(fake_logits, temperature=0.3)
temp_high_id = temperature_sample(fake_logits, temperature=2.0)
topk_id      = top_k_sample(fake_logits, k=10)
topp_id      = top_p_sample(fake_logits, p=0.9)

print("采样策略对比:")
print(f"  Greedy (确定性):    token={greedy_id}  (概率最高的 token)")
print(f"  Temperature=0.3:   token={temp_low_id}   (保守，集中)")
print(f"  Temperature=2.0:   token={temp_high_id}  (随机，发散)")
print(f"  Top-K (k=10):      token={topk_id}")
print(f"  Top-P (p=0.9):     token={topp_id}")

print("""
采样策略选择指南:
  贪心解码   → 代码生成、数学题（需要确定性）
  低温采样   → 事实性问答、指令遵循
  Top-P=0.9 → 创意写作、对话（GPT-3/4 默认）
  高温采样   → 诗歌创作、头脑风暴
""")
```

## 深入理解：GPT 参数量计算

```python
def calculate_gpt_params(vocab_size, n_positions, n_embd, n_layer, n_head):
    """精确计算 GPT 模型各组件的参数量"""
    # Token & Position Embedding
    token_emb  = vocab_size * n_embd
    pos_emb    = n_positions * n_embd

    # 每个 Transformer Block 的参数
    # - CausalSelfAttention: c_attn (3*d×d) + c_proj (d×d)
    attn_params = n_embd * (3 * n_embd) + n_embd * n_embd  # 4 * d^2
    # - MLP: c_fc (d×4d) + c_proj (4d×d)
    mlp_params  = n_embd * (4 * n_embd) + (4 * n_embd) * n_embd  # 8 * d^2
    # - LayerNorm x2: weight + bias
    ln_params   = 2 * (n_embd * 2)

    per_block   = attn_params + mlp_params + ln_params
    all_blocks  = n_layer * per_block

    # 最终 LayerNorm + LM Head（与 token_emb 共享权重，不重复计）
    final_ln    = n_embd * 2

    total = token_emb + pos_emb + all_blocks + final_ln

    print(f"{'组件':^20} {'参数量':>15}")
    print("-" * 37)
    print(f"{'Token Embedding':^20} {token_emb:>15,}")
    print(f"{'Position Embedding':^20} {pos_emb:>15,}")
    print(f"{'Attention (×layer)':^20} {attn_params:>15,}")
    print(f"{'MLP (×layer)':^20} {mlp_params:>15,}")
    print(f"{'LayerNorm (×layer)':^20} {ln_params:>15,}")
    print(f"{'所有 {n_layer} 层合计':^20} {all_blocks:>15,}")
    print("-" * 37)
    print(f"{'总参数量':^20} {total:>15,}  ({total/1e6:.1f}M)")
    return total


print("=== GPT-2 Small (124M) ===")
calculate_gpt_params(vocab_size=50257, n_positions=1024,
                     n_embd=768, n_layer=12, n_head=12)

print("\n=== 自定义小模型 ===")
calculate_gpt_params(vocab_size=5000, n_positions=256,
                     n_embd=128, n_layer=4, n_head=4)
```

## GPT 变体对比

| 模型 | 参数量 | 层数 | 维度 | 头数 | 上下文长度 |
|------|--------|------|------|------|-----------|
| GPT-2 Small | 124M | 12 | 768 | 12 | 1K |
| GPT-2 Large | 774M | 36 | 1280 | 20 | 1K |
| GPT-3 | 175B | 96 | 12288 | 96 | 2K |
| LLaMA 2 7B | 7B | 32 | 4096 | 32 | 4K |
| LLaMA 3 70B | 70B | 80 | 8192 | 64 | 8K |

## 下一步 Next

[下一章：LLM 训练技术 →](./05-training-llm.md)
