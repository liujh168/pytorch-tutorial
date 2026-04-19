# 语言模型目标函数 (Language Modeling Objectives)

## 概述 Overview

语言模型的核心是预测下一个 token。本章介绍不同类型的语言模型目标。

## 代码实现 Implementation

### 1. Causal Language Modeling (CLM)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_lm_loss(logits, targets, ignore_index=-100):
    """
    自回归语言模型损失
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    """
    # 移位：预测下一个 token
    # 输入: [A, B, C, D]
    # 目标: [B, C, D, E]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()

    # 展平计算交叉熵
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )

    return loss

# 示例
batch_size, seq_len, vocab_size = 2, 100, 32000
logits = torch.randn(batch_size, seq_len, vocab_size)
targets = torch.randint(0, vocab_size, (batch_size, seq_len))

loss = causal_lm_loss(logits, targets)
print(f"CLM Loss: {loss.item():.4f}")
print(f"Perplexity: {torch.exp(loss).item():.2f}")
```

### 2. Masked Language Modeling (MLM)

```python
import torch
import torch.nn as nn
import random

class MLMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        return self.decoder(x)

def create_mlm_inputs(tokens, vocab_size, mask_token_id, mlm_prob=0.15):
    """创建 MLM 训练数据"""
    labels = tokens.clone()
    input_ids = tokens.clone()

    # 随机选择 15% 的位置
    probability_matrix = torch.full(tokens.shape, mlm_prob)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 80% 替换为 [MASK]
    indices_replaced = torch.bernoulli(torch.full(tokens.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10% 替换为随机 token
    indices_random = torch.bernoulli(torch.full(tokens.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, tokens.shape)
    input_ids[indices_random] = random_words[indices_random]

    # 10% 保持不变

    # 只计算 masked 位置的损失
    labels[~masked_indices] = -100

    return input_ids, labels

# 使用
tokens = torch.randint(0, 32000, (2, 100))
input_ids, labels = create_mlm_inputs(tokens, vocab_size=32000, mask_token_id=103)
```

### 3. 困惑度 (Perplexity)

```python
import torch
import math

def calculate_perplexity(model, dataloader, device='cuda'):
    """计算模型困惑度"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum',
                ignore_index=-100
            )

            total_loss += loss.item()
            total_tokens += (shift_labels != -100).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity

# 困惑度越低越好
# 典型值：
# - 随机模型: vocab_size (如 32000)
# - 好的模型: 10-50
# - 非常好的模型: < 10
```

## 深入理解：字符级语言模型完整训练示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 字符级语言模型（不依赖外部数据集，开箱即用）
TEXT = """
深度学习是机器学习的一个分支，它基于人工神经网络进行学习。
深度学习模型能够自动从数据中学习特征表示。
PyTorch 是一个灵活的深度学习框架，广泛用于研究和工业界。
Transformer 架构改变了自然语言处理的面貌。
GPT 系列模型展示了大语言模型的强大能力。
""" * 20   # 重复以增加数据量

# 构建字符词汇表
chars   = sorted(set(TEXT))
stoi    = {c: i for i, c in enumerate(chars)}
itos    = {i: c for i, c in enumerate(chars)}
VOCAB   = len(chars)
print(f"词汇表大小: {VOCAB} 个字符")

# 编码
data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)

# 数据集
SEQ_LEN  = 32
BATCH    = 16
DEVICE   = torch.device('cpu')

def get_batch(data, seq_len, batch_size):
    idx = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x   = torch.stack([data[i:i+seq_len]   for i in idx])
    y   = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)


# 极简 Transformer 语言模型
class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks  = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # 权重绑定

    def forward(self, x):
        B, S = x.shape
        pos  = torch.arange(S, device=x.device).unsqueeze(0)
        h    = self.tok_emb(x) + self.pos_emb(pos)
        # 因果掩码
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        for block in self.blocks:
            h = block(h, src_mask=causal_mask)
        return self.head(self.norm(h))


model     = TinyLM(VOCAB).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
params    = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {params:,}")

# 训练
print("\n开始训练...")
for step in range(500):
    xb, yb = get_batch(data, SEQ_LEN, BATCH)
    model.train()
    logits = model(xb)
    loss   = F.cross_entropy(logits.view(-1, VOCAB), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0:
        ppl = math.exp(loss.item())
        print(f"Step {step:4d}: loss={loss.item():.4f}, perplexity={ppl:.2f}")

# 文本生成
def generate(model, start_text: str, n_chars: int = 50, temperature: float = 0.8):
    model.eval()
    context = torch.tensor([stoi.get(c, 0) for c in start_text], dtype=torch.long).unsqueeze(0)
    result  = start_text
    with torch.no_grad():
        for _ in range(n_chars):
            logits = model(context[:, -SEQ_LEN:])
            probs  = F.softmax(logits[0, -1] / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            result  += itos[next_id]
            context  = torch.cat([context, torch.tensor([[next_id]])], dim=1)
    return result

print(f"\n生成文本:\n{generate(model, '深度学习', n_chars=60)}")
```

## 深入理解：困惑度的直觉理解

```python
import torch, math

# 困惑度 = 2^(平均每个 token 的比特数) = exp(平均 NLL loss)
# 直觉：模型平均在多少个候选中"困惑"地选择
# - PPL=2:     模型非常确定，只在2个选项中选
# - PPL=100:   模型相当困惑，相当于在100个等概率选项中随机选
# - PPL=词汇量: 等同于完全随机模型（最差情况）

def loss_to_ppl(loss_value: float) -> float:
    return math.exp(loss_value)

# 不同 loss 对应的困惑度
print("Loss  →  Perplexity  →  含义")
print("-" * 50)
for loss in [0.1, 0.5, 1.0, 2.0, 3.0, 4.6, 10.0]:
    ppl = loss_to_ppl(loss)
    if ppl < 5:      meaning = "极好（接近完美）"
    elif ppl < 20:   meaning = "很好"
    elif ppl < 100:  meaning = "一般"
    elif ppl < 1000: meaning = "较差"
    else:            meaning = "接近随机"
    print(f"{loss:5.1f}  →  {ppl:10.1f}  →  {meaning}")

print(f"\n随机模型基线 (vocab=32000): PPL={32000}")
```

## 下一步 Next

[下一章：GPT 架构详解 →](./04-gpt-architecture.md)
