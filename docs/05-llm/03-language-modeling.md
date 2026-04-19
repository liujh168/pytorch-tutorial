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

## 下一步 Next

[下一章：GPT 架构详解 →](./04-gpt-architecture.md)
