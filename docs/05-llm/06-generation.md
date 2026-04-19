# 文本生成策略 (Text Generation)

## 概述 Overview

文本生成是 LLM 的核心应用。本章介绍各种采样和解码策略。

## 代码实现 Implementation

### 1. 贪婪解码

```python
import torch
import torch.nn.functional as F

def greedy_decode(model, input_ids, max_length, eos_token_id):
    """贪婪解码：每步选择概率最高的 token"""
    model.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            outputs = model(generated)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

    return generated
```

### 2. 采样方法

```python
import torch
import torch.nn.functional as F

def sample_with_temperature(logits, temperature=1.0):
    """温度采样"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def top_k_sampling(logits, k=50, temperature=1.0):
    """Top-k 采样"""
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    probs = F.softmax(top_k_logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices.gather(-1, idx)

def top_p_sampling(logits, p=0.9, temperature=1.0):
    """Top-p (Nucleus) 采样"""
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # 移除累积概率超过 p 的 token
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')

    probs = F.softmax(sorted_logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)

    return sorted_indices.gather(-1, idx)
```

### 3. 完整的生成函数

```python
import torch
import torch.nn.functional as F

def generate(
    model,
    input_ids,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    eos_token_id=None,
    pad_token_id=None
):
    """灵活的文本生成函数"""
    model.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 获取 logits
            outputs = model(generated)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            next_token_logits = logits[:, -1, :]

            if do_sample:
                # 应用温度
                next_token_logits = next_token_logits / temperature

                # Top-k 过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # 检查 EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    return generated

# 使用示例
# generated = generate(
#     model,
#     input_ids,
#     max_new_tokens=100,
#     temperature=0.8,
#     top_k=50,
#     top_p=0.95,
#     do_sample=True
# )
```

### 4. Beam Search

```python
import torch
import torch.nn.functional as F

def beam_search(model, input_ids, max_length, num_beams=5, eos_token_id=None):
    """Beam Search 解码"""
    model.eval()
    batch_size = input_ids.size(0)
    device = input_ids.device

    # 初始化 beam
    beam_scores = torch.zeros(batch_size, num_beams, device=device)
    beam_scores[:, 1:] = -1e9  # 除第一个 beam 外都设为极小值

    beam_seqs = input_ids.unsqueeze(1).repeat(1, num_beams, 1)

    with torch.no_grad():
        for step in range(max_length - input_ids.size(1)):
            # 展平 beam
            flat_seqs = beam_seqs.view(batch_size * num_beams, -1)

            outputs = model(flat_seqs)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            next_token_logits = logits[:, -1, :]

            vocab_size = next_token_logits.size(-1)
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            log_probs = log_probs.view(batch_size, num_beams, -1)

            # 计算所有候选的分数
            next_scores = beam_scores.unsqueeze(-1) + log_probs
            next_scores = next_scores.view(batch_size, -1)

            # 选择 top-k
            next_scores, next_indices = torch.topk(next_scores, num_beams, dim=-1)

            beam_indices = next_indices // vocab_size
            token_indices = next_indices % vocab_size

            # 更新 beam
            beam_seqs = beam_seqs.gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, beam_seqs.size(-1)))
            beam_seqs = torch.cat([beam_seqs, token_indices.unsqueeze(-1)], dim=-1)
            beam_scores = next_scores

    # 返回最佳 beam
    return beam_seqs[:, 0, :]
```

## 生成参数建议

| 场景 | temperature | top_k | top_p |
|------|-------------|-------|-------|
| 代码生成 | 0.2-0.5 | 10-40 | 0.9 |
| 创意写作 | 0.8-1.2 | 50-100 | 0.95 |
| 问答 | 0.3-0.7 | 20-50 | 0.9 |
| 翻译 | 0.1-0.3 | - | - |

## 下一步 Next

恭喜完成 LLM 篇！接下来学习模型部署。

[下一章：模型导出 →](../06-deployment/01-model-export.md)
