# LLM 训练技术 (Training LLMs)

## 概述 Overview

训练大型语言模型需要特殊的技术和大量计算资源。本章介绍预训练和微调的关键技术。

## 代码实现 Implementation

### 1. 预训练数据准备

```python
import torch
from torch.utils.data import Dataset, DataLoader

class PretrainDataset(Dataset):
    """预训练数据集"""

    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据
        with open(data_path, 'r') as f:
            self.texts = f.readlines()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 编码
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()

        # 对于 CLM，labels = input_ids
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # 忽略 padding

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
```

### 2. 微调技术

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# === 全参数微调 ===
def full_finetune(model, train_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

# === LoRA 微调 ===
class LoRALayer(nn.Module):
    """Low-Rank Adaptation"""

    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # 低秩分解
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) / rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.scaling = alpha / rank

    def forward(self, x):
        # ΔW = BA * scaling
        return (x @ self.lora_A @ self.lora_B) * self.scaling

def add_lora_to_linear(linear_layer, rank=8):
    """给 Linear 层添加 LoRA"""
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features

    lora = LoRALayer(in_features, out_features, rank)

    class LinearWithLoRA(nn.Module):
        def __init__(self, linear, lora):
            super().__init__()
            self.linear = linear
            self.lora = lora
            # 冻结原始权重
            for param in self.linear.parameters():
                param.requires_grad = False

        def forward(self, x):
            return self.linear(x) + self.lora(x)

    return LinearWithLoRA(linear_layer, lora)

# === 量化微调 (QLoRA) ===
# 使用 bitsandbytes 库实现 4-bit 量化
# pip install bitsandbytes

# from transformers import BitsAndBytesConfig
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-7b",
#     quantization_config=bnb_config
# )
```

### 3. 训练优化技术

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train_with_optimizations(
    model,
    train_loader,
    optimizer,
    num_epochs,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0
):
    """包含各种优化的训练循环"""
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # 混合精度
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            # 梯度累积
            if (step + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
```

## 训练规模参考

| 模型规模 | 训练 Tokens | GPU 需求 |
|----------|-------------|----------|
| 7B | 1-2T | 8x A100 |
| 13B | 1-2T | 16x A100 |
| 70B | 1.4T | 64+ A100 |

## 下一步 Next

[下一章：文本生成策略 →](./06-generation.md)
