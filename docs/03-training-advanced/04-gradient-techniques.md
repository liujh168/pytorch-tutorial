# 梯度技巧 (Gradient Techniques)

## 概述 Overview

掌握梯度相关的技巧对于训练深度模型至关重要，包括梯度裁剪、梯度累积和混合精度训练。

**难度级别**：⭐ 重点章节

## 代码实现 Implementation

### 1. 梯度裁剪 (Gradient Clipping)

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)
optimizer = torch.optim.Adam(model.parameters())

# === 按范数裁剪（推荐）===
# 将梯度的总范数限制在 max_norm 以内
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# === 按值裁剪 ===
# 将每个梯度元素限制在 [-clip_value, clip_value]
loss.backward()
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
optimizer.step()

# === 完整示例 ===
def train_with_gradient_clipping(model, dataloader, optimizer, criterion, max_grad_norm=1.0):
    model.train()
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # 裁剪前检查梯度范数
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        print(f"Gradient norm: {total_norm:.4f}")

        optimizer.step()
```

### 2. 梯度累积 (Gradient Accumulation)

```python
import torch

def train_with_accumulation(model, dataloader, optimizer, criterion,
                           accumulation_steps=4, device='cuda'):
    """
    梯度累积：用多个小 batch 模拟大 batch
    有效 batch_size = batch_size * accumulation_steps
    """
    model.train()
    optimizer.zero_grad()

    for i, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 归一化损失
        loss = loss / accumulation_steps
        loss.backward()

        # 每 accumulation_steps 步更新一次
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # 处理剩余的梯度
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 混合精度训练 (Mixed Precision Training)

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(model, dataloader, optimizer, criterion, device='cuda'):
    """
    混合精度训练：
    - 前向传播使用 float16（更快、更省内存）
    - 损失计算和参数更新使用 float32（保持精度）
    """
    model.train()
    scaler = GradScaler()

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # 自动混合精度
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # 缩放损失并反向传播
        scaler.scale(loss).backward()

        # 取消缩放并更新
        scaler.step(optimizer)
        scaler.update()

# === 结合梯度累积 ===
def train_amp_with_accumulation(model, dataloader, optimizer, criterion,
                                accumulation_steps=4, device='cuda'):
    model.train()
    scaler = GradScaler()
    optimizer.zero_grad()

    for i, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        with autocast():
            output = model(data)
            loss = criterion(output, target) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

### 4. 梯度检查点 (Gradient Checkpointing)

```python
import torch
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(torch.nn.Module):
    """
    梯度检查点：用计算换内存
    不保存中间激活，反向传播时重新计算
    """
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1024, 1024)
        self.layer2 = torch.nn.Linear(1024, 1024)
        self.layer3 = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        # 使用 checkpoint 包装，不保存中间激活
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = torch.relu(x)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        x = torch.relu(x)
        x = self.layer3(x)
        return x

# 内存节省约 30-50%，但训练时间增加约 20-30%
```

### 5. 梯度监控

```python
import torch

def monitor_gradients(model):
    """监控梯度统计信息"""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm

            # 检测异常
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
            if grad_norm > 100:
                print(f"Large gradient in {name}: {grad_norm:.4f}")

    return grad_norms

# 使用钩子自动监控
def gradient_hook(name):
    def hook(grad):
        print(f"{name}: grad norm = {grad.norm():.4f}")
        return grad
    return hook

for name, param in model.named_parameters():
    param.register_hook(gradient_hook(name))
```

## 小结 Summary

| 技术 | 作用 | 使用场景 |
|------|------|----------|
| 梯度裁剪 | 防止梯度爆炸 | RNN/Transformer |
| 梯度累积 | 模拟大 batch | 显存不足时 |
| 混合精度 | 加速训练 | GPU 训练 |
| 梯度检查点 | 节省显存 | 大模型训练 |

## 下一步 Next

[下一章：GPU 训练与多卡并行 →](./05-gpu-training.md)
