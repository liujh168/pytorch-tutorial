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

## 深入理解：混合精度训练实战（CPU 兼容版）

```python
import torch
import torch.nn as nn

def demo_mixed_precision():
    """
    混合精度训练原理演示
    - float16: 速度快，范围小 (±65504)，可能溢出
    - float32: 速度慢，范围大，用于参数存储和梯度累积
    - GradScaler: 将 loss 放大避免 float16 下溢，更新前再缩小
    """
    model = nn.Sequential(
        nn.Linear(256, 1024), nn.ReLU(),
        nn.Linear(1024, 256), nn.ReLU(),
        nn.Linear(256, 10),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 检测可用设备和精度支持
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'   # AMP 在 CPU 上无意义
    model   = model.to(device)

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("使用 CUDA AMP (float16 混合精度)")
    else:
        print("使用 CPU (float32 全精度)")

    X = torch.randn(64, 256).to(device)
    y = torch.randint(0, 10, (64,)).to(device)

    for step in range(3):
        optimizer.zero_grad()

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(X)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            # 梯度裁剪（需要先 unscale）
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print(f"Step {step}: loss={loss.item():.4f}")

demo_mixed_precision()
```

## 深入理解：梯度累积与梯度裁剪组合使用

```python
import torch
import torch.nn as nn

def train_with_accumulation_and_clipping(
    model, dataloader, optimizer,
    accumulation_steps=4, max_grad_norm=1.0
):
    """
    梯度累积 + 梯度裁剪的正确组合方式
    注意：clip_grad_norm_ 应在 accumulation 结束后、optimizer.step() 前调用
    """
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()

    for step, (X, y) in enumerate(dataloader):
        logits = model(X)
        # loss 除以累积步数，等效于对 batch 内样本求平均
        loss = criterion(logits, y) / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            # 所有 mini-batch 梯度累积完毕后再裁剪
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            print(f"更新步 {(step+1)//accumulation_steps}: "
                  f"grad_norm={total_norm:.3f}, loss={loss.item()*accumulation_steps:.4f}")

# 注意事项：
# 1. loss 必须除以 accumulation_steps（否则等效 loss 偏大）
# 2. 梯度裁剪在累积结束后（否则每个 mini-batch 都裁剪，等效于更小的裁剪阈值）
# 3. 有效 batch_size = batch_size * accumulation_steps * num_gpus
```

## 小结 Summary

| 技术 | 作用 | 使用场景 |
|------|------|----------|
| 梯度裁剪 | 防止梯度爆炸 | RNN/Transformer |
| 梯度累积 | 模拟大 batch | 显存不足时 |
| 混合精度 | 加速训练 | CUDA GPU 训练 |
| 梯度检查点 | 节省显存 | 大模型训练 |

## 下一步 Next

[下一章：GPU 训练与多卡并行 →](./05-gpu-training.md)
