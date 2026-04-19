# 学习率调度策略 (Learning Rate Schedulers)

## 概述 Overview

学习率是深度学习中最重要的超参数之一。合适的学习率调度策略可以显著提高训练效果和收敛速度。

**难度级别**：⭐ 重点章节

## 代码实现 Implementation

### 1. 基础调度器

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR,
    CosineAnnealingLR, ReduceLROnPlateau
)

model = torch.nn.Linear(10, 5)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === StepLR ===
# 每 step_size 个 epoch 学习率乘以 gamma
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# epoch 0-29: lr=0.001, epoch 30-59: lr=0.0001, ...

# === MultiStepLR ===
# 在指定的 epoch 降低学习率
scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
# epoch 0-29: lr=0.001, epoch 30-79: lr=0.0001, epoch 80+: lr=0.00001

# === ExponentialLR ===
# 每个 epoch 学习率乘以 gamma
scheduler = ExponentialLR(optimizer, gamma=0.95)

# === CosineAnnealingLR ===
# 余弦退火：学习率从初始值平滑降低到 eta_min
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,     # 半周期长度
    eta_min=1e-6   # 最小学习率
)

# === ReduceLROnPlateau ===
# 当指标不再改善时降低学习率（需要传入指标值）
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',        # 'min' 或 'max'
    factor=0.1,        # 学习率乘以的因子
    patience=10,       # 等待多少个 epoch
    threshold=1e-4,    # 改善的最小阈值
    min_lr=1e-7        # 最小学习率
)

# 使用方式
for epoch in range(100):
    train_loss = train_epoch(...)
    val_loss = validate(...)

    # ReduceLROnPlateau 需要传入指标
    scheduler.step(val_loss)

    # 其他调度器直接 step
    # scheduler.step()
```

### 2. Warmup 策略

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, LinearLR, ConstantLR

# === Linear Warmup ===
def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """线性 warmup + 线性衰减"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        return max(0.0, (total_steps - current_step) / (total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)

# === Warmup + Cosine Decay（常用于 Transformer）===
import math

def get_cosine_with_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """Warmup + 余弦衰减"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

# === 使用 PyTorch 内置 ===
# LinearLR: 线性变化
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,  # 起始倍率
    end_factor=1.0,    # 结束倍率
    total_iters=1000   # warmup 步数
)

# 组合多个调度器
from torch.optim.lr_scheduler import SequentialLR

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, CosineAnnealingLR(optimizer, T_max=9000)],
    milestones=[1000]  # 切换点
)
```

### 3. One Cycle 策略

```python
from torch.optim.lr_scheduler import OneCycleLR

# One Cycle: 学习率先升后降，同时动量先降后升
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,           # 最大学习率
    total_steps=10000,     # 总步数
    pct_start=0.3,         # warmup 占比
    anneal_strategy='cos', # 'cos' 或 'linear'
    div_factor=25,         # 初始 lr = max_lr / div_factor
    final_div_factor=1e4   # 最终 lr = max_lr / final_div_factor
)

# 每个 batch 后调用
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()  # 每个 batch 更新
```

### 4. 可视化学习率曲线

```python
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

def visualize_scheduler(scheduler_fn, name, total_steps=1000):
    model = torch.nn.Linear(10, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = scheduler_fn(optimizer)

    lrs = []
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    plt.plot(lrs, label=name)

plt.figure(figsize=(12, 6))

# 比较不同调度器
visualize_scheduler(
    lambda opt: CosineAnnealingLR(opt, T_max=1000),
    'CosineAnnealing'
)
visualize_scheduler(
    lambda opt: OneCycleLR(opt, max_lr=0.01, total_steps=1000),
    'OneCycle'
)

plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.legend()
plt.title('Learning Rate Schedulers Comparison')
# plt.show()
```

## 小结 Summary

| 调度器 | 适用场景 | 特点 |
|--------|----------|------|
| StepLR | 简单基线 | 阶梯下降 |
| CosineAnnealingLR | CV/NLP | 平滑变化 |
| OneCycleLR | 快速训练 | 学习率先升后降 |
| ReduceLROnPlateau | 自适应 | 根据指标调整 |
| Warmup + Cosine | Transformer | 防止初期震荡 |

## 下一步 Next

[下一章：正则化技术 →](./03-regularization.md)
