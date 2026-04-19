# 优化器深度对比 (Optimizers Deep Dive)

## 概述 Overview

优化器决定了模型参数如何根据梯度进行更新。选择合适的优化器对训练效率和最终性能有重要影响。本章将深入对比各种优化器的原理和适用场景。

**难度级别**：⭐ 重点章节

## 前置知识 Prerequisites

- [04-loss-functions](../02-core/04-loss-functions.md) - 损失函数详解
- 梯度下降的基本概念

## 核心概念 Core Concepts

### 优化器的数学基础

```
基本梯度下降：θ = θ - η * ∇L(θ)

其中：
- θ: 模型参数
- η: 学习率
- ∇L(θ): 损失函数对参数的梯度
```

## 代码实现 Implementation

### 1. SGD 及其变体

```python
import torch
import torch.optim as optim

model = torch.nn.Linear(10, 5)

# === 基础 SGD ===
# θ = θ - lr * g
optimizer = optim.SGD(model.parameters(), lr=0.01)

# === SGD with Momentum ===
# v = momentum * v + g
# θ = θ - lr * v
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9  # 动量因子
)

# === SGD with Nesterov Momentum ===
# "先看后跳"策略，通常更快收敛
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True
)

# === 带权重衰减的 SGD ===
# L2 正则化的等效形式
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4  # L2 正则化系数
)
```

### 2. Adam 系列

```python
import torch.optim as optim

# === Adam ===
# 结合动量和自适应学习率
# m = β1 * m + (1-β1) * g           # 一阶矩估计
# v = β2 * v + (1-β2) * g²          # 二阶矩估计
# m_hat = m / (1 - β1^t)            # 偏差校正
# v_hat = v / (1 - β2^t)
# θ = θ - lr * m_hat / (√v_hat + ε)
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,         # 学习率（默认 0.001）
    betas=(0.9, 0.999),  # β1, β2
    eps=1e-8,         # 数值稳定项
    weight_decay=0    # L2 正则化
)

# === AdamW ===
# 解耦的权重衰减（推荐用于 Transformer）
# Adam 的 weight_decay 是 L2 正则化
# AdamW 的 weight_decay 是真正的权重衰减
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01  # 权重衰减系数
)

# === Amsgrad ===
# 解决 Adam 在某些情况下不收敛的问题
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    amsgrad=True  # 使用 Amsgrad 变体
)

# === NAdam ===
# Nesterov + Adam
optimizer = optim.NAdam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999)
)

# === RAdam ===
# 带自动热身的 Adam
optimizer = optim.RAdam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999)
)
```

### 3. 其他优化器

```python
import torch.optim as optim

# === RMSprop ===
# 自适应学习率，适合 RNN
# v = α * v + (1-α) * g²
# θ = θ - lr * g / √(v + ε)
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,  # 平滑因子
    eps=1e-8,
    momentum=0,
    weight_decay=0
)

# === Adagrad ===
# 自适应学习率，适合稀疏特征
# 问题：学习率会越来越小
optimizer = optim.Adagrad(
    model.parameters(),
    lr=0.01,
    lr_decay=0,
    eps=1e-10
)

# === Adadelta ===
# 改进的 Adagrad，不需要设置学习率
optimizer = optim.Adadelta(
    model.parameters(),
    lr=1.0,
    rho=0.9
)

# === LBFGS ===
# 二阶优化，需要特殊使用方式
optimizer = optim.LBFGS(
    model.parameters(),
    lr=1,
    max_iter=20
)

# LBFGS 的使用方式不同
def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    return loss

optimizer.step(closure)
```

### 4. 参数分组

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.head = nn.Linear(64, 10)

model = Model()

# === 不同参数使用不同学习率 ===
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},  # 主干较小学习率
    {'params': model.head.parameters(), 'lr': 1e-3}       # 分类头较大学习率
])

# === 某些参数不使用权重衰减 ===
# 通常 bias 和 LayerNorm 参数不加权重衰减
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if 'bias' in name or 'norm' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=1e-3)

# === 冻结部分参数 ===
# 方法1：不传入优化器
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params)

# 方法2：设置 requires_grad=False
for param in model.backbone.parameters():
    param.requires_grad = False
optimizer = optim.Adam(model.parameters())  # 只会更新 head
```

### 5. 优化器对比实验

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 创建简单的优化问题
def rosenbrock(x, y):
    """Rosenbrock 函数：常用的优化测试函数"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def optimize_rosenbrock(optimizer_class, **kwargs):
    """测试优化器在 Rosenbrock 函数上的表现"""
    x = torch.tensor([-2.0], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)

    optimizer = optimizer_class([x, y], **kwargs)
    history = {'x': [], 'y': [], 'loss': []}

    for _ in range(1000):
        optimizer.zero_grad()
        loss = rosenbrock(x, y)
        loss.backward()
        optimizer.step()

        history['x'].append(x.item())
        history['y'].append(y.item())
        history['loss'].append(loss.item())

    return history

# 对比不同优化器
optimizers = {
    'SGD': (torch.optim.SGD, {'lr': 0.001}),
    'SGD+Momentum': (torch.optim.SGD, {'lr': 0.001, 'momentum': 0.9}),
    'Adam': (torch.optim.Adam, {'lr': 0.01}),
    'RMSprop': (torch.optim.RMSprop, {'lr': 0.01}),
}

results = {}
for name, (opt_class, kwargs) in optimizers.items():
    results[name] = optimize_rosenbrock(opt_class, **kwargs)

# 绘制收敛曲线
plt.figure(figsize=(12, 4))
for name, history in results.items():
    plt.semilogy(history['loss'], label=name)
plt.xlabel('Iteration')
plt.ylabel('Loss (log scale)')
plt.legend()
plt.title('Optimizer Comparison on Rosenbrock Function')
plt.grid(True)
# plt.show()
```

## 深入理解 Deep Dive

### 优化器选择指南

```
任务类型              推荐优化器           备注
─────────────────────────────────────────────────
计算机视觉            SGD+Momentum        配合学习率调度
NLP/Transformer       AdamW               配合 warmup
RNN                   RMSprop/Adam        自适应学习率
GAN                   Adam (β1=0.5)       降低动量
强化学习              Adam                稳定性好
微调预训练模型        AdamW (低学习率)    避免灾难性遗忘
```

### Adam vs SGD

```python
# Adam 的优点：
# 1. 自适应学习率，对不同参数使用不同步长
# 2. 对超参数不敏感，默认值通常就很好
# 3. 收敛速度快

# SGD 的优点：
# 1. 通常能找到更好的全局最优
# 2. 泛化性能更好
# 3. 内存占用更少

# 推荐策略：
# - 快速实验用 Adam
# - 追求最佳性能用 SGD+调度
```

## 常见问题 FAQ

### Q1: 学习率设置

```python
# 经验法则：
# SGD: 0.01 - 0.1
# Adam: 0.001 - 0.0001
# AdamW: 0.001 - 0.00001

# 学习率查找器
def find_lr(model, train_loader, optimizer, criterion, init_lr=1e-7, final_lr=10):
    """实现学习率查找器"""
    lrs = []
    losses = []
    lr = init_lr

    for batch_idx, (data, target) in enumerate(train_loader):
        # 设置学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())

        lr *= (final_lr / init_lr) ** (1 / len(train_loader))

        if loss.item() > 4 * min(losses):
            break

    return lrs, losses
```

## 深入理解：Adam vs AdamW 差异实验

```python
import torch
import torch.nn as nn

# 理解 Adam 与 AdamW 的本质区别
# Adam:  θ = θ - lr * m̂ / (√v̂ + ε) - lr * λ * θ / (√v̂ + ε)  ← L2 正则化（自适应）
# AdamW: θ = θ - lr * m̂ / (√v̂ + ε) - lr * λ * θ             ← 权重衰减（固定比例）

def compare_adam_adamw(steps=500):
    """对比 Adam 和 AdamW 的权重衰减行为"""
    # 创建两个相同初始权重的模型
    torch.manual_seed(42)
    model_adam  = nn.Linear(20, 2)
    model_adamw = nn.Linear(20, 2)
    # 让两个模型权重完全相同
    model_adamw.load_state_dict(model_adam.state_dict())

    opt_adam  = torch.optim.Adam( model_adam.parameters(),  lr=1e-3, weight_decay=0.1)
    opt_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=1e-3, weight_decay=0.1)

    X = torch.randn(64, 20)
    y = torch.randint(0, 2, (64,))
    criterion = nn.CrossEntropyLoss()

    for _ in range(steps):
        for model, opt in [(model_adam, opt_adam), (model_adamw, opt_adamw)]:
            opt.zero_grad()
            criterion(model(X), y).backward()
            opt.step()

    # 衡量权重大小（AdamW 的权重应更小，正则化更强）
    norm_adam  = sum(p.norm().item() for p in model_adam.parameters())
    norm_adamw = sum(p.norm().item() for p in model_adamw.parameters())
    print(f"Adam  权重范数: {norm_adam:.4f}")
    print(f"AdamW 权重范数: {norm_adamw:.4f}  ← AdamW 正则化效果更强")

compare_adam_adamw()

# 结论：在 Transformer 训练中始终优先选择 AdamW
# 因为 Adam 的 weight_decay 被自适应学习率缩放，
# 导致大梯度参数的正则化强度实际上被降低了
```

## 深入理解：梯度裁剪最佳实践

```python
import torch
import torch.nn as nn

def demonstrate_gradient_clipping():
    """演示梯度裁剪对训练稳定性的影响"""
    model = nn.Sequential(
        nn.Linear(10, 64), nn.Tanh(),   # Tanh 容易梯度爆炸
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    X = torch.randn(32, 10)
    y = torch.randn(32, 1)

    for step in range(5):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X), y)
        loss.backward()

        # 计算裁剪前的梯度范数
        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        # 裁剪（max_norm=1.0 是大多数场景的安全值）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        after_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        optimizer.step()
        print(f"Step {step}: loss={loss.item():.4f}  "
              f"grad_norm {total_norm:.2f} → {after_norm:.2f}")

demonstrate_gradient_clipping()

# 经验法则：
# - Transformer / LSTM: max_norm=1.0
# - 一般 CNN/MLP: max_norm=5.0
# - 如果经常裁剪，说明学习率可能偏大
```

## 小结 Summary

| 优化器 | 适用场景 | 典型学习率 |
|--------|----------|-----------|
| SGD+Momentum | CV, 追求最佳性能 | 0.01-0.1 |
| Adam | 快速实验, NLP | 0.001 |
| AdamW | Transformer | 1e-4 - 1e-5 |
| RMSprop | RNN | 0.001-0.01 |

## 下一步 Next

[下一章：学习率调度策略 →](./02-lr-schedulers.md)
