# 损失函数详解 (Loss Functions)

## 概述 Overview

损失函数（Loss Function）是深度学习的核心组件，它量化了模型预测与真实值之间的差距。选择合适的损失函数对模型性能至关重要。

完成本章后，你将：

- 理解各种损失函数的数学原理
- 知道如何为不同任务选择损失函数
- 学会自定义损失函数
- 掌握处理类别不平衡的技巧

**难度级别**：🟡 进阶级

## 前置知识 Prerequisites

- [03-training-loop](./03-training-loop.md) - 训练循环最佳实践
- 基本的概率论知识

## 核心概念 Core Concepts

### 损失函数的作用

```
模型预测 ŷ  ──┐
              ├──→ Loss Function ──→ loss 值 ──→ backward() ──→ 更新参数
真实标签 y  ──┘
```

### 损失函数分类

| 任务类型 | 常用损失函数 |
|----------|-------------|
| 回归 | MSELoss, L1Loss, SmoothL1Loss |
| 二分类 | BCELoss, BCEWithLogitsLoss |
| 多分类 | CrossEntropyLoss, NLLLoss |
| 多标签 | BCEWithLogitsLoss |
| 序列 | CTCLoss |

## 代码实现 Implementation

### 1. 回归损失函数

```python
import torch
import torch.nn as nn

# 示例数据
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])
y_true = torch.tensor([3.0, -0.5, 2.0, 7.5])

# === MSE Loss (Mean Squared Error) ===
# L = (1/n) * Σ(ŷᵢ - yᵢ)²
mse_loss = nn.MSELoss()
loss = mse_loss(y_pred, y_true)
print(f"MSE Loss: {loss.item():.4f}")

# 手动计算验证
manual_mse = ((y_pred - y_true) ** 2).mean()
print(f"Manual MSE: {manual_mse.item():.4f}")

# reduction 参数
mse_none = nn.MSELoss(reduction='none')  # 返回每个样本的损失
mse_sum = nn.MSELoss(reduction='sum')    # 返回总和
print(f"Per-sample losses: {mse_none(y_pred, y_true)}")

# === L1 Loss (Mean Absolute Error) ===
# L = (1/n) * Σ|ŷᵢ - yᵢ|
l1_loss = nn.L1Loss()
loss = l1_loss(y_pred, y_true)
print(f"L1 Loss: {loss.item():.4f}")

# === Smooth L1 Loss (Huber Loss) ===
# 在 |x| < beta 时是平滑的二次函数，在外部是线性的
# 对异常值更鲁棒
smooth_l1 = nn.SmoothL1Loss(beta=1.0)
loss = smooth_l1(y_pred, y_true)
print(f"Smooth L1 Loss: {loss.item():.4f}")

# === 可视化比较 ===
import matplotlib.pyplot as plt

x = torch.linspace(-3, 3, 100)
y_zero = torch.zeros_like(x)

mse_values = nn.MSELoss(reduction='none')(x, y_zero)
l1_values = nn.L1Loss(reduction='none')(x, y_zero)
smooth_l1_values = nn.SmoothL1Loss(reduction='none')(x, y_zero)

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), mse_values.numpy(), label='MSE')
plt.plot(x.numpy(), l1_values.numpy(), label='L1')
plt.plot(x.numpy(), smooth_l1_values.numpy(), label='Smooth L1')
plt.xlabel('Error')
plt.ylabel('Loss')
plt.legend()
plt.title('Comparison of Regression Loss Functions')
plt.grid(True)
# plt.show()
```

### 2. 分类损失函数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Binary Cross Entropy (BCE) ===
# 用于二分类
# L = -(y * log(ŷ) + (1-y) * log(1-ŷ))

# BCELoss 需要概率输入（经过 sigmoid）
y_pred_prob = torch.tensor([0.9, 0.1, 0.8, 0.3])
y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])

bce_loss = nn.BCELoss()
loss = bce_loss(y_pred_prob, y_true)
print(f"BCE Loss: {loss.item():.4f}")

# BCEWithLogitsLoss 接受 logits（更数值稳定）
y_pred_logits = torch.tensor([2.0, -2.0, 1.5, -1.0])
bce_logits_loss = nn.BCEWithLogitsLoss()
loss = bce_logits_loss(y_pred_logits, y_true)
print(f"BCE with Logits Loss: {loss.item():.4f}")

# 等效于：
# loss = bce_loss(torch.sigmoid(y_pred_logits), y_true)

# === Cross Entropy Loss ===
# 用于多分类
# L = -Σ yᵢ * log(softmax(ŷ)ᵢ)

# 注意：CrossEntropyLoss 期望：
# - input: (N, C) logits
# - target: (N,) 类别索引（整数）

y_pred = torch.tensor([
    [2.0, 1.0, 0.1],  # 样本1：类别0得分最高
    [0.5, 2.5, 0.3],  # 样本2：类别1得分最高
    [0.2, 0.3, 3.0]   # 样本3：类别2得分最高
])
y_true = torch.tensor([0, 1, 2])  # 真实类别

ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(y_pred, y_true)
print(f"Cross Entropy Loss: {loss.item():.4f}")

# 手动计算
log_softmax = F.log_softmax(y_pred, dim=1)
nll_loss = F.nll_loss(log_softmax, y_true)
print(f"Manual CE Loss: {nll_loss.item():.4f}")

# === 带权重的 Cross Entropy（处理类别不平衡）===
# 假设类别0有100样本，类别1有10样本，类别2有5样本
class_weights = torch.tensor([1.0, 10.0, 20.0])  # 反比于样本数
weighted_ce = nn.CrossEntropyLoss(weight=class_weights)
loss = weighted_ce(y_pred, y_true)
print(f"Weighted CE Loss: {loss.item():.4f}")

# === Label Smoothing ===
# 防止过拟合，提高泛化能力
smooth_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
# 相当于：target = 0.9 * one_hot + 0.1 / num_classes
loss = smooth_ce(y_pred, y_true)
print(f"Label Smoothed CE Loss: {loss.item():.4f}")

# === NLL Loss (Negative Log Likelihood) ===
# 需要 log_softmax 输入
log_probs = F.log_softmax(y_pred, dim=1)
nll = nn.NLLLoss()
loss = nll(log_probs, y_true)
print(f"NLL Loss: {loss.item():.4f}")
```

### 3. 特殊损失函数

```python
import torch
import torch.nn as nn

# === Focal Loss（处理极度不平衡）===
class FocalLoss(nn.Module):
    """
    Focal Loss: 减少易分类样本的权重，关注难分类样本
    FL = -α(1-p)^γ * log(p)
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 使用
focal = FocalLoss(gamma=2.0)
y_pred = torch.randn(10, 5)  # 10 samples, 5 classes
y_true = torch.randint(0, 5, (10,))
loss = focal(y_pred, y_true)
print(f"Focal Loss: {loss.item():.4f}")

# === Dice Loss（用于分割任务）===
class DiceLoss(nn.Module):
    """
    Dice Loss: 基于 Dice 系数
    Dice = 2|X ∩ Y| / (|X| + |Y|)
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (N, C, H, W) 预测
        # targets: (N, C, H, W) one-hot 编码的目标
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice

# === Triplet Loss（用于度量学习）===
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
anchor = torch.randn(32, 128)    # 锚点
positive = torch.randn(32, 128)  # 正样本
negative = torch.randn(32, 128)  # 负样本
loss = triplet_loss(anchor, positive, negative)
print(f"Triplet Loss: {loss.item():.4f}")

# === Contrastive Loss ===
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss: 用于孪生网络
    L = (1-y) * D² + y * max(margin - D, 0)²
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        # y=0: 同类，y=1: 不同类
        dist = F.pairwise_distance(x1, x2)
        loss = (1 - y) * dist.pow(2) + y * F.relu(self.margin - dist).pow(2)
        return loss.mean()

# === KL Divergence（用于分布匹配）===
kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=False)
# 注意：输入是 log 概率，目标是概率
log_pred = F.log_softmax(torch.randn(10, 5), dim=1)
target = F.softmax(torch.randn(10, 5), dim=1)
loss = kl_loss(log_pred, target)
print(f"KL Divergence Loss: {loss.item():.4f}")

# === CTC Loss（用于序列转录）===
ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
# T=50 时间步，N=16 batch，C=20 类别
log_probs = F.log_softmax(torch.randn(50, 16, 20), dim=2)
targets = torch.randint(1, 20, (16, 30))  # 目标序列
input_lengths = torch.full((16,), 50, dtype=torch.long)
target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
print(f"CTC Loss: {loss.item():.4f}")
```

### 4. 自定义损失函数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 方式一：函数式 ===
def custom_loss(y_pred, y_true, alpha=0.5):
    """组合 MSE 和 L1 损失"""
    mse = F.mse_loss(y_pred, y_true)
    l1 = F.l1_loss(y_pred, y_true)
    return alpha * mse + (1 - alpha) * l1

# === 方式二：继承 nn.Module ===
class CombinedLoss(nn.Module):
    """组合多个损失函数"""
    def __init__(self, alpha=0.5, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred_class, pred_reg, target_class, target_reg):
        loss_ce = self.ce(pred_class, target_class)
        loss_mse = self.mse(pred_reg, target_reg)
        return self.alpha * loss_ce + self.beta * loss_mse

# === 带可学习参数的损失函数 ===
class LearnableLoss(nn.Module):
    """损失权重可学习"""
    def __init__(self, num_losses=2):
        super().__init__()
        # 学习 log(σ²)，确保正值
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        """
        losses: list of loss tensors
        基于不确定性加权多任务学习
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss

# 使用
learnable_loss = LearnableLoss(num_losses=2)
loss1 = torch.tensor(1.0)
loss2 = torch.tensor(0.5)
total = learnable_loss([loss1, loss2])

# === 带正则化的损失 ===
class RegularizedLoss(nn.Module):
    def __init__(self, base_loss, model, l1_lambda=0.0, l2_lambda=0.0):
        super().__init__()
        self.base_loss = base_loss
        self.model = model
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, pred, target):
        loss = self.base_loss(pred, target)

        # L1 正则化
        if self.l1_lambda > 0:
            l1_reg = sum(p.abs().sum() for p in self.model.parameters())
            loss += self.l1_lambda * l1_reg

        # L2 正则化
        if self.l2_lambda > 0:
            l2_reg = sum(p.pow(2).sum() for p in self.model.parameters())
            loss += self.l2_lambda * l2_reg

        return loss
```

### 5. 处理类别不平衡

```python
import torch
import torch.nn as nn
import numpy as np

# === 方法一：类别权重 ===
# 假设数据分布：类0=1000, 类1=100, 类2=10
class_counts = torch.tensor([1000, 100, 10], dtype=torch.float)

# 权重 = 总样本数 / (类别数 * 该类样本数)
weights = class_counts.sum() / (len(class_counts) * class_counts)
print(f"Class weights: {weights}")

criterion = nn.CrossEntropyLoss(weight=weights)

# === 方法二：Focal Loss ===
# 见上面的实现

# === 方法三：采样策略 ===
from torch.utils.data import WeightedRandomSampler

# 为每个样本计算权重
labels = torch.tensor([0, 0, 0, 1, 1, 2])  # 示例标签
class_sample_counts = torch.bincount(labels)
weights_per_class = 1.0 / class_sample_counts.float()
sample_weights = weights_per_class[labels]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# === 方法四：过采样少数类 ===
def oversample_minority(data, labels, target_ratio=1.0):
    """简单过采样"""
    unique, counts = np.unique(labels.numpy(), return_counts=True)
    max_count = counts.max()

    new_data = [data]
    new_labels = [labels]

    for cls, count in zip(unique, counts):
        if count < max_count * target_ratio:
            # 需要增加的样本数
            n_samples = int(max_count * target_ratio - count)
            # 找到该类的索引
            indices = (labels == cls).nonzero().squeeze()
            # 随机采样（有放回）
            sample_indices = indices[torch.randint(len(indices), (n_samples,))]
            new_data.append(data[sample_indices])
            new_labels.append(labels[sample_indices])

    return torch.cat(new_data), torch.cat(new_labels)

# === 方法五：Class-Balanced Loss ===
class ClassBalancedLoss(nn.Module):
    """
    基于有效样本数的类别平衡损失
    """
    def __init__(self, samples_per_class, num_classes, beta=0.9999):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, pred, target):
        return self.ce(pred, target)
```

## 深入理解 Deep Dive

### 损失函数的数学解释

```python
import torch
import torch.nn.functional as F

# === Cross Entropy 的信息论解释 ===
# Cross Entropy H(p, q) = -Σ p(x) log q(x)
# 当 p 是真实分布，q 是预测分布时
# 最小化交叉熵 = 最小化预测分布与真实分布的差异

# 对于分类任务，p 是 one-hot 向量
# H(p, q) = -log q(true_class)
# 所以 CE loss 实际上是 -log(预测正确类别的概率)

y_pred = torch.tensor([2.0, 1.0, 0.1])  # logits
y_true = 0  # 真实类别是 0

probs = F.softmax(y_pred, dim=0)
ce_manual = -torch.log(probs[y_true])
print(f"CE = -log(p[{y_true}]) = -log({probs[y_true]:.4f}) = {ce_manual:.4f}")

# === MSE 的概率解释 ===
# 假设 y = f(x) + ε, ε ~ N(0, σ²)
# 最大化似然 = 最小化 MSE
# 因为 log p(y|x) ∝ -(y - f(x))² / (2σ²)
```

### 数值稳定性

```python
import torch
import torch.nn.functional as F

# === 为什么用 BCEWithLogitsLoss 而不是 BCE ===
# 直接计算可能数值不稳定
y_pred_prob = torch.tensor([0.0, 1.0])  # 极端概率
y_true = torch.tensor([0.0, 1.0])

# 这会有问题：log(0) = -inf, log(1) = 0
# bce = -(y_true * torch.log(y_pred_prob) + (1-y_true) * torch.log(1-y_pred_prob))

# BCEWithLogitsLoss 使用 log-sum-exp 技巧
# loss = max(x, 0) - x*y + log(1 + exp(-|x|))
# 这保证了数值稳定

# === 为什么 CrossEntropyLoss 比 Softmax + NLLLoss 好 ===
# 直接计算 softmax 可能溢出
logits = torch.tensor([1000.0, 1.0, 0.1])  # 很大的 logit
# exp(1000) 会溢出

# CrossEntropyLoss 使用 log-softmax
# log_softmax(x)_i = x_i - log(Σ exp(x_j))
#                  = x_i - max(x) - log(Σ exp(x_j - max(x)))
# 减去 max(x) 保证数值稳定
```

## 常见问题 FAQ

### Q1: Loss 变成 NaN

```python
# 可能原因：
# 1. 学习率太大
# 2. 输入包含 NaN/Inf
# 3. log(0) 或 除以 0

# 调试方法
def debug_loss(pred, target, loss_fn):
    print(f"Pred min/max: {pred.min():.4f}/{pred.max():.4f}")
    print(f"Pred has NaN: {torch.isnan(pred).any()}")
    print(f"Target min/max: {target.min():.4f}/{target.max():.4f}")

    loss = loss_fn(pred, target)
    print(f"Loss: {loss.item()}")
    return loss

# 添加数值保护
def safe_log(x, eps=1e-7):
    return torch.log(x.clamp(min=eps))
```

### Q2: 如何选择损失函数

```python
# 决策树：
# 1. 回归问题
#    - 异常值少：MSELoss
#    - 异常值多：L1Loss 或 SmoothL1Loss
#
# 2. 二分类
#    - 输出 logits：BCEWithLogitsLoss
#    - 输出概率：BCELoss
#
# 3. 多分类
#    - 标签是类别索引：CrossEntropyLoss
#    - 标签是概率分布：KLDivLoss
#
# 4. 类别不平衡
#    - 使用 weight 参数或 Focal Loss
```

## 小结 Summary

本章要点：

1. **回归损失**
   - `MSELoss`: 标准选择
   - `L1Loss`: 对异常值鲁棒
   - `SmoothL1Loss`: 两者结合

2. **分类损失**
   - `CrossEntropyLoss`: 多分类标准
   - `BCEWithLogitsLoss`: 二分类/多标签

3. **处理不平衡**
   - 类别权重
   - Focal Loss
   - 采样策略

4. **自定义损失**
   ```python
   class MyLoss(nn.Module):
       def forward(self, pred, target):
           return custom_computation
   ```

## 延伸阅读 Further Reading

- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Focal Loss 论文](https://arxiv.org/abs/1708.02002)

## 下一步 Next

恭喜你完成了核心篇！你已经掌握了 PyTorch 的核心组件。接下来，我们将进入训练进阶篇，学习更高级的优化技术。

[下一章：优化器深度对比 →](../03-training-advanced/01-optimizers.md)
