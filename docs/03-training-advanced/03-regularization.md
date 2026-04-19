# 正则化技术 (Regularization Techniques)

## 概述 Overview

正则化是防止过拟合、提高模型泛化能力的关键技术。本章介绍深度学习中常用的正则化方法。

**难度级别**：⭐ 重点章节

## 代码实现 Implementation

### 1. Dropout

```python
import torch
import torch.nn as nn

# === 基本 Dropout ===
class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(p=0.5)  # 50% 丢弃率
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # 训练时随机丢弃
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

model = DropoutModel()

# 训练模式：Dropout 激活
model.train()
output = model(torch.randn(32, 784))

# 评估模式：Dropout 关闭
model.eval()
output = model(torch.randn(32, 784))

# === Dropout 变体 ===
# Dropout2d: 用于 CNN，丢弃整个通道
dropout2d = nn.Dropout2d(p=0.5)

# AlphaDropout: 用于 SELU 激活
alpha_dropout = nn.AlphaDropout(p=0.5)
```

### 2. Batch Normalization

```python
import torch
import torch.nn as nn

# === 全连接层的 BatchNorm ===
class BNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)  # 特征维度
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # 批归一化
        x = torch.relu(x)
        return self.fc2(x)

# === CNN 的 BatchNorm ===
class CNNWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # 通道数
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return x

# BatchNorm 的参数
bn = nn.BatchNorm1d(
    num_features=64,
    eps=1e-5,         # 数值稳定
    momentum=0.1,     # running mean/var 更新系数
    affine=True,      # 是否有可学习的 γ 和 β
    track_running_stats=True  # 是否追踪 running statistics
)
```

### 3. Layer Normalization

```python
import torch
import torch.nn as nn

# Layer Norm: Transformer 的标配
class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, 8)
        self.norm1 = nn.LayerNorm(d_model)  # 在特征维度归一化
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm 架构
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

# LayerNorm vs BatchNorm
# - BatchNorm: 在 batch 维度归一化，依赖 batch size
# - LayerNorm: 在特征维度归一化，不依赖 batch size
# - LayerNorm 更适合 NLP 和小 batch 训练
```

### 4. 权重正则化

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 5)

# === L2 正则化（通过 weight_decay）===
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# === L1 正则化（需要手动添加）===
def l1_regularization(model, lambda_l1=1e-5):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.abs(param).sum()
    return lambda_l1 * l1_loss

# 在训练循环中
loss = criterion(output, target)
loss += l1_regularization(model)
loss.backward()

# === Elastic Net（L1 + L2）===
def elastic_net(model, lambda_l1=1e-5, lambda_l2=1e-4):
    l1_loss = sum(torch.abs(p).sum() for p in model.parameters())
    l2_loss = sum(torch.pow(p, 2).sum() for p in model.parameters())
    return lambda_l1 * l1_loss + lambda_l2 * l2_loss
```

### 5. 数据增强

```python
from torchvision import transforms

# === 图像增强 ===
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5),  # Cutout
])

# === MixUp ===
def mixup_data(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# === CutMix ===
def cutmix_data(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size)

    # 计算裁剪框
    cut_ratio = torch.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy = torch.randint(W, (1,)).item(), torch.randint(H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))

    return x, y, y[index], lam
```

### 6. 早停 (Early Stopping)

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

# 使用
early_stopping = EarlyStopping(patience=10)
for epoch in range(100):
    val_loss = validate(...)
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

## 小结 Summary

| 技术 | 作用 | 适用场景 |
|------|------|----------|
| Dropout | 随机丢弃神经元 | 全连接层 |
| BatchNorm | 批量归一化 | CNN |
| LayerNorm | 层归一化 | Transformer/RNN |
| L2 正则化 | 权重衰减 | 所有模型 |
| 数据增强 | 增加数据多样性 | 图像任务 |
| 早停 | 防止过拟合 | 所有任务 |

## 下一步 Next

[下一章：梯度技巧 →](./04-gradient-techniques.md)
