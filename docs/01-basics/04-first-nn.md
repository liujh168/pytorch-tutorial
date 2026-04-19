# 构建第一个神经网络 (First Neural Network)

## 概述 Overview

本章将综合前几章所学，构建你的第一个完整的神经网络。我们将从最简单的感知机开始，逐步构建多层神经网络，并完成一个实际的分类任务。

完成本章后，你将：

- 理解神经网络的基本结构
- 学会使用 `torch.nn` 模块
- 完成一个完整的训练流程
- 能够构建和训练简单的神经网络

**难度级别**：🟢 入门级

## 前置知识 Prerequisites

- [02-tensors](./02-tensors.md) - Tensor 基础操作
- [03-autograd](./03-autograd.md) - 自动微分机制

## 核心概念 Core Concepts

### 神经网络的基本组成

```
输入层 → 隐藏层(们) → 输出层

每一层：
  输入 x → 线性变换 (Wx + b) → 激活函数 f → 输出 y
```

### PyTorch 神经网络核心组件

```python
import torch.nn as nn

# 核心模块
nn.Module      # 所有神经网络模块的基类
nn.Linear      # 全连接层（线性变换）
nn.ReLU        # 激活函数
nn.Sigmoid     # 激活函数
nn.Sequential  # 顺序容器

# 损失函数
nn.MSELoss         # 均方误差（回归）
nn.CrossEntropyLoss # 交叉熵（分类）

# 优化器
torch.optim.SGD    # 随机梯度下降
torch.optim.Adam   # Adam 优化器
```

## 代码实现 Implementation

### 1. 从零开始：手动实现神经网络

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# === 生成模拟数据 ===
torch.manual_seed(42)

# 二分类数据：两个簇
n_samples = 200
X1 = torch.randn(n_samples // 2, 2) + torch.tensor([2.0, 2.0])
X2 = torch.randn(n_samples // 2, 2) + torch.tensor([-2.0, -2.0])
X = torch.cat([X1, X2], dim=0)
y = torch.cat([torch.zeros(n_samples // 2),
               torch.ones(n_samples // 2)]).reshape(-1, 1)

print(f"X shape: {X.shape}")  # (200, 2)
print(f"y shape: {y.shape}")  # (200, 1)

# === 手动实现的神经网络 ===
class ManualNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 随机初始化权重
        self.W1 = torch.randn(input_size, hidden_size) * 0.01
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(hidden_size, output_size) * 0.01
        self.b2 = torch.zeros(output_size)

        # 启用梯度追踪
        self.W1.requires_grad_(True)
        self.b1.requires_grad_(True)
        self.W2.requires_grad_(True)
        self.b2.requires_grad_(True)

    def forward(self, x):
        # 第一层：线性 + ReLU
        self.z1 = x @ self.W1 + self.b1
        self.a1 = torch.relu(self.z1)

        # 第二层：线性 + Sigmoid
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = torch.sigmoid(self.z2)
        return self.a2

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

# 创建模型
model = ManualNN(input_size=2, hidden_size=8, output_size=1)

# === 训练循环 ===
learning_rate = 0.1
epochs = 1000
losses = []

for epoch in range(epochs):
    # 前向传播
    y_pred = model.forward(X)

    # 计算损失（二元交叉熵）
    loss = F.binary_cross_entropy(y_pred, y)
    losses.append(loss.item())

    # 反向传播
    loss.backward()

    # 更新参数
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            param.grad.zero_()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# === 评估 ===
with torch.no_grad():
    y_pred = model.forward(X)
    predictions = (y_pred > 0.5).float()
    accuracy = (predictions == y).float().mean()
    print(f"Accuracy: {accuracy.item():.2%}")
```

### 2. 使用 nn.Module 构建网络

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    """使用 nn.Module 构建的神经网络"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # 必须调用父类构造函数

        # 定义层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 创建模型
model = SimpleNN(input_size=2, hidden_size=8, output_size=1)
print(model)

# 查看模型参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# === 更简洁的方式：nn.Sequential ===
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
print(model)
```

### 3. 完整的训练流程

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# === 准备数据 ===
torch.manual_seed(42)

# 生成数据
n_samples = 1000
X = torch.randn(n_samples, 10)
y = (X[:, 0] + X[:, 1] > 0).float().reshape(-1, 1)

# 划分训练集和测试集
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === 定义模型 ===
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = NeuralNetwork()

# === 定义损失函数和优化器 ===
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 训练函数 ===
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()  # 设置为训练模式
    total_loss = 0

    for batch_X, batch_y in dataloader:
        # 1. 清零梯度
        optimizer.zero_grad()

        # 2. 前向传播
        outputs = model(batch_X)

        # 3. 计算损失
        loss = criterion(outputs, batch_y)

        # 4. 反向传播
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# === 评估函数 ===
def evaluate(model, dataloader, criterion):
    model.eval()  # 设置为评估模式
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

# === 训练循环 ===
epochs = 50
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")

print(f"\nFinal Test Accuracy: {test_accuracies[-1]:.2%}")
```

### 4. 多分类问题

```python
import torch
import torch.nn as nn
import torch.optim as optim

# === 生成多分类数据 ===
torch.manual_seed(42)

n_samples = 1000
n_classes = 5
X = torch.randn(n_samples, 20)
y = torch.randint(0, n_classes, (n_samples,))  # 整数标签

# 划分数据
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# === 多分类模型 ===
class MultiClassNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
            # 注意：不需要 Softmax，CrossEntropyLoss 包含了
        )

    def forward(self, x):
        return self.network(x)

model = MultiClassNN(input_size=20, hidden_size=64, num_classes=5)

# === 损失函数：CrossEntropyLoss ===
# 它组合了 LogSoftmax 和 NLLLoss
# 输入：(N, C) 的 logits，(N,) 的类别索引
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 训练 ===
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)  # (N, C)
    loss = criterion(outputs, y_train)  # y_train 是整数

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        # 评估
        model.eval()
        with torch.no_grad():
            train_pred = outputs.argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean()

            test_outputs = model(X_test)
            test_pred = test_outputs.argmax(dim=1)
            test_acc = (test_pred == y_test).float().mean()

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Train Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}")

# === 预测新数据 ===
model.eval()
with torch.no_grad():
    new_data = torch.randn(5, 20)
    logits = model(new_data)
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)

    print("\nPredictions for new data:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"  Sample {i}: Class {pred.item()}, Confidence: {prob[pred].item():.2%}")
```

### 5. 使用 GPU 训练

```python
import torch
import torch.nn as nn
import torch.optim as optim

# === 设备选择 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === 模型定义 ===
class GPUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# 将模型移到 GPU
model = GPUModel().to(device)
print(f"Model on device: {next(model.parameters()).device}")

# === 数据也要移到同一设备 ===
X = torch.randn(1000, 100).to(device)
y = torch.randint(0, 10, (1000,)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# === 训练 ===
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# === 更好的写法：使用 DataLoader ===
from torch.utils.data import DataLoader, TensorDataset

# 数据保持在 CPU，训练时移到 GPU
X_cpu = torch.randn(1000, 100)
y_cpu = torch.randint(0, 10, (1000,))

dataset = TensorDataset(X_cpu, y_cpu)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for batch_X, batch_y in loader:
        # 每个 batch 移到 GPU
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

### 6. 模型保存与加载

```python
import torch
import torch.nn as nn

# === 定义模型 ===
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MyModel(10, 64, 5)

# 假设已训练...

# === 保存方式 1：只保存参数（推荐）===
torch.save(model.state_dict(), 'model_weights.pth')

# 加载
new_model = MyModel(10, 64, 5)  # 先创建相同结构的模型
new_model.load_state_dict(torch.load('model_weights.pth'))
new_model.eval()  # 设为评估模式

# === 保存方式 2：保存整个模型 ===
torch.save(model, 'full_model.pth')

# 加载
loaded_model = torch.load('full_model.pth')
loaded_model.eval()

# === 保存 checkpoint（训练中断恢复）===
checkpoint = {
    'epoch': 100,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.5,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载 checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
loss = checkpoint['loss']

# === 跨设备加载 ===
# GPU 训练的模型在 CPU 上加载
model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))

# 指定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.to(device)
```

## 深入理解 Deep Dive

### 权重初始化

```python
import torch
import torch.nn as nn

class InitializedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

        # 手动初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier 初始化（适合 tanh/sigmoid）
                nn.init.xavier_uniform_(m.weight)
                # Kaiming 初始化（适合 ReLU）
                # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 查看初始化后的权重
model = InitializedModel()
print(f"fc1 weight mean: {model.fc1.weight.mean().item():.4f}")
print(f"fc1 weight std: {model.fc1.weight.std().item():.4f}")
```

### 训练 vs 评估模式

```python
import torch.nn as nn

class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout
        self.bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)      # BatchNorm 在训练和评估时行为不同
        x = self.dropout(x) # Dropout 只在训练时激活
        return x

model = ModelWithDropout()

# 训练时
model.train()
print(f"Training mode: {model.training}")  # True

# 评估时
model.eval()
print(f"Eval mode: {model.training}")  # False

# 重要：推理时要用 eval 模式和 no_grad
model.eval()
with torch.no_grad():
    output = model(torch.randn(1, 10))
```

## 常见问题 FAQ

### Q1: 损失不下降

```python
# 可能的原因和解决方案：

# 1. 学习率太大或太小
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 尝试不同值

# 2. 梯度消失/爆炸
# - 使用合适的初始化
# - 使用 BatchNorm
# - 使用 ReLU 而非 Sigmoid

# 3. 数据问题
# - 检查数据是否正确加载
# - 检查标签是否正确

# 4. 模型太简单/复杂
# - 增加/减少层数或神经元数量
```

### Q2: 过拟合

```python
# 训练损失低，测试损失高

# 1. 添加正则化
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 2. 添加 Dropout
self.dropout = nn.Dropout(0.5)

# 3. 数据增强（图像任务）

# 4. 早停（Early Stopping）
best_loss = float('inf')
patience = 10
counter = 0
for epoch in range(epochs):
    val_loss = evaluate(...)
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break
```

### Q3: GPU 内存不足

```python
# 1. 减小 batch size
train_loader = DataLoader(dataset, batch_size=16)  # 减小

# 2. 使用梯度累积
accumulation_steps = 4
for i, (x, y) in enumerate(loader):
    loss = model(x).sum() / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 清理缓存
torch.cuda.empty_cache()

# 4. 使用混合精度训练（后续章节详解）
```

## 小结 Summary

本章要点：

1. **模型定义**
   ```python
   class Model(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc = nn.Linear(10, 5)

       def forward(self, x):
           return self.fc(x)
   ```

2. **训练循环**
   ```python
   for epoch in range(epochs):
       optimizer.zero_grad()
       output = model(x)
       loss = criterion(output, y)
       loss.backward()
       optimizer.step()
   ```

3. **评估**
   ```python
   model.eval()
   with torch.no_grad():
       predictions = model(test_data)
   ```

4. **保存/加载**
   ```python
   torch.save(model.state_dict(), 'model.pth')
   model.load_state_dict(torch.load('model.pth'))
   ```

## 延伸阅读 Further Reading

- [PyTorch 神经网络教程](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- [nn.Module 文档](https://pytorch.org/docs/stable/nn.html)

## 下一步 Next

恭喜你完成了基础篇！你已经掌握了 PyTorch 的核心概念。接下来，我们将深入学习 `nn.Module` 的高级用法。

[下一章：nn.Module 深入理解 →](../02-core/01-nn-module.md)
