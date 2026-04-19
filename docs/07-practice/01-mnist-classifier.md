# 项目1: 手写数字识别 (MNIST Classifier)

## 概述 Overview

通过 MNIST 数据集构建一个完整的图像分类项目，综合运用前面学到的知识。

## 完整代码 Complete Code

### 1. 数据准备

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
])

# 加载数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 数据加载器
BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# 查看数据
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")

# 可视化样本
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png')
plt.close()
```

### 2. 模型定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    """CNN 分类器"""

    def __init__(self, num_classes=10):
        super().__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # 全连接层
        # 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 第一层: 28x28 -> 14x14
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # 第二层: 14x14 -> 7x7
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # 第三层: 7x7 -> 3x3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# 简单版本：MLP 分类器
class MLPClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# 创建模型
model = CNNClassifier().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 3. 训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })

    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total

# 训练配置
EPOCHS = 10
LEARNING_RATE = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# 训练历史
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

best_acc = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # 训练
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )

    # 验证
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)

    # 学习率调度
    scheduler.step(val_loss)

    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Model saved! Best acc: {best_acc*100:.2f}%")
```

### 4. 结果可视化

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 绘制训练曲线
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()

    # Accuracy
    ax2.plot([a*100 for a in history['train_acc']], label='Train')
    ax2.plot([a*100 for a in history['val_acc']], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curve')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

plot_history(history)

# 混淆矩阵
def plot_confusion_matrix(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return cm

cm = plot_confusion_matrix(model, test_loader, device)
```

### 5. 模型导出与推理

```python
import torch

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 导出 TorchScript
example_input = torch.randn(1, 1, 28, 28).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('mnist_model.pt')

# 单张图片推理
def predict_single(model, image, device):
    """预测单张图片"""
    model.eval()

    if image.dim() == 2:  # (H, W)
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif image.dim() == 3:  # (1, H, W)
        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=-1)
        pred = probs.argmax().item()
        confidence = probs.max().item()

    return pred, confidence

# 测试
test_image, test_label = test_dataset[0]
pred, conf = predict_single(model, test_image, device)
print(f"True: {test_label}, Predicted: {pred}, Confidence: {conf*100:.2f}%")

# 批量推理
def predict_batch(model, images, device):
    """批量预测"""
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=-1)
        preds = probs.argmax(dim=-1)
        confs = probs.max(dim=-1).values

    return preds.cpu().numpy(), confs.cpu().numpy()

# 可视化预测结果
def visualize_predictions(model, dataset, num_samples=10, device='cpu'):
    model.eval()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]
        pred, conf = predict_single(model, image, device)

        ax.imshow(image.squeeze(), cmap='gray')
        color = 'green' if pred == label else 'red'
        ax.set_title(f"True: {label}, Pred: {pred}\nConf: {conf*100:.1f}%", color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

visualize_predictions(model, test_dataset, device=device)
```

## 项目结构

```
mnist-classifier/
├── data/                   # 数据目录
├── train.py               # 训练脚本
├── model.py               # 模型定义
├── utils.py               # 工具函数
├── inference.py           # 推理脚本
├── best_model.pth         # 最佳模型权重
├── mnist_model.pt         # TorchScript 模型
└── requirements.txt       # 依赖
```

## 预期结果

| 指标 | MLP | CNN |
|------|-----|-----|
| 测试准确率 | ~97% | ~99% |
| 参数量 | ~400K | ~200K |
| 训练时间 | 快 | 中等 |

## 下一步 Next

[下一章：文本分类 →](./02-text-classification.md)
