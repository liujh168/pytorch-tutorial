"""
示例 09: 端到端 MNIST 手写数字分类
对应文档: docs/07-practice/01-mnist-classifier.md
运行方式: python examples/09_mnist_classifier.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"使用设备: {device}\n")


# ── 1. 数据准备 ───────────────────────────────────────────────
print("=== 1. 加载 MNIST 数据集 ===")

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,)),   # MNIST 的均值和标准差
])

train_set = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_set  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=0)

print(f"训练集: {len(train_set):,} 张  测试集: {len(test_set):,} 张")
x_sample, _ = next(iter(train_loader))
print(f"图像 shape: {x_sample.shape}  (batch, channels, H, W)")


# ── 2. CNN 模型定义 ───────────────────────────────────────────
print("\n=== 2. 模型定义 ===")

class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 卷积块
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28×28 → 28×28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → 14×14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14×14 → 14×14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → 7×7
        )
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(x))


model = CNNClassifier().to(device)
params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {params:,}")

# 验证前向传播维度
with torch.no_grad():
    dummy = torch.randn(2, 1, 28, 28).to(device)
    print(f"输出 shape: {model(dummy).shape}  (应为 [2, 10])")


# ── 3. 训练配置 ───────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2,
    steps_per_epoch=len(train_loader), epochs=5
)


# ── 4. 训练与评估 ─────────────────────────────────────────────
def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def eval_epoch(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)


print("\n=== 3. 开始训练 (5 epochs) ===")
for epoch in range(1, 6):
    train_loss, train_acc = train_epoch(model, train_loader)
    test_acc = eval_epoch(model, test_loader)
    print(f"Epoch {epoch}/5 | loss={train_loss:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")


# ── 5. 每类准确率 ─────────────────────────────────────────────
print("\n=== 4. 各类别准确率 ===")
class_correct = [0] * 10
class_total   = [0] * 10

model.eval()
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        for label, pred in zip(labels, preds):
            class_total[label]   += 1
            class_correct[label] += (pred == label).item()

for i in range(10):
    acc = class_correct[i] / class_total[i]
    print(f"  数字 {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")


# ── 6. 保存模型 ───────────────────────────────────────────────
torch.save(model.state_dict(), "examples/mnist_cnn.pth")
print("\n模型已保存到 examples/mnist_cnn.pth")
print("\n示例 09 运行完成")
