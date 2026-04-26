"""
示例 05: 完整训练循环（含早停 & 模型保存）
对应文档: docs/02-core/03-training-loop.md
运行方式: python examples/05_training_loop.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"使用设备: {device}\n")


# ── 数据 ──────────────────────────────────────────────────────
class ToyDataset(Dataset):
    def __init__(self, n=800):
        X = torch.randn(n, 10)
        # 二分类：特征和 > 0 为正类
        y = (X.sum(dim=1) > 0).long()
        self.X, self.y = X, y

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


dataset = ToyDataset()
train_set, val_set = random_split(dataset, [640, 160], generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=0)


# ── 模型 ──────────────────────────────────────────────────────
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, x): return self.net(x)


model     = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ── 早停辅助类 ────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience   # True 表示应该停止

    def restore_best(self, model: nn.Module):
        model.load_state_dict(self.best_state)


# ── 训练 / 验证函数 ───────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ── 主训练循环 ────────────────────────────────────────────────
print("=== 开始训练 ===")
early_stop = EarlyStopping(patience=5)

for epoch in range(1, 51):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss,   val_acc   = eval_epoch(model, val_loader, criterion)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | "
              f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}")

    if early_stop.step(val_loss, model):
        print(f"\n早停触发 (patience={early_stop.patience})，在 epoch {epoch} 停止")
        break

early_stop.restore_best(model)
_, final_acc = eval_epoch(model, val_loader, criterion)
print(f"\n最终验证准确率: {final_acc:.3f}")
print("\n示例 05 运行完成")
