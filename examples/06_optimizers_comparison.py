"""
示例 06: 优化器对比实验 (SGD / Adam / AdamW)
对应文档: docs/03-training-advanced/01-optimizers.md
运行方式: python examples/06_optimizers_comparison.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")   # 无显示器时使用非交互后端
import matplotlib.pyplot as plt

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"使用设备: {device}\n")


# ── 数据 ──────────────────────────────────────────────────────
torch.manual_seed(42)
X = torch.randn(500, 20)
y = (X[:, :5].sum(dim=1) > 0).long()
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True, num_workers=0)


# ── 模型工厂 ──────────────────────────────────────────────────
def make_model():
    return nn.Sequential(
        nn.Linear(20, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 2),
    ).to(device)


# ── 训练函数 ──────────────────────────────────────────────────
def train(optimizer_name: str, epochs: int = 30):
    model     = make_model()
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif optimizer_name == "AdamW":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    else:
        raise ValueError(f"未知优化器: {optimizer_name}")

    history = []
    for _ in range(epochs):
        epoch_loss = 0.0
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        history.append(epoch_loss / len(loader))
    return history


# ── 运行对比 ──────────────────────────────────────────────────
print("=== 优化器对比实验 ===")
results = {}
for name in ["SGD", "Adam", "AdamW"]:
    print(f"  训练 {name}...")
    results[name] = train(name, epochs=30)

# 打印最终 loss
for name, hist in results.items():
    print(f"  {name:6s} 最终 loss: {hist[-1]:.4f}")


# ── 梯度裁剪示例 ──────────────────────────────────────────────
print("\n=== 梯度裁剪 (Gradient Clipping) ===")
model     = make_model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward()

    # 裁剪前
    total_norm_before = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 裁剪后
    total_norm_after = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    optimizer.step()
    print(f"  梯度范数: {total_norm_before:.4f} → 裁剪后 {total_norm_after:.4f}")
    break


# ── 绘制 loss 曲线 ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for name, hist in results.items():
    ax.plot(hist, label=name)
ax.set_xlabel("Epoch")
ax.set_ylabel("Train Loss")
ax.set_title("优化器收敛速度对比")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("examples/optimizers_comparison.png", dpi=120)
print("\n  loss 曲线已保存到 examples/optimizers_comparison.png")

print("\n示例 06 运行完成")
