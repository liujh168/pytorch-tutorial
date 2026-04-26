"""
示例 04: Dataset & DataLoader
对应文档: docs/02-core/02-data-loading.md
运行方式: python examples/04_data_loading.py
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"使用设备: {device}\n")


# ── 1. 自定义 Dataset ─────────────────────────────────────────
print("=== 1. 自定义 Dataset ===")

class RegressionDataset(Dataset):
    """合成回归数据集: y = 2x₁ + 3x₂ + noise"""

    def __init__(self, n_samples: int = 1000, noise: float = 0.1):
        super().__init__()
        X = torch.randn(n_samples, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + noise * torch.randn(n_samples)
        self.X = X
        self.y = y.unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


dataset = RegressionDataset(n_samples=1000)
print(f"数据集大小: {len(dataset)}")
x_sample, y_sample = dataset[0]
print(f"单样本 X shape: {x_sample.shape}, y shape: {y_sample.shape}")


# ── 2. 训练/验证集划分 ────────────────────────────────────────
print("\n=== 2. 数据集划分 ===")
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42))
print(f"训练集: {len(train_set)}  验证集: {len(val_set)}")


# ── 3. DataLoader ─────────────────────────────────────────────
print("\n=== 3. DataLoader ===")
train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    num_workers=0,        # Windows 上设为 0 避免 multiprocessing 问题
    pin_memory=(device.type == "cuda"),
)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

print(f"训练 batches: {len(train_loader)}")
print(f"验证 batches: {len(val_loader)}")

# 查看一个 batch
X_batch, y_batch = next(iter(train_loader))
print(f"Batch X shape: {X_batch.shape}, y shape: {y_batch.shape}")


# ── 4. 自定义 collate_fn ─────────────────────────────────────
print("\n=== 4. 自定义 collate_fn (变长序列示例) ===")

class VariableLengthDataset(Dataset):
    """每条样本长度不同，需要 padding"""
    def __init__(self, n=20):
        self.data = [torch.randn(torch.randint(3, 10, ()).item()) for _ in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_collate(batch):
    """将变长序列 padding 到同一长度"""
    lengths = torch.tensor([len(x) for x in batch])
    max_len = lengths.max().item()
    padded = torch.zeros(len(batch), max_len)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded, lengths


vl_dataset = VariableLengthDataset(n=20)
vl_loader  = DataLoader(vl_dataset, batch_size=4, collate_fn=pad_collate)
padded, lengths = next(iter(vl_loader))
print(f"Padded batch shape: {padded.shape}")
print(f"序列长度: {lengths.tolist()}")


# ── 5. 数据集统计（均值/标准差）───────────────────────────────
print("\n=== 5. 计算数据集统计量 ===")
# 用于归一化的均值和标准差
all_X = torch.stack([dataset[i][0] for i in range(len(dataset))])
mean = all_X.mean(dim=0)
std  = all_X.std(dim=0)
print(f"特征均值: {mean}")
print(f"特征标准差: {std}")

print("\n示例 04 运行完成")
