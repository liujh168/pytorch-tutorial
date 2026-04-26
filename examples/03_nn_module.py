"""
示例 03: nn.Module 自定义模型
对应文档: docs/02-core/01-nn-module.md
运行方式: python examples/03_nn_module.py
"""

import torch
import torch.nn as nn

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"使用设备: {device}\n")


# ── 1. 最简单的自定义层 ───────────────────────────────────────
print("=== 1. 自定义线性层 ===")

class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # nn.Parameter 会被自动注册为可训练参数
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias


layer = MyLinear(4, 2).to(device)
x = torch.randn(3, 4).to(device)
out = layer(x)
print(f"输入 shape: {x.shape}  输出 shape: {out.shape}")
print(f"参数数量: {sum(p.numel() for p in layer.parameters())}")


# ── 2. 多层感知机 (MLP) ───────────────────────────────────────
print("\n=== 2. 多层感知机 (MLP) ===")

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


mlp = MLP(784, 256, 10).to(device)
x = torch.randn(32, 784).to(device)
out = mlp(x)
print(f"输入: (32, 784)  输出: {out.shape}")

# 统计参数
total = sum(p.numel() for p in mlp.parameters())
trainable = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
print(f"总参数: {total:,}  可训练参数: {trainable:,}")


# ── 3. 模型结构打印 ───────────────────────────────────────────
print("\n=== 3. 模型结构 ===")
print(mlp)


# ── 4. 参数访问与冻结 ─────────────────────────────────────────
print("\n=== 4. 参数管理 ===")
for name, param in mlp.named_parameters():
    print(f"  {name:30s}  shape={str(param.shape):20s}  requires_grad={param.requires_grad}")

# 冻结第一层权重（迁移学习时常用）
print("\n冻结 net.0 的参数:")
for param in mlp.net[0].parameters():
    param.requires_grad = False

frozen = sum(p.numel() for p in mlp.parameters() if not p.requires_grad)
print(f"  冻结参数数: {frozen:,}")


# ── 5. 训练/评估模式切换 ─────────────────────────────────────
print("\n=== 5. 训练/评估模式 ===")
mlp.train()   # 启用 Dropout / BatchNorm 的训练行为
print(f"training mode: {mlp.training}")

mlp.eval()    # 关闭 Dropout，使 BatchNorm 使用统计量
print(f"eval mode:     {mlp.training}")

# 推理时配合 no_grad 使用
with torch.no_grad():
    pred = mlp(torch.randn(1, 784).to(device))
    print(f"推理输出 shape: {pred.shape}")


# ── 6. 模型保存与加载 ─────────────────────────────────────────
print("\n=== 6. 模型保存与加载 ===")
import tempfile, os

# 只保存参数（推荐方式）
with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
    path = f.name

torch.save(mlp.state_dict(), path)
print(f"保存到: {path}")

# 加载参数
mlp2 = MLP(784, 256, 10)
mlp2.load_state_dict(torch.load(path, map_location="cpu"))
mlp2.eval()
print(f"加载成功，参数相同: {all(torch.allclose(p1, p2) for p1, p2 in zip(mlp.parameters(), mlp2.parameters()))}")

os.unlink(path)

print("\n示例 03 运行完成")
