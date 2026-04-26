"""
示例 01: Tensor 基础操作
对应文档: docs/01-basics/02-tensors.md
运行方式: python examples/01_tensor_basics.py
"""

import torch

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"使用设备: {device}\n")


# ── 1. 创建 Tensor ────────────────────────────────────────────
print("=== 1. 创建 Tensor ===")

# 从 Python 列表创建
a = torch.tensor([1.0, 2.0, 3.0])
print(f"从列表创建: {a}")

# 从嵌套列表创建二维张量
b = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(f"二维张量 shape={b.shape}:\n{b}")

# 常用工厂函数
print(f"\ntorch.zeros(2,3):\n{torch.zeros(2, 3)}")
print(f"torch.ones(2,3):\n{torch.ones(2, 3)}")
print(f"torch.rand(2,3):\n{torch.rand(2, 3)}")
print(f"torch.arange(0,10,2): {torch.arange(0, 10, 2)}")
print(f"torch.linspace(0,1,5): {torch.linspace(0, 1, 5)}")


# ── 2. 形状操作 ───────────────────────────────────────────────
print("\n=== 2. 形状操作 ===")
x = torch.arange(12, dtype=torch.float32)
print(f"原始 shape={x.shape}: {x}")

# reshape / view
x2d = x.reshape(3, 4)
print(f"reshape(3,4):\n{x2d}")

# squeeze / unsqueeze
x_unsq = x2d.unsqueeze(0)   # 在第 0 维添加一个维度
print(f"unsqueeze(0) shape: {x_unsq.shape}")

x_sq = x_unsq.squeeze(0)    # 去掉大小为 1 的维度
print(f"squeeze(0) shape: {x_sq.shape}")

# permute — 调换维度顺序
img = torch.rand(3, 64, 64)  # (C, H, W)
img_hwc = img.permute(1, 2, 0)  # (H, W, C)
print(f"permute (C,H,W)->(H,W,C): {img.shape} -> {img_hwc.shape}")


# ── 3. 数学运算 ───────────────────────────────────────────────
print("\n=== 3. 数学运算 ===")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"加法:      {a + b}")
print(f"逐元素乘:  {a * b}")
print(f"点积:      {torch.dot(a, b)}")
print(f"矩阵乘法:  {torch.mm(a.unsqueeze(0), b.unsqueeze(1))}")  # (1,3) x (3,1)

# 广播机制 (Broadcasting)
m = torch.ones(3, 3)
v = torch.tensor([1.0, 2.0, 3.0])  # shape (3,) 会广播为 (3,3)
print(f"\n广播加法 (3x3) + (3,):\n{m + v}")


# ── 4. 索引与切片 ─────────────────────────────────────────────
print("\n=== 4. 索引与切片 ===")
x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
print(f"原矩阵:\n{x}")
print(f"第2行:     {x[1]}")
print(f"第2行第3列: {x[1, 2]}")
print(f"前2行:     \n{x[:2]}")
print(f"布尔索引 >8:\n{x[x > 8]}")


# ── 5. 设备转移 ───────────────────────────────────────────────
print("\n=== 5. 设备管理 ===")
cpu_tensor = torch.randn(3, 3)
gpu_tensor = cpu_tensor.to(device)
print(f"CPU tensor: {cpu_tensor.device}")
print(f"目标设备 tensor: {gpu_tensor.device}")

# 始终确保运算在同一设备上
result = gpu_tensor @ gpu_tensor.T
print(f"矩阵乘法结果 shape: {result.shape}, device: {result.device}")

print("\n示例 01 运行完成")
