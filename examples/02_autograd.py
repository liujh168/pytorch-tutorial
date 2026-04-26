"""
示例 02: 自动微分 (Autograd)
对应文档: docs/01-basics/03-autograd.md
运行方式: python examples/02_autograd.py
"""

import torch

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"使用设备: {device}\n")


# ── 1. 基础梯度计算 ───────────────────────────────────────────
print("=== 1. 基础梯度计算 ===")

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1   # y = x^2 + 2x + 1

y.backward()              # 反向传播计算梯度
# dy/dx = 2x + 2，在 x=3 时 = 8
print(f"x = {x.item()}")
print(f"y = x^2+2x+1 = {y.item()}")
print(f"dy/dx = {x.grad.item()}  (理论值: 2*3+2=8)")


# ── 2. 多变量梯度 ─────────────────────────────────────────────
print("\n=== 2. 多变量梯度 ===")
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# z = a^3 + b^2
z = a ** 3 + b ** 2
z.backward()

print(f"a={a.item()}, b={b.item()}")
print(f"z = a^3+b^2 = {z.item()}")
print(f"dz/da = 3a^2 = {a.grad.item()}  (理论值: {3*4})")
print(f"dz/db = 2b  = {b.grad.item()}  (理论值: {2*3})")


# ── 3. 计算图与 grad_fn ───────────────────────────────────────
print("\n=== 3. 计算图 ===")
x = torch.randn(3, requires_grad=True)
y = x * 2          # MulBackward
z = y.mean()       # MeanBackward

print(f"x.grad_fn: {x.grad_fn}")    # None，叶子节点
print(f"y.grad_fn: {y.grad_fn}")    # MulBackward0
print(f"z.grad_fn: {z.grad_fn}")    # MeanBackward0

z.backward()
print(f"x.grad: {x.grad}")          # 均值对 x 的梯度 = 2/3


# ── 4. 梯度累积 ───────────────────────────────────────────────
print("\n=== 4. 梯度累积 (常用于大 batch 拆分) ===")
w = torch.tensor(1.0, requires_grad=True)

# 模拟 3 个 mini-batch 的累积
accumulation_steps = 3
for i in range(accumulation_steps):
    loss = (w * (i + 1)) ** 2
    loss.backward()   # 梯度会累加，不清零
    print(f"  step {i+1}: loss={loss.item():.2f}, w.grad={w.grad.item():.2f}")

print(f"累积后 w.grad = {w.grad.item():.2f}")


# ── 5. 关闭梯度追踪 ───────────────────────────────────────────
print("\n=== 5. 关闭梯度追踪 (推理时使用) ===")
x = torch.randn(3, requires_grad=True)

# 方式一: torch.no_grad() 上下文管理器（推理标准写法）
with torch.no_grad():
    y = x * 2
    print(f"no_grad 内部 y.requires_grad: {y.requires_grad}")

# 方式二: detach() 从计算图中分离
z = x.detach()
print(f"detach() 后 z.requires_grad: {z.requires_grad}")

# 方式三: requires_grad_(False) 原地修改
x_no_grad = torch.randn(3)
x_no_grad.requires_grad_(True)
x_no_grad.requires_grad_(False)
print(f"requires_grad_(False): {x_no_grad.requires_grad}")


# ── 6. 手动实现一步梯度下降 ───────────────────────────────────
print("\n=== 6. 手动梯度下降演示 ===")
# 目标: 最小化 f(w) = (w - 5)²，最优解 w*=5
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1

print(f"初始 w = {w.item():.4f}")
for step in range(20):
    loss = (w - 5) ** 2
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
    w.grad.zero_()         # 每步必须手动清零梯度

print(f"收敛后 w = {w.item():.4f}  (目标: 5.0)")

print("\n示例 02 运行完成")
