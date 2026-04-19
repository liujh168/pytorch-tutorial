# 自动微分机制详解 (Autograd)

## 概述 Overview

自动微分（Automatic Differentiation）是 PyTorch 的核心特性，它能够自动计算张量运算的梯度。理解 autograd 对于掌握深度学习训练过程至关重要。

完成本章后，你将：

- 理解计算图的概念
- 掌握梯度的自动计算
- 学会控制梯度流动
- 能够调试和优化梯度计算

**难度级别**：🟡 进阶级

## 前置知识 Prerequisites

- [02-tensors](./02-tensors.md) - Tensor 基础操作
- 基本的微积分知识（导数、链式法则）

## 核心概念 Core Concepts

### 什么是自动微分？

自动微分是一种精确计算导数的技术，不同于：
- **数值微分**：通过有限差分近似，有截断误差
- **符号微分**：生成导数的解析表达式，可能很复杂

自动微分通过跟踪运算并应用链式法则，精确计算导数。

### 计算图 (Computation Graph)

```
前向传播（Forward Pass）：
x → [mul by w] → y → [add b] → z → [square] → loss

反向传播（Backward Pass）：
∂loss/∂x ← ∂loss/∂y ← ∂loss/∂z ← ∂loss/∂loss
```

PyTorch 在前向传播时构建计算图，在调用 `.backward()` 时进行反向传播计算梯度。

### 核心组件

```python
# 1. requires_grad: 标记需要计算梯度的张量
x = torch.tensor([2.0], requires_grad=True)

# 2. grad_fn: 记录创建张量的操作
y = x ** 2
print(y.grad_fn)  # <PowBackward0 object>

# 3. grad: 存储计算得到的梯度
y.backward()
print(x.grad)  # tensor([4.])

# 4. backward(): 触发反向传播
```

## 代码实现 Implementation

### 1. 基本梯度计算

```python
import torch

# === 标量梯度 ===
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

# 计算 dy/dx
y.backward()
print(f"x = {x.item()}")
print(f"y = {y.item()}")
print(f"dy/dx = {x.grad.item()}")  # 2x + 3 = 7

# === 向量梯度 ===
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()  # 需要标量才能调用 backward()

z.backward()
print(f"x = {x}")
print(f"dz/dx = {x.grad}")  # [2, 4, 6]

# === 多步计算 ===
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
z = y * y * 3  # z = 12x²
out = z.mean()  # out = 6x² 的均值

out.backward()
print(f"Gradient: {x.grad}")  # [6, 12]

# === 链式法则演示 ===
# f(x) = sin(x²)
# df/dx = cos(x²) * 2x
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
z = torch.sin(y)

z.backward()
expected = torch.cos(x ** 2) * 2 * x
print(f"Computed: {x.grad}")
print(f"Expected: {expected}")
```

### 2. 多变量梯度

```python
import torch

# === 多输入单输出 ===
x = torch.tensor([1.0], requires_grad=True)
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

y = w * x + b  # y = 2*1 + 3 = 5

y.backward()
print(f"dy/dx = {x.grad}")  # 2 (w 的值)
print(f"dy/dw = {w.grad}")  # 1 (x 的值)
print(f"dy/db = {b.grad}")  # 1

# === 多输入多输出（Jacobian）===
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2  # 简单的向量运算

# backward 需要标量，所以传入梯度向量
# 这相当于计算 Jacobian-vector product
v = torch.tensor([1.0, 0.1, 0.01])
y.backward(v)
print(f"Gradient with v: {x.grad}")  # [2, 0.2, 0.02]

# === 计算完整 Jacobian ===
def jacobian(y, x):
    """计算 Jacobian 矩阵"""
    jac = []
    for i in range(y.shape[0]):
        grad_output = torch.zeros_like(y)
        grad_output[i] = 1.0
        y.backward(grad_output, retain_graph=True)
        jac.append(x.grad.clone())
        x.grad.zero_()
    return torch.stack(jac)

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.stack([x[0] ** 2, x[0] * x[1], x[1] ** 2])
# y = [x₀², x₀x₁, x₁²]

J = jacobian(y, x)
print(f"Jacobian:\n{J}")
# [[2x₀, 0],
#  [x₁, x₀],
#  [0, 2x₁]]
```

### 3. 梯度累积与清零

```python
import torch

# === 梯度会累积 ===
x = torch.tensor([1.0], requires_grad=True)

# 第一次 backward
y = x ** 2
y.backward()
print(f"First backward: {x.grad}")  # 2

# 第二次 backward（梯度累积！）
y = x ** 2
y.backward()
print(f"After second backward: {x.grad}")  # 4 (2 + 2)

# === 清零梯度 ===
x.grad.zero_()
print(f"After zero_(): {x.grad}")  # 0

# === 优化器中的梯度清零 ===
# 这就是为什么训练循环中需要 optimizer.zero_grad()
model = torch.nn.Linear(10, 5)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(3):
    optimizer.zero_grad()  # 清零梯度（重要！）
    x = torch.randn(32, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

# === 梯度累积的正确用法（大 batch 模拟）===
# 当显存不足时，可以用多个小 batch 累积梯度
accumulation_steps = 4
for i, (x, y) in enumerate(dataloader):
    output = model(x)
    loss = criterion(output, y) / accumulation_steps  # 平均
    loss.backward()  # 梯度累积

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. 控制梯度追踪

```python
import torch

# === requires_grad 属性 ===
x = torch.randn(3, 3)
print(f"Default requires_grad: {x.requires_grad}")  # False

x.requires_grad_(True)  # 原地修改
print(f"After requires_grad_(): {x.requires_grad}")  # True

# === torch.no_grad() 上下文 ===
x = torch.randn(3, 3, requires_grad=True)
y = x * 2
print(f"y requires_grad: {y.requires_grad}")  # True

with torch.no_grad():
    z = x * 2
    print(f"z requires_grad (in no_grad): {z.requires_grad}")  # False

# 常用于推理阶段
model.eval()
with torch.no_grad():
    predictions = model(test_data)

# === torch.set_grad_enabled() ===
is_training = False
with torch.set_grad_enabled(is_training):
    y = model(x)  # 根据 is_training 决定是否追踪梯度

# === detach() ===
x = torch.randn(3, 3, requires_grad=True)
y = x ** 2
z = y.detach()  # z 不再追踪梯度

print(f"y requires_grad: {y.requires_grad}")  # True
print(f"z requires_grad: {z.requires_grad}")  # False
print(f"z 和 y 共享数据: {z.data_ptr() == y.data_ptr()}")  # True

# 常用于：
# 1. 冻结部分网络
# 2. 从计算图中提取值
# 3. 避免计算图过大

# === retain_graph ===
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

# 默认 backward 后计算图被销毁
y.backward(retain_graph=True)
print(f"First grad: {x.grad}")

x.grad.zero_()
y.backward()  # 可以再次 backward
print(f"Second grad: {x.grad}")
```

### 5. 高阶导数

```python
import torch

# === 二阶导数 ===
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3  # y = x³

# 一阶导数
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = {dy_dx}")  # 3x² = 12

# 二阶导数
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"d²y/dx² = {d2y_dx2}")  # 6x = 12

# === 使用 autograd.grad ===
# 更灵活的梯度计算接口
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# 计算梯度，不修改 x.grad
grad = torch.autograd.grad(
    outputs=y.sum(),
    inputs=x,
    create_graph=False,  # 如果需要高阶导数，设为 True
    retain_graph=False   # 是否保留计算图
)[0]
print(f"Grad via autograd.grad: {grad}")

# === Hessian 矩阵 ===
def compute_hessian(f, x):
    """计算 Hessian 矩阵"""
    n = x.shape[0]
    hessian = torch.zeros(n, n)

    # 计算梯度
    grad = torch.autograd.grad(f, x, create_graph=True)[0]

    for i in range(n):
        grad2 = torch.autograd.grad(
            grad[i], x, retain_graph=True
        )[0]
        hessian[i] = grad2

    return hessian

x = torch.tensor([1.0, 2.0], requires_grad=True)
f = x[0] ** 2 + x[1] ** 2 + x[0] * x[1]  # f = x₀² + x₁² + x₀x₁

H = compute_hessian(f, x)
print(f"Hessian:\n{H}")
# [[2, 1],
#  [1, 2]]
```

### 6. 自定义 autograd 函数

```python
import torch
from torch.autograd import Function

class MyReLU(Function):
    """自定义 ReLU 函数"""

    @staticmethod
    def forward(ctx, input):
        """
        前向传播
        ctx: 上下文对象，用于保存信息供反向传播使用
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        grad_output: 上游梯度
        返回对应每个输入的梯度
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# 使用自定义函数
x = torch.randn(5, requires_grad=True)
my_relu = MyReLU.apply
y = my_relu(x)
y.sum().backward()
print(f"Input: {x.data}")
print(f"Gradient: {x.grad}")

# === 更复杂的例子：带参数的函数 ===
class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        return input @ weight.T + bias

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_output @ weight
        grad_weight = grad_output.T @ input
        grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

# 使用
x = torch.randn(10, 5, requires_grad=True)
w = torch.randn(3, 5, requires_grad=True)
b = torch.randn(3, requires_grad=True)

linear_fn = LinearFunction.apply
y = linear_fn(x, w, b)
loss = y.sum()
loss.backward()

print(f"x.grad shape: {x.grad.shape}")
print(f"w.grad shape: {w.grad.shape}")
print(f"b.grad shape: {b.grad.shape}")

# === 梯度检验 ===
from torch.autograd import gradcheck

# 检验自定义函数的梯度是否正确
input = torch.randn(5, requires_grad=True, dtype=torch.double)
test = gradcheck(MyReLU.apply, input, eps=1e-6, atol=1e-4)
print(f"Gradient check passed: {test}")
```

### 7. 梯度钩子 (Hooks)

```python
import torch

# === Tensor 钩子 ===
def print_grad(grad):
    """打印梯度的钩子函数"""
    print(f"Gradient: {grad}")
    return grad  # 可以修改并返回新梯度

x = torch.randn(3, requires_grad=True)
x.register_hook(print_grad)

y = x * 2
z = y.sum()
z.backward()  # 会打印梯度

# === 修改梯度的钩子 ===
def scale_grad(grad):
    """梯度缩放"""
    return grad * 0.1

x = torch.randn(3, requires_grad=True)
handle = x.register_hook(scale_grad)

y = x ** 2
y.sum().backward()
print(f"Scaled gradient: {x.grad}")

# 移除钩子
handle.remove()

# === Module 钩子 ===
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MyModel()

# 前向钩子：在 forward 之后执行
def forward_hook(module, input, output):
    print(f"{module.__class__.__name__}")
    print(f"  Input shape: {input[0].shape}")
    print(f"  Output shape: {output.shape}")

# 反向钩子：在 backward 时执行
def backward_hook(module, grad_input, grad_output):
    print(f"Backward through {module.__class__.__name__}")
    print(f"  grad_output: {grad_output[0].shape}")

# 注册钩子
handle1 = model.fc1.register_forward_hook(forward_hook)
handle2 = model.fc1.register_full_backward_hook(backward_hook)

# 前向和反向传播
x = torch.randn(32, 10)
y = model(x)
y.sum().backward()

# 清理
handle1.remove()
handle2.remove()
```

## 深入理解 Deep Dive

### 计算图的动态性

```python
import torch

# PyTorch 的计算图是动态的，每次前向传播都会重新构建

x = torch.randn(5, requires_grad=True)

for i in range(3):
    # 每次迭代可以有不同的计算图
    if i % 2 == 0:
        y = x ** 2
    else:
        y = x ** 3

    y.sum().backward()
    print(f"Iteration {i}, grad: {x.grad}")
    x.grad.zero_()

# 这种灵活性允许：
# 1. 动态的网络结构
# 2. 条件计算
# 3. 循环神经网络中的可变序列长度
```

### 内存管理

```python
import torch

# === 计算图占用内存 ===
# 中间结果会被保存用于反向传播

x = torch.randn(1000, 1000, requires_grad=True)
y = x @ x @ x  # 保存多个中间结果

# === checkpoint 技术 ===
# 用计算换内存
from torch.utils.checkpoint import checkpoint

class HeavyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(1000, 1000) for _ in range(10)
        ])

    def forward(self, x):
        for layer in self.layers:
            # 使用 checkpoint，不保存中间激活
            x = checkpoint(layer, x, use_reentrant=False)
        return x

# === 清理未使用的张量 ===
del y
torch.cuda.empty_cache()  # 清理 GPU 缓存
```

### 常见的梯度问题

```python
import torch

# === 梯度消失 ===
x = torch.randn(1, requires_grad=True)
y = x
for _ in range(100):
    y = torch.sigmoid(y)  # sigmoid 的梯度最大为 0.25

y.backward()
print(f"Gradient after 100 sigmoids: {x.grad}")  # 接近 0

# === 梯度爆炸 ===
x = torch.randn(1, requires_grad=True)
y = x
for _ in range(100):
    y = y * 2  # 指数增长

# y.backward()  # 可能 overflow

# === 解决方案 ===
# 1. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 更好的激活函数
y = torch.relu(x)  # ReLU 没有梯度消失问题（对正值）

# 3. 残差连接
# y = F.relu(layer(x)) + x  # 梯度可以直接流过

# 4. 批归一化
# 使激活保持在合理范围
```

## 常见问题 FAQ

### Q1: "RuntimeError: Trying to backward through the graph a second time"

```python
# 问题：计算图默认只能 backward 一次
x = torch.randn(3, requires_grad=True)
y = x ** 2
y.sum().backward()
# y.sum().backward()  # 报错！

# 解决方案 1：retain_graph=True
y = x ** 2
y.sum().backward(retain_graph=True)
y.sum().backward()  # OK

# 解决方案 2：重新计算
x.grad.zero_()
y = x ** 2
y.sum().backward()
```

### Q2: "element 0 of tensors does not require grad"

```python
# 问题：对不需要梯度的张量调用 backward
x = torch.randn(3)  # requires_grad=False
y = x ** 2
# y.sum().backward()  # 报错！

# 解决方案
x = torch.randn(3, requires_grad=True)
```

### Q3: 梯度为 None

```python
# 问题：某些参数的梯度为 None
x = torch.randn(3, requires_grad=True)
y = x * 2
z = x.detach() ** 2  # detach 后的操作不影响 x 的梯度

z.sum().backward()
print(x.grad)  # None，因为 z 和 x 的梯度流断开了
```

## 小结 Summary

本章要点：

1. **基本概念**
   ```python
   x = torch.tensor([2.0], requires_grad=True)
   y = x ** 2
   y.backward()
   print(x.grad)  # tensor([4.])
   ```

2. **控制梯度追踪**
   ```python
   with torch.no_grad():
       # 推理时禁用梯度
       pass

   x.detach()  # 从计算图分离
   ```

3. **梯度管理**
   ```python
   optimizer.zero_grad()  # 清零梯度
   loss.backward()        # 反向传播
   optimizer.step()       # 更新参数
   ```

4. **高级技巧**
   ```python
   # 高阶导数
   grad = torch.autograd.grad(y, x, create_graph=True)

   # 梯度钩子
   x.register_hook(lambda g: g * 0.1)
   ```

## 练习题 Exercises

**练习 1（🟢 入门）**: 用 autograd 计算函数 `f(x) = sin(x²) + cos(x)` 在 `x = π/4` 处的导数，并与解析解对比。

<details>
<summary>提示</summary>

解析解：`f'(x) = 2x·cos(x²) - sin(x)`

</details>

<details>
<summary>参考答案</summary>

```python
import torch, math

x = torch.tensor(math.pi / 4, requires_grad=True)
f = torch.sin(x**2) + torch.cos(x)
f.backward()

autograd_val = x.grad.item()
analytic_val = 2 * x.item() * math.cos(x.item()**2) - math.sin(x.item())
print(f"Autograd: {autograd_val:.6f}")
print(f"解析解:   {analytic_val:.6f}")
```

</details>

---

**练习 2（🟡 进阶）**: 实现一个**梯度下降**找最小值的过程，最小化 `f(x, y) = (x-3)² + (y+2)²`，初始点 `(x₀, y₀) = (0, 0)`，学习率 0.1，迭代 100 步，验证收敛到 `(3, -2)`。

<details>
<summary>提示</summary>

每步：计算 loss → backward → 用 `no_grad` 更新 → 手动清零梯度。

</details>

<details>
<summary>参考答案</summary>

```python
x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(100):
    loss = (x - 3)**2 + (y + 2)**2
    loss.backward()
    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
    x.grad.zero_()
    y.grad.zero_()

print(f"x={x.item():.4f} (目标: 3.0)")
print(f"y={y.item():.4f} (目标: -2.0)")
```

</details>

---

**练习 3（🔴 挑战）**: 实现一个自定义 autograd 函数 `LeakyReLU`（使用 `torch.autograd.Function`），当 `x > 0` 时导数为 1，否则为 `alpha=0.01`，并与 `nn.LeakyReLU` 的结果对比。

<details>
<summary>提示</summary>

继承 `torch.autograd.Function`，实现 `forward` 和 `backward` 静态方法，在 `ctx.save_for_backward` 中保存输入。

</details>

<details>
<summary>参考答案</summary>

```python
import torch

class LeakyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=0.01):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return torch.where(x > 0, x, alpha * x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = torch.where(x > 0, torch.ones_like(x), torch.full_like(x, ctx.alpha))
        return grad_output * grad, None  # None 对应 alpha（非 Tensor）

x = torch.randn(5, requires_grad=True)
out = LeakyReLUFunction.apply(x)
out.sum().backward()
print("自定义梯度:", x.grad)
```

</details>

## 延伸阅读 Further Reading

- [PyTorch Autograd 教程](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Autograd 机制详解](https://pytorch.org/docs/stable/notes/autograd.html)
- [自定义 autograd 函数](https://pytorch.org/docs/stable/notes/extending.html)

## 下一步 Next

掌握了自动微分后，下一章我们将综合运用所学知识，构建你的**第一个神经网络**。

[下一章：构建第一个神经网络 →](./04-first-nn.md)
