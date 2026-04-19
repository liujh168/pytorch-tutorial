# Tensor 基础操作 (Tensor Basics)

## 概述 Overview

Tensor（张量）是 PyTorch 中最基本的数据结构，类似于 NumPy 的 ndarray，但具有 GPU 加速和自动微分的能力。本章将全面介绍 Tensor 的创建、操作和属性。

完成本章后，你将：

- 掌握各种创建 Tensor 的方法
- 理解 Tensor 的属性（shape、dtype、device）
- 熟练进行 Tensor 的索引、切片和变形
- 掌握常用的数学运算

**难度级别**：🟢 入门级

## 前置知识 Prerequisites

- [01-introduction](./01-introduction.md) - PyTorch 简介与环境搭建
- NumPy 基础（推荐但非必需）

## 核心概念 Core Concepts

### 什么是 Tensor？

Tensor 可以理解为多维数组的推广：

```
标量（0阶张量）: 3.14
向量（1阶张量）: [1, 2, 3]
矩阵（2阶张量）: [[1, 2], [3, 4]]
3阶张量: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
```

### Tensor 的三个核心属性

```python
import torch

x = torch.randn(3, 4)

print(x.shape)   # torch.Size([3, 4]) - 形状
print(x.dtype)   # torch.float32 - 数据类型
print(x.device)  # cpu - 所在设备
```

## 代码实现 Implementation

### 1. 创建 Tensor

```python
import torch

# === 从数据创建 ===
# 从 Python 列表
x = torch.tensor([1, 2, 3, 4])
print(f"From list: {x}")

# 从 NumPy 数组
import numpy as np
np_array = np.array([[1, 2], [3, 4]])
x = torch.from_numpy(np_array)
print(f"From numpy: {x}")

# 注意：共享内存
np_array[0, 0] = 100
print(f"After modifying numpy: {x}")  # tensor 也变了

# 如果不想共享内存，使用 clone
x = torch.from_numpy(np_array).clone()

# === 指定形状创建 ===
# 全零张量
zeros = torch.zeros(3, 4)
print(f"Zeros:\n{zeros}")

# 全一张量
ones = torch.ones(2, 3)
print(f"Ones:\n{ones}")

# 指定值填充
full = torch.full((2, 2), fill_value=7.0)
print(f"Full:\n{full}")

# 单位矩阵
eye = torch.eye(3)
print(f"Identity:\n{eye}")

# 未初始化（随机内存值，速度快）
empty = torch.empty(2, 3)
print(f"Empty (uninitialized):\n{empty}")

# === 随机张量 ===
# 均匀分布 [0, 1)
rand = torch.rand(3, 3)
print(f"Uniform [0,1):\n{rand}")

# 标准正态分布
randn = torch.randn(3, 3)
print(f"Standard normal:\n{randn}")

# 指定范围的整数
randint = torch.randint(low=0, high=10, size=(3, 3))
print(f"Random integers [0,10):\n{randint}")

# 随机排列
randperm = torch.randperm(10)
print(f"Random permutation: {randperm}")

# === 序列张量 ===
# 类似 range
arange = torch.arange(0, 10, 2)  # start, end, step
print(f"Arange: {arange}")

# 等间距
linspace = torch.linspace(0, 1, 5)  # start, end, steps
print(f"Linspace: {linspace}")

# 对数等间距
logspace = torch.logspace(0, 2, 5)  # 10^0 到 10^2，5个点
print(f"Logspace: {logspace}")

# === 从已有 Tensor 创建 ===
x = torch.tensor([[1, 2], [3, 4]])

# 相同形状的零/一/随机张量
zeros_like = torch.zeros_like(x)
ones_like = torch.ones_like(x)
rand_like = torch.rand_like(x, dtype=torch.float32)

print(f"Original:\n{x}")
print(f"Zeros like:\n{zeros_like}")
print(f"Ones like:\n{ones_like}")
print(f"Rand like:\n{rand_like}")
```

### 2. 数据类型 (dtype)

```python
import torch

# === 常用数据类型 ===
# 浮点类型
x_float16 = torch.tensor([1.0], dtype=torch.float16)   # 半精度
x_float32 = torch.tensor([1.0], dtype=torch.float32)   # 单精度（默认）
x_float64 = torch.tensor([1.0], dtype=torch.float64)   # 双精度

# 整数类型
x_int8 = torch.tensor([1], dtype=torch.int8)
x_int16 = torch.tensor([1], dtype=torch.int16)
x_int32 = torch.tensor([1], dtype=torch.int32)
x_int64 = torch.tensor([1], dtype=torch.int64)  # long

# 布尔类型
x_bool = torch.tensor([True, False], dtype=torch.bool)

# === 类型转换 ===
x = torch.tensor([1.5, 2.7, 3.9])

# 方式一：使用 to()
x_int = x.to(torch.int32)
print(f"Float to int: {x_int}")  # tensor([1, 2, 3])

# 方式二：使用快捷方法
x_long = x.long()     # 转为 int64
x_float = x.float()   # 转为 float32
x_double = x.double() # 转为 float64
x_half = x.half()     # 转为 float16

# === 检查类型 ===
print(f"Is float: {x.is_floating_point()}")
print(f"Dtype: {x.dtype}")

# === 默认类型设置 ===
# 设置默认浮点类型
torch.set_default_dtype(torch.float64)
x = torch.tensor([1.0, 2.0])
print(f"Default dtype: {x.dtype}")  # float64

# 恢复默认
torch.set_default_dtype(torch.float32)
```

### 3. 设备管理 (device)

```python
import torch

# === 检查设备 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === 创建时指定设备 ===
x_cpu = torch.randn(3, 3)
x_gpu = torch.randn(3, 3, device='cuda')  # 直接在 GPU 创建

# === 移动张量 ===
# 方式一：to()
x = torch.randn(3, 3)
x = x.to(device)
x = x.to('cuda:0')  # 指定 GPU 索引

# 方式二：快捷方法
x = x.cuda()     # 移到默认 GPU
x = x.cuda(0)    # 移到 GPU 0
x = x.cpu()      # 移回 CPU

# === 检查设备 ===
print(f"Device: {x.device}")
print(f"Is CUDA: {x.is_cuda}")

# === 多 GPU 选择 ===
if torch.cuda.device_count() > 1:
    print(f"Available GPUs: {torch.cuda.device_count()}")
    x = x.to('cuda:1')  # 移到第二块 GPU

# === 同步操作 ===
# GPU 操作是异步的，有时需要同步
torch.cuda.synchronize()

# === 常见模式 ===
def create_tensor_on_device(shape, device):
    """在指定设备上创建张量"""
    return torch.randn(shape, device=device)

# 确保模型和数据在同一设备
model_device = next(model.parameters()).device  # 获取模型设备
x = x.to(model_device)
```

### 4. 索引与切片

```python
import torch

# 创建示例张量
x = torch.arange(12).reshape(3, 4)
print(f"Original:\n{x}")
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# === 基本索引 ===
print(f"x[0]: {x[0]}")          # 第一行
print(f"x[0, 0]: {x[0, 0]}")    # 元素 (0,0)
print(f"x[-1]: {x[-1]}")        # 最后一行
print(f"x[-1, -1]: {x[-1, -1]}")# 最后一个元素

# === 切片 ===
print(f"x[:2]: \n{x[:2]}")       # 前两行
print(f"x[:, :2]: \n{x[:, :2]}") # 前两列
print(f"x[1:3, 1:3]: \n{x[1:3, 1:3]}")  # 子矩阵

# === 步长切片 ===
print(f"x[::2]: \n{x[::2]}")     # 每隔一行
print(f"x[:, ::2]: \n{x[:, ::2]}")  # 每隔一列

# === 高级索引 ===
# 使用列表索引
indices = [0, 2]
print(f"x[indices]: \n{x[indices]}")  # 第0和第2行

# 使用张量索引
idx = torch.tensor([0, 2])
print(f"x[idx]: \n{x[idx]}")

# 布尔索引
mask = x > 5
print(f"Mask:\n{mask}")
print(f"x[mask]: {x[mask]}")  # 所有大于5的元素

# 花式索引
row_idx = torch.tensor([0, 1, 2])
col_idx = torch.tensor([0, 1, 2])
print(f"Diagonal: {x[row_idx, col_idx]}")  # 对角线元素

# === 修改元素 ===
x_copy = x.clone()
x_copy[0, 0] = 100
x_copy[:, -1] = 0
print(f"Modified:\n{x_copy}")

# 使用条件修改
x_copy[x_copy > 50] = 50
print(f"Clipped:\n{x_copy}")
```

### 5. 形状操作

```python
import torch

x = torch.arange(12)
print(f"Original: {x}, shape: {x.shape}")

# === reshape ===
y = x.reshape(3, 4)
print(f"Reshaped (3,4):\n{y}")

y = x.reshape(2, 2, 3)
print(f"Reshaped (2,2,3):\n{y}")

# 使用 -1 自动推断
y = x.reshape(-1, 4)  # 自动计算行数
print(f"Reshaped (-1,4):\n{y}")

# === view (更高效，但需要连续内存) ===
y = x.view(3, 4)
print(f"View (3,4):\n{y}")

# 注意：view 和 reshape 的区别
# view 要求内存连续，reshape 会在必要时复制数据
# 推荐：先用 view，不行再用 reshape

# === 扁平化 ===
y = torch.randn(2, 3, 4)
flat = y.flatten()  # 完全扁平化
print(f"Flatten: {flat.shape}")  # torch.Size([24])

flat_partial = y.flatten(start_dim=1)  # 从第1维开始
print(f"Partial flatten: {flat_partial.shape}")  # torch.Size([2, 12])

# === 增加/移除维度 ===
x = torch.randn(3, 4)
print(f"Original shape: {x.shape}")

# 增加维度
y = x.unsqueeze(0)  # 在第0维增加
print(f"Unsqueeze(0): {y.shape}")  # torch.Size([1, 3, 4])

y = x.unsqueeze(-1)  # 在最后增加
print(f"Unsqueeze(-1): {y.shape}")  # torch.Size([3, 4, 1])

# 移除维度（只能移除大小为1的维度）
y = torch.randn(1, 3, 1, 4)
z = y.squeeze()  # 移除所有大小为1的维度
print(f"Squeeze: {z.shape}")  # torch.Size([3, 4])

z = y.squeeze(0)  # 只移除第0维
print(f"Squeeze(0): {z.shape}")  # torch.Size([3, 1, 4])

# === 转置和维度交换 ===
x = torch.randn(2, 3, 4)

# 转置（仅2D）
y = torch.randn(3, 4)
print(f"Transpose: {y.T.shape}")  # torch.Size([4, 3])

# 交换两个维度
z = x.transpose(0, 2)
print(f"Transpose(0,2): {z.shape}")  # torch.Size([4, 3, 2])

# 任意维度重排
z = x.permute(2, 0, 1)
print(f"Permute(2,0,1): {z.shape}")  # torch.Size([4, 2, 3])

# === 拼接 ===
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# cat: 在现有维度上拼接
c = torch.cat([a, b], dim=0)
print(f"Cat dim=0: {c.shape}")  # torch.Size([4, 3])

c = torch.cat([a, b], dim=1)
print(f"Cat dim=1: {c.shape}")  # torch.Size([2, 6])

# stack: 在新维度上堆叠
c = torch.stack([a, b], dim=0)
print(f"Stack dim=0: {c.shape}")  # torch.Size([2, 2, 3])

# === 分割 ===
x = torch.randn(6, 4)

# 等分
chunks = x.chunk(3, dim=0)  # 分成3份
print(f"Chunk: {[c.shape for c in chunks]}")

# 按指定大小分
splits = x.split([2, 4], dim=0)  # 分成大小为2和4的两份
print(f"Split: {[s.shape for s in splits]}")
```

### 6. 数学运算

```python
import torch

# === 逐元素运算 ===
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 基本运算
print(f"Add: {a + b}")
print(f"Sub: {a - b}")
print(f"Mul: {a * b}")
print(f"Div: {a / b}")
print(f"Power: {a ** 2}")

# 数学函数
print(f"Sqrt: {torch.sqrt(a)}")
print(f"Exp: {torch.exp(a)}")
print(f"Log: {torch.log(a)}")
print(f"Sin: {torch.sin(a)}")

# 原地操作（节省内存）
a.add_(1)  # a = a + 1
print(f"In-place add: {a}")

# === 聚合运算 ===
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(f"Sum: {x.sum()}")
print(f"Sum dim=0: {x.sum(dim=0)}")  # 按列求和
print(f"Sum dim=1: {x.sum(dim=1)}")  # 按行求和

print(f"Mean: {x.mean()}")
print(f"Std: {x.std()}")
print(f"Var: {x.var()}")

print(f"Max: {x.max()}")
print(f"Min: {x.min()}")
print(f"Argmax: {x.argmax()}")  # 最大值索引

# 保持维度
print(f"Sum keepdim: {x.sum(dim=1, keepdim=True)}")

# === 矩阵运算 ===
A = torch.randn(2, 3)
B = torch.randn(3, 4)

# 矩阵乘法
C = torch.matmul(A, B)
C = A @ B  # 等价写法
print(f"Matrix multiply: {C.shape}")

# 批量矩阵乘法
batch_A = torch.randn(10, 2, 3)
batch_B = torch.randn(10, 3, 4)
batch_C = torch.bmm(batch_A, batch_B)
print(f"Batch matmul: {batch_C.shape}")

# 点积
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"Dot product: {torch.dot(a, b)}")

# 外积
print(f"Outer product:\n{torch.outer(a, b)}")

# === 线性代数 ===
A = torch.randn(3, 3)

# 转置
print(f"Transpose:\n{A.T}")

# 逆矩阵
A_inv = torch.linalg.inv(A)
print(f"Inverse:\n{A_inv}")

# 行列式
det = torch.linalg.det(A)
print(f"Determinant: {det}")

# 特征值和特征向量
eigenvalues, eigenvectors = torch.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")

# SVD 分解
U, S, V = torch.linalg.svd(A)
print(f"SVD: U{U.shape}, S{S.shape}, V{V.shape}")

# === 比较运算 ===
x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([5, 4, 3, 2, 1])

print(f"Equal: {x == y}")
print(f"Greater: {x > y}")
print(f"Max element-wise: {torch.maximum(x, y)}")

# 近似相等
a = torch.tensor([1.0, 2.0])
b = torch.tensor([1.0001, 1.9999])
print(f"Allclose: {torch.allclose(a, b, atol=1e-3)}")
```

### 7. 广播机制 (Broadcasting)

```python
import torch

# 广播规则：
# 1. 从最后一维开始比较
# 2. 每个维度要么相等，要么其中一个为1
# 3. 缺失的维度视为1

# === 基本广播 ===
a = torch.randn(3, 4)
b = torch.randn(4)  # 自动扩展为 (3, 4)
c = a + b
print(f"(3,4) + (4) = {c.shape}")

# === 更复杂的广播 ===
a = torch.randn(3, 1)
b = torch.randn(1, 4)
c = a + b
print(f"(3,1) + (1,4) = {c.shape}")  # (3, 4)

# === 显式扩展 ===
a = torch.randn(3, 1)
b = a.expand(3, 4)  # 扩展到 (3, 4)
print(f"Expand: {b.shape}")

# 使用 -1 保持原始大小
b = a.expand(-1, 4)
print(f"Expand with -1: {b.shape}")

# 重复（实际复制数据）
a = torch.tensor([[1, 2, 3]])
b = a.repeat(3, 2)  # 行方向重复3次，列方向重复2次
print(f"Repeat:\n{b}")

# === 广播的应用 ===
# 标准化
x = torch.randn(32, 10)  # batch_size=32, features=10
mean = x.mean(dim=0)     # (10,)
std = x.std(dim=0)       # (10,)
x_normalized = (x - mean) / std  # 广播
print(f"Normalized shape: {x_normalized.shape}")
```

## 深入理解 Deep Dive

### 内存布局

```python
import torch

# === 连续性 (Contiguity) ===
x = torch.randn(3, 4)
print(f"Is contiguous: {x.is_contiguous()}")  # True

# 转置后不连续
y = x.T
print(f"After transpose: {y.is_contiguous()}")  # False

# 使 tensor 连续
y = y.contiguous()
print(f"After contiguous(): {y.is_contiguous()}")  # True

# === Stride（步长）===
x = torch.randn(3, 4)
print(f"Shape: {x.shape}")
print(f"Stride: {x.stride()}")  # (4, 1) - 每个维度移动需要跳过的元素数

# 转置改变 stride 而非数据
y = x.T
print(f"Transposed stride: {y.stride()}")  # (1, 4)

# === 共享存储 ===
x = torch.randn(4)
y = x.view(2, 2)
y[0, 0] = 100
print(f"x after modifying y: {x}")  # x 也变了

# 使用 clone 创建独立副本
y = x.clone().view(2, 2)
y[0, 0] = 200
print(f"x after modifying clone: {x}")  # x 不变
```

### 性能优化技巧

```python
import torch

# === 预分配内存 ===
# 不好的做法
result = []
for i in range(100):
    result.append(torch.randn(100))
result = torch.stack(result)

# 好的做法
result = torch.empty(100, 100)
for i in range(100):
    result[i] = torch.randn(100)

# === 使用原地操作 ===
x = torch.randn(1000, 1000)

# 非原地（创建新张量）
y = x + 1

# 原地（修改原张量）
x.add_(1)

# === 避免不必要的复制 ===
# 检查是否需要 contiguous
x = torch.randn(3, 4).T
if not x.is_contiguous():
    x = x.contiguous()  # 仅在需要时调用

# === 使用 torch.no_grad() ===
# 推理时禁用梯度追踪
with torch.no_grad():
    y = model(x)  # 不构建计算图，节省内存
```

## 常见问题 FAQ

### Q1: view 和 reshape 的区别？

```python
# view: 要求内存连续，返回视图（共享内存）
# reshape: 尽可能返回视图，必要时复制数据

x = torch.randn(3, 4)
y = x.T  # 不连续

# y.view(12)  # 报错！
z = y.reshape(12)  # 正常，但会复制数据
```

### Q2: 原地操作的注意事项

```python
# 原地操作会影响梯度计算
x = torch.randn(3, requires_grad=True)
y = x ** 2
# x.add_(1)  # RuntimeError: 不能修改需要梯度的张量

# 解决方案
x_no_grad = x.detach()
x_no_grad.add_(1)  # 这会影响 x，但不影响梯度计算
```

### Q3: 如何高效地处理变长序列？

```python
# 使用 pad_sequence
from torch.nn.utils.rnn import pad_sequence

seqs = [torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
        torch.tensor([6, 7, 8, 9])]

padded = pad_sequence(seqs, batch_first=True, padding_value=0)
print(f"Padded:\n{padded}")
```

## 小结 Summary

本章要点：

1. **创建 Tensor**
   ```python
   torch.tensor([1, 2, 3])      # 从数据
   torch.zeros(3, 4)            # 全零
   torch.randn(3, 4)            # 随机
   torch.arange(0, 10, 2)       # 序列
   ```

2. **核心属性**
   ```python
   x.shape   # 形状
   x.dtype   # 数据类型
   x.device  # 设备
   ```

3. **形状操作**
   ```python
   x.reshape(3, 4)    # 重塑
   x.unsqueeze(0)     # 增加维度
   x.squeeze()        # 移除维度
   x.permute(2, 0, 1) # 维度重排
   ```

4. **数学运算**
   ```python
   a + b, a * b       # 逐元素
   a @ b              # 矩阵乘法
   x.sum(dim=0)       # 聚合
   ```

## 延伸阅读 Further Reading

- [PyTorch Tensor 官方文档](https://pytorch.org/docs/stable/tensors.html)
- [NumPy 与 PyTorch 对比](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)

## 下一步 Next

掌握了 Tensor 基础操作后，下一章我们将学习 PyTorch 最强大的特性之一 - **自动微分（Autograd）**。

[下一章：自动微分机制详解 →](./03-autograd.md)
