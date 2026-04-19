# PyTorch 简介与环境搭建 (Introduction to PyTorch)

## 概述 Overview

本章将介绍 PyTorch 的基本概念、发展历史，以及如何搭建开发环境。完成本章后，你将：

- 理解 PyTorch 是什么以及它的核心优势
- 成功安装并配置 PyTorch 环境
- 运行第一个 PyTorch 程序
- 了解 PyTorch 的生态系统

**难度级别**：🟢 入门级

## 前置知识 Prerequisites

- Python 基础（变量、函数、类、列表等）
- 基本的命令行操作
- 了解什么是机器学习（可选，但有帮助）

## 核心概念 Core Concepts

### 什么是 PyTorch？

PyTorch 是一个开源的深度学习框架，由 Meta（原 Facebook）的 AI 研究团队开发。它的主要特点包括：

1. **动态计算图 (Dynamic Computation Graph)**
   - 也称为 "define-by-run"
   - 计算图在运行时动态构建
   - 便于调试和实现复杂模型

2. **Pythonic 设计**
   - API 设计符合 Python 习惯
   - 学习曲线平缓
   - 与 NumPy 高度兼容

3. **强大的 GPU 加速**
   - 无缝切换 CPU/GPU 计算
   - 高效的内存管理
   - 支持多 GPU 训练

4. **丰富的生态系统**
   - torchvision：计算机视觉
   - torchaudio：音频处理
   - torchtext：自然语言处理
   - HuggingFace Transformers：预训练模型

### PyTorch vs TensorFlow

| 特性 | PyTorch | TensorFlow |
|------|---------|------------|
| 计算图 | 动态 | 静态（2.0 后支持动态） |
| 调试 | 简单（使用标准 Python 调试器） | 相对复杂 |
| 学习曲线 | 平缓 | 较陡峭 |
| 生产部署 | TorchScript/ONNX | TensorFlow Serving |
| 社区 | 学术界广泛使用 | 工业界广泛使用 |

### PyTorch 核心组件

```
PyTorch 核心架构
├── torch           # 核心张量库
├── torch.nn        # 神经网络模块
├── torch.optim     # 优化器
├── torch.autograd  # 自动微分
├── torch.utils     # 工具函数
│   └── data        # 数据加载
├── torch.cuda      # GPU 支持
└── torch.jit       # JIT 编译（TorchScript）
```

## 环境搭建 Environment Setup

### 方式一：使用 pip（推荐）

```bash
# 仅 CPU 版本
pip install torch torchvision torchaudio

# CUDA 11.8 版本（如果有 NVIDIA GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 方式二：使用 conda

```bash
# 仅 CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# CUDA 11.8 版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1 版本
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 方式三：使用 Docker

```dockerfile
# 使用官方 PyTorch 镜像
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
```

### 验证安装

```python
import torch

# 基本信息
print(f"PyTorch version: {torch.__version__}")

# 检查 CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")

# 简单测试
x = torch.rand(3, 3)
print(f"\nRandom tensor:\n{x}")
print(f"Tensor device: {x.device}")
```

**预期输出示例**：
```
PyTorch version: 2.1.0
CUDA available: True
CUDA version: 11.8
cuDNN version: 8700
GPU count: 1
Current GPU: NVIDIA GeForce RTX 3080

Random tensor:
tensor([[0.4963, 0.7682, 0.0885],
        [0.1320, 0.3074, 0.6341],
        [0.4901, 0.8964, 0.4556]])
Tensor device: cpu
```

## 代码实现 Implementation

### Hello PyTorch

```python
import torch

# 创建张量
print("=== 创建张量 ===")
x = torch.tensor([1, 2, 3, 4, 5])
print(f"1D Tensor: {x}")
print(f"Shape: {x.shape}")
print(f"Dtype: {x.dtype}")

# 张量运算
print("\n=== 张量运算 ===")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"dot product = {torch.dot(a, b)}")

# GPU 加速（如果可用）
print("\n=== GPU 测试 ===")
if torch.cuda.is_available():
    # 将张量移动到 GPU
    x_gpu = x.to('cuda')
    print(f"Tensor on GPU: {x_gpu}")
    print(f"Device: {x_gpu.device}")

    # GPU 运算
    y_gpu = x_gpu * 2
    print(f"GPU computation result: {y_gpu}")

    # 移回 CPU
    y_cpu = y_gpu.to('cpu')
    print(f"Back to CPU: {y_cpu}")
else:
    print("CUDA not available, using CPU")
```

### 简单的线性回归示例

```python
import torch

# 生成模拟数据
# y = 2x + 1 + noise
torch.manual_seed(42)  # 设置随机种子以保证可重复性

X = torch.linspace(0, 10, 100).reshape(-1, 1)  # 100 个样本
y = 2 * X + 1 + torch.randn(100, 1) * 0.5  # 添加噪声

# 定义模型参数
w = torch.randn(1, requires_grad=True)  # 权重
b = torch.zeros(1, requires_grad=True)  # 偏置

# 训练参数
learning_rate = 0.01
epochs = 100

# 训练循环
print("Training Linear Regression...")
for epoch in range(epochs):
    # 前向传播：计算预测值
    y_pred = X * w + b

    # 计算损失（均方误差）
    loss = ((y_pred - y) ** 2).mean()

    # 反向传播：计算梯度
    loss.backward()

    # 更新参数（手动梯度下降）
    with torch.no_grad():  # 禁用梯度追踪
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # 清零梯度（重要！）
        w.grad.zero_()
        b.grad.zero_()

    # 每 20 轮打印一次
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print(f"\nLearned parameters:")
print(f"w = {w.item():.4f} (true value: 2.0)")
print(f"b = {b.item():.4f} (true value: 1.0)")
```

**预期输出**：
```
Training Linear Regression...
Epoch [20/100], Loss: 0.3521
Epoch [40/100], Loss: 0.2654
Epoch [60/100], Loss: 0.2523
Epoch [80/100], Loss: 0.2503
Epoch [100/100], Loss: 0.2500

Learned parameters:
w = 1.9834 (true value: 2.0)
b = 1.0892 (true value: 1.0)
```

## 深入理解 Deep Dive

### PyTorch 的设计哲学

1. **命令式编程（Imperative Programming）**
   ```python
   # PyTorch 代码就是普通的 Python 代码
   for i in range(10):
       x = torch.randn(3, 3)
       if x.sum() > 0:
           y = x * 2
       else:
           y = x / 2
   ```

2. **Eager Execution（即时执行）**
   - 每行代码立即执行，无需编译
   - 便于调试：可以使用 print、pdb 等

3. **与 NumPy 的互操作**
   ```python
   import numpy as np

   # NumPy → PyTorch
   np_array = np.array([1, 2, 3])
   tensor = torch.from_numpy(np_array)

   # PyTorch → NumPy
   tensor = torch.tensor([1, 2, 3])
   np_array = tensor.numpy()

   # 注意：它们共享内存！
   np_array[0] = 100
   print(tensor)  # tensor([100, 2, 3])
   ```

### 常见的 Device 操作

```python
import torch

# 检查可用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 创建时指定设备
x = torch.randn(3, 3, device=device)

# 移动已有张量
y = torch.randn(3, 3)
y = y.to(device)

# 更简洁的写法
z = torch.randn(3, 3).to(device)

# 检查张量所在设备
print(f"x is on: {x.device}")

# 确保所有张量在同一设备上
# 错误示例：
# result = x + torch.randn(3, 3)  # 如果 x 在 GPU 上会报错
# 正确示例：
result = x + torch.randn(3, 3, device=device)
```

## 常见问题 FAQ

### Q1: CUDA out of memory 错误

```python
# 问题：RuntimeError: CUDA out of memory
# 解决方案：

# 1. 减小 batch size
batch_size = 16  # 尝试更小的值

# 2. 清理缓存
torch.cuda.empty_cache()

# 3. 使用混合精度训练（后续章节详解）
# 4. 使用梯度累积（后续章节详解）
```

### Q2: 版本兼容问题

```python
# 检查版本兼容性
print(f"PyTorch: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")

# 如果遇到 "CUDA error: no kernel image is available"
# 说明 PyTorch CUDA 版本与 GPU 不兼容
# 解决：安装匹配的 PyTorch 版本
```

### Q3: CPU 和 GPU 张量混合运算

```python
# 错误示例
x_cpu = torch.randn(3, 3)
x_gpu = torch.randn(3, 3).cuda()
# result = x_cpu + x_gpu  # RuntimeError!

# 正确做法
result = x_cpu.cuda() + x_gpu  # 移动到同一设备
# 或者
result = x_cpu + x_gpu.cpu()
```

## 小结 Summary

本章要点：

1. **PyTorch 特点**
   - 动态计算图
   - Pythonic API
   - 强大的 GPU 支持

2. **环境搭建**
   - pip/conda 安装
   - CUDA 版本选择
   - 验证安装

3. **基本操作**
   ```python
   import torch

   # 创建张量
   x = torch.tensor([1, 2, 3])

   # GPU 操作
   if torch.cuda.is_available():
       x = x.to('cuda')

   # 张量运算
   y = x * 2
   ```

4. **自动微分**（预览）
   ```python
   x = torch.tensor([2.0], requires_grad=True)
   y = x ** 2
   y.backward()
   print(x.grad)  # tensor([4.])
   ```

## 延伸阅读 Further Reading

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch (免费电子书)](https://pytorch.org/deep-learning-with-pytorch)

## 下一步 Next

现在你已经成功搭建了 PyTorch 环境，下一章我们将深入学习 PyTorch 的核心数据结构 - **Tensor（张量）**。

[下一章：Tensor 基础操作 →](./02-tensors.md)
