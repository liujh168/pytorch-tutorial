# nn.Module 深入理解 (Understanding nn.Module)

## 概述 Overview

`nn.Module` 是 PyTorch 中所有神经网络模块的基类。深入理解它的工作原理对于构建复杂的深度学习模型至关重要。

完成本章后，你将：

- 理解 `nn.Module` 的内部机制
- 掌握参数管理和子模块管理
- 学会构建复杂的模块化网络
- 了解常用的容器类型

**难度级别**：🟡 进阶级

## 前置知识 Prerequisites

- [04-first-nn](../01-basics/04-first-nn.md) - 构建第一个神经网络

## 核心概念 Core Concepts

### nn.Module 的核心职责

1. **参数管理**：自动追踪可学习参数
2. **子模块管理**：组织网络层次结构
3. **设备管理**：支持 CPU/GPU 切换
4. **模式切换**：训练/评估模式
5. **序列化**：保存和加载模型

### 重要属性和方法

```python
# 属性
module.training        # 是否处于训练模式
module._parameters     # 直接参数字典
module._modules        # 子模块字典
module._buffers        # 缓冲区（非参数的持久状态）

# 常用方法
module.parameters()    # 迭代所有参数
module.named_parameters()  # 带名称的参数迭代
module.children()      # 直接子模块
module.modules()       # 所有模块（递归）
module.to(device)      # 移动到设备
module.train()         # 训练模式
module.eval()          # 评估模式
```

## 代码实现 Implementation

### 1. 自定义 Module 基础

```python
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    """自定义线性层"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()  # 必须调用！

        # 使用 nn.Parameter 注册可学习参数
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            # 使用 register_parameter 注册 None
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: (batch_size, in_features)
        # weight: (out_features, in_features)
        output = x @ self.weight.T
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self):
        """自定义打印信息"""
        return f'in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}'

# 使用
layer = MyLinear(10, 5)
print(layer)
# MyLinear(in_features=10, out_features=5)

x = torch.randn(32, 10)
y = layer(x)
print(f"Output shape: {y.shape}")  # torch.Size([32, 5])

# 查看参数
for name, param in layer.named_parameters():
    print(f"{name}: {param.shape}")
```

### 2. 参数 vs 缓冲区 vs 普通属性

```python
import torch
import torch.nn as nn

class ExampleModule(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. nn.Parameter - 可学习参数，参与梯度计算
        self.weight = nn.Parameter(torch.randn(10, 10))

        # 2. Buffer - 非参数的持久状态，会被保存
        self.register_buffer('running_mean', torch.zeros(10))
        # 不保存到 state_dict
        self.register_buffer('temp', torch.ones(5), persistent=False)

        # 3. 普通属性 - 不会被保存或移动
        self.some_constant = 3.14

        # 4. 子模块 - 自动注册
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        # 更新 buffer（不需要梯度）
        if self.training:
            self.running_mean = 0.9 * self.running_mean + 0.1 * x.mean(0)
        return self.linear(x)

model = ExampleModule()

# 检查注册情况
print("Parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

print("\nBuffers:")
for name, buf in model.named_buffers():
    print(f"  {name}: {buf.shape}")

print("\nModules:")
for name, mod in model.named_modules():
    print(f"  {name}: {type(mod).__name__}")

# state_dict 包含参数和持久化缓冲区
print("\nState dict keys:")
print(list(model.state_dict().keys()))
```

### 3. 容器类型

```python
import torch
import torch.nn as nn

# === nn.Sequential ===
# 按顺序执行的模块列表
sequential = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

# 可以用索引访问
print(sequential[0])  # Linear(in_features=10, out_features=64)

# 带名称的 Sequential
from collections import OrderedDict
sequential_named = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(10, 64)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(64, 10))
]))
print(sequential_named.fc1)  # 可以用名称访问

# === nn.ModuleList ===
# 当需要动态添加或索引层时使用
class DynamicNet(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

# 注意：普通 list 不会注册子模块！
class BrokenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10)]  # 错误！不会被注册

# === nn.ModuleDict ===
# 当需要用名称访问层时使用
class BranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleDict({
            'branch_a': nn.Linear(10, 5),
            'branch_b': nn.Linear(10, 8),
            'branch_c': nn.Linear(10, 3)
        })

    def forward(self, x, branch_name):
        return self.branches[branch_name](x)

# === nn.ParameterList / nn.ParameterDict ===
class ParameterExample(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.expert_weights = nn.ParameterList([
            nn.Parameter(torch.randn(10, 10))
            for _ in range(num_experts)
        ])

    def forward(self, x, expert_idx):
        return x @ self.expert_weights[expert_idx].T
```

### 4. 模块化设计模式

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 残差块 ===
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)  # 残差连接
        return x

# === 注意力块 ===
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = torch.softmax(q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5), dim=-1)
        return attn @ v

# === 组合成更大的网络 ===
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x
```

### 5. 前向钩子和反向钩子

```python
import torch
import torch.nn as nn

class HookedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = HookedModel()

# === 前向钩子 ===
activations = {}

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# 注册钩子
handle1 = model.fc1.register_forward_hook(save_activation('fc1'))

# 运行前向传播
x = torch.randn(5, 10)
output = model(x)

print(f"fc1 activation shape: {activations['fc1'].shape}")

# 移除钩子
handle1.remove()

# === 前向预处理钩子 ===
def input_modifier(module, input):
    # input 是元组
    x = input[0]
    return (x * 2,)  # 返回修改后的输入（必须是元组）

handle = model.fc1.register_forward_pre_hook(input_modifier)

# === 反向钩子 ===
gradients = {}

def save_gradient(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output[0].detach()
    return hook

handle2 = model.fc1.register_full_backward_hook(save_gradient('fc1'))

# 反向传播
output = model(x)
output.sum().backward()

print(f"fc1 gradient shape: {gradients['fc1'].shape}")

# 清理
handle.remove()
handle2.remove()
```

### 6. 自定义参数初始化

```python
import torch
import torch.nn as nn

class CustomInitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10)

        # 应用自定义初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """递归初始化所有子模块"""
        if isinstance(module, nn.Conv2d):
            # Kaiming 初始化（适合 ReLU）
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = x.mean(dim=[2, 3])  # Global average pooling
        return self.fc(x)

# 使用
model = CustomInitModel()

# 检查初始化效果
print(f"Conv weight std: {model.conv.weight.std().item():.4f}")
print(f"BN weight: {model.bn.weight.mean().item():.4f}")
```

### 7. 模型参数冻结与解冻

```python
import torch
import torch.nn as nn

# === 冻结参数 ===
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# 方法 1：设置 requires_grad = False
for param in model[0].parameters():
    param.requires_grad = False

# 方法 2：使用 requires_grad_()
model[0].requires_grad_(False)

# 检查哪些参数会被训练
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# === 只训练部分参数 ===
# 只传入需要训练的参数给优化器
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=0.001)

# === 解冻参数 ===
model[0].requires_grad_(True)

# === 常见模式：微调预训练模型 ===
class FineTunedModel(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.backbone = pretrained_model

        # 冻结主干网络
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 只训练新添加的分类头
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():  # 额外优化：跳过反向传播
            features = self.backbone(x)
        return self.classifier(features)
```

## 深入理解 Deep Dive

### nn.Module 内部机制

```python
import torch
import torch.nn as nn

# 查看 Module 的内部结构
module = nn.Linear(10, 5)

# _parameters: 直接参数
print(f"_parameters: {module._parameters.keys()}")

# _modules: 直接子模块
print(f"_modules: {module._modules.keys()}")

# _buffers: 缓冲区
print(f"_buffers: {module._buffers.keys()}")

# === __setattr__ 的魔法 ===
# 当你设置属性时，Module 会自动检测类型并注册
class MagicModule(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Parameter 自动注册到 _parameters
        self.weight = nn.Parameter(torch.randn(10))
        # nn.Module 自动注册到 _modules
        self.linear = nn.Linear(10, 5)
        # 普通张量不会被注册
        self.constant = torch.ones(5)

model = MagicModule()
print(f"Registered parameters: {list(model._parameters.keys())}")
print(f"Registered modules: {list(model._modules.keys())}")

# constant 不在 state_dict 中
print(f"State dict: {list(model.state_dict().keys())}")
```

### to() 方法的实现原理

```python
import torch
import torch.nn as nn

class UnderstandTo(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(5))
        self.register_buffer('buf', torch.ones(5))

    def forward(self, x):
        return x

model = UnderstandTo()

# to() 会递归移动所有参数和缓冲区
model.to('cuda')

print(f"Param device: {model.param.device}")
print(f"Buffer device: {model.buf.device}")

# _apply 方法
# to() 实际上调用 _apply(fn)，fn 会被应用到所有参数和缓冲区
def print_fn(tensor):
    print(f"Processing tensor of shape {tensor.shape}")
    return tensor

# model._apply(print_fn)  # 会打印所有张量的形状
```

## 常见问题 FAQ

### Q1: 参数没有被注册

```python
class BrokenModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 错误：普通张量不会被追踪
        self.weight = torch.randn(10, 10, requires_grad=True)

# 正确做法
class CorrectModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))
```

### Q2: 子模块没有被注册

```python
class BrokenModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 错误：普通 list 不会注册子模块
        self.layers = [nn.Linear(10, 10) for _ in range(3)]

# 正确做法
class CorrectModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
```

### Q3: 忘记调用 super().__init__()

```python
class BrokenModule(nn.Module):
    def __init__(self):
        # 错误：没有调用 super().__init__()
        self.linear = nn.Linear(10, 5)  # 报错！

# 正确做法
class CorrectModule(nn.Module):
    def __init__(self):
        super().__init__()  # 必须调用！
        self.linear = nn.Linear(10, 5)
```

## 小结 Summary

本章要点：

1. **Module 结构**
   ```python
   class MyModule(nn.Module):
       def __init__(self):
           super().__init__()
           self.param = nn.Parameter(...)
           self.submodule = nn.Linear(...)

       def forward(self, x):
           return self.submodule(x)
   ```

2. **容器类型**
   - `nn.Sequential`: 顺序执行
   - `nn.ModuleList`: 列表存储
   - `nn.ModuleDict`: 字典存储

3. **钩子机制**
   ```python
   module.register_forward_hook(hook_fn)
   module.register_full_backward_hook(hook_fn)
   ```

4. **参数管理**
   ```python
   model.parameters()        # 所有参数
   model.named_parameters()  # 带名称
   param.requires_grad = False  # 冻结
   ```

## 练习题 Exercises

**练习 1（🟢 入门）**: 创建一个 `ResidualBlock`，输入维度等于输出维度，前向传播为 `output = F.relu(linear(x)) + x`（残差连接），验证 shape 不变。

<details>
<summary>参考答案</summary>

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x)) + x  # 残差连接

block = ResidualBlock(64)
x = torch.randn(4, 64)
print(block(x).shape)  # torch.Size([4, 64])
```

</details>

---

**练习 2（🟡 进阶）**: 实现一个带 **共享权重** 的模型：定义一个 `nn.Linear(8, 8)` 层，在 `forward` 中使用该层两次（`x = layer(x); x = layer(x)`）。统计参数量，验证确实只有一份权重。

<details>
<summary>提示</summary>

直接在 `__init__` 中定义一个层，在 `forward` 里调用两次即可。`sum(p.numel() for p in model.parameters())` 应等于单层参数量。

</details>

<details>
<summary>参考答案</summary>

```python
class SharedWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(8, 8)   # 只定义一次

    def forward(self, x):
        x = torch.relu(self.shared(x))
        x = torch.relu(self.shared(x))  # 第二次复用同一层
        return x

model = SharedWeightModel()
print("参数量:", sum(p.numel() for p in model.parameters()))  # 8*8+8 = 72
```

</details>

---

**练习 3（🔴 挑战）**: 实现一个 `EnsembleModel`，包含 N 个子模型，`forward` 输出各子模型 logits 的**平均值**。支持在初始化时传入模型列表，并确保所有子模型被正确注册（用 `nn.ModuleList`，而非普通 Python list）。解释为什么不能用普通 list。

<details>
<summary>参考答案</summary>

```python
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        # ModuleList 会注册所有子模型；普通 list 不会，导致参数无法被 optimizer 管理
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = torch.stack([m(x) for m in self.models], dim=0)
        return outputs.mean(dim=0)

models = [nn.Linear(4, 2) for _ in range(3)]
ensemble = EnsembleModel(models)
x = torch.randn(5, 4)
print(ensemble(x).shape)  # (5, 2)
print("参数量:", sum(p.numel() for p in ensemble.parameters()))  # 3 * (4*2+2) = 30
```

</details>

## 延伸阅读 Further Reading

- [nn.Module 源码](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py)
- [PyTorch 自定义模块教程](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)

## 下一步 Next

理解了 `nn.Module` 后，下一章我们将学习如何高效地加载和预处理数据。

[下一章：数据加载与预处理 →](./02-data-loading.md)
