# 模型导出 (Model Export)

## 概述 Overview

训练完成的模型需要导出为可部署的格式。本章介绍 TorchScript 和 ONNX 两种主流导出方式。

## 代码实现 Implementation

### 1. TorchScript 导出

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleModel(784, 256, 10)
model.eval()

# === 方法1: Tracing ===
# 适用于没有控制流的模型
example_input = torch.randn(1, 784)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# === 方法2: Scripting ===
# 适用于有控制流的模型
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# 加载和使用
loaded_model = torch.jit.load("model_traced.pt")
output = loaded_model(example_input)
print(f"Output shape: {output.shape}")
```

### 2. 处理动态控制流

```python
import torch
import torch.nn as nn

class DynamicModel(nn.Module):
    """包含条件分支的模型"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(20, 5)

    def forward(self, x, use_branch_a: bool = True):
        x = self.fc1(x)
        x = torch.relu(x)

        if use_branch_a:
            return self.fc2(x)
        else:
            return self.fc3(x)

model = DynamicModel()
model.eval()

# Tracing 无法处理动态控制流，使用 Scripting
scripted = torch.jit.script(model)

# 测试两个分支
x = torch.randn(1, 10)
out_a = scripted(x, True)
out_b = scripted(x, False)
print(f"Branch A: {out_a.shape}, Branch B: {out_b.shape}")

scripted.save("dynamic_model.pt")
```

### 3. ONNX 导出

```python
import torch
import torch.nn as nn
import torch.onnx

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CNNModel()
model.eval()

# 导出 ONNX
dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("ONNX model exported successfully!")
```

### 4. ONNX 模型验证与使用

```python
import onnx
import onnxruntime as ort
import numpy as np

# 验证 ONNX 模型
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# 使用 ONNX Runtime 推理
ort_session = ort.InferenceSession("model.onnx")

# 获取输入输出信息
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

print(f"Input name: {input_name}")
print(f"Output name: {output_name}")

# 推理
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
outputs = ort_session.run([output_name], {input_name: input_data})

print(f"Output shape: {outputs[0].shape}")
```

### 5. 导出 Transformer 模型

```python
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)

model = SimpleTransformer(
    vocab_size=10000,
    d_model=256,
    nhead=4,
    num_layers=2
)
model.eval()

# 导出为 TorchScript
example_input = torch.randint(0, 10000, (1, 32))

# 使用 trace（注意：对于变长序列可能需要 script）
with torch.no_grad():
    traced = torch.jit.trace(model, example_input)

traced.save("transformer.pt")

# 导出为 ONNX
torch.onnx.export(
    model,
    example_input,
    "transformer.onnx",
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch', 1: 'sequence'}
    },
    opset_version=14
)

print("Transformer model exported!")
```

### 6. 模型优化

```python
import torch
import torch.nn as nn

# TorchScript 优化
model = SimpleModel(784, 256, 10)
model.eval()

example_input = torch.randn(1, 784)
traced = torch.jit.trace(model, example_input)

# 冻结模型（将参数内联）
frozen = torch.jit.freeze(traced)

# 优化推理
optimized = torch.jit.optimize_for_inference(frozen)

optimized.save("model_optimized.pt")

# 比较性能
import time

def benchmark(model, input_data, num_runs=100):
    # Warmup
    for _ in range(10):
        model(input_data)

    start = time.time()
    for _ in range(num_runs):
        model(input_data)
    end = time.time()

    return (end - start) / num_runs * 1000  # ms

input_data = torch.randn(32, 784)

time_original = benchmark(traced, input_data)
time_optimized = benchmark(optimized, input_data)

print(f"Original: {time_original:.3f} ms")
print(f"Optimized: {time_optimized:.3f} ms")
print(f"Speedup: {time_original / time_optimized:.2f}x")
```

## 导出格式对比

| 特性 | TorchScript | ONNX |
|------|-------------|------|
| 跨框架 | ❌ PyTorch only | ✅ 多框架支持 |
| 动态控制流 | ✅ 通过 script | ⚠️ 有限支持 |
| 性能 | ✅ 优秀 | ✅ 优秀 |
| 生态 | PyTorch 原生 | 广泛 |
| 调试 | ✅ 易于调试 | ⚠️ 较难 |

## 下一步 Next

[下一章：模型量化 →](./02-quantization.md)
