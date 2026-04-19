# 模型量化 (Model Quantization)

## 概述 Overview

量化通过降低数值精度（如 FP32 → INT8）来减小模型大小和加速推理。本章介绍三种量化方法。

## 代码实现 Implementation

### 1. 动态量化 (Dynamic Quantization)

```python
import torch
import torch.nn as nn
import torch.quantization

class LSTMModel(nn.Module):
    """LSTM 模型示例"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

# 创建模型
model = LSTMModel(10000, 256, 512, 2)
model.eval()

# 动态量化（主要针对 Linear 和 LSTM）
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},  # 量化这些层
    dtype=torch.qint8
)

# 比较模型大小
def get_model_size(model):
    torch.save(model.state_dict(), "temp.pt")
    import os
    size = os.path.getsize("temp.pt") / (1024 * 1024)
    os.remove("temp.pt")
    return size

print(f"Original size: {get_model_size(model):.2f} MB")
print(f"Quantized size: {get_model_size(quantized_model):.2f} MB")

# 推理测试
x = torch.randint(0, 10000, (1, 32))
with torch.no_grad():
    out_orig = model(x)
    out_quant = quantized_model(x)

print(f"Output difference: {(out_orig - out_quant).abs().mean():.6f}")
```

### 2. 静态量化 (Static Quantization)

```python
import torch
import torch.nn as nn
import torch.quantization

class ConvModel(nn.Module):
    """CNN 模型用于静态量化"""
    def __init__(self):
        super().__init__()
        # 量化需要 QuantStub 和 DeQuantStub
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.quant(x)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.dequant(x)
        return x

    def fuse_model(self):
        """融合 Conv-BN-ReLU"""
        torch.quantization.fuse_modules(
            self,
            [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']],
            inplace=True
        )

# 准备静态量化
model = ConvModel()
model.eval()

# 1. 融合层
model.fuse_model()

# 2. 配置量化
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 3. 准备量化
torch.quantization.prepare(model, inplace=True)

# 4. 校准（用代表性数据）
calibration_data = torch.randn(100, 1, 28, 28)
with torch.no_grad():
    for i in range(0, 100, 10):
        model(calibration_data[i:i+10])

# 5. 转换为量化模型
torch.quantization.convert(model, inplace=True)

print("Static quantization completed!")

# 测试
x = torch.randn(1, 1, 28, 28)
output = model(x)
print(f"Output shape: {output.shape}")
```

### 3. 量化感知训练 (QAT)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization

class QATModel(nn.Module):
    """量化感知训练模型"""
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self,
            [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']],
            inplace=True
        )

def train_qat():
    model = QATModel()

    # 1. 融合层
    model.train()
    model.fuse_model()

    # 2. 配置 QAT
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    # 3. 准备 QAT
    torch.quantization.prepare_qat(model, inplace=True)

    # 4. 正常训练（带 fake quantization）
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 模拟训练
    for epoch in range(5):
        model.train()
        # 假设有训练数据
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 5. 转换为量化模型
    model.eval()
    quantized_model = torch.quantization.convert(model)

    return quantized_model

quantized_model = train_qat()
print("QAT completed!")
```

### 4. INT8 量化（使用 PyTorch 2.0+）

```python
import torch
import torch.nn as nn

# PyTorch 2.0+ 提供更简单的量化 API
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleNet()
model.eval()

# 使用 torch.compile 进行优化（包含自动量化）
# compiled_model = torch.compile(model, mode="reduce-overhead")

# 手动 INT8 量化（使用 torch.ao.quantization）
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

example_inputs = (torch.randn(1, 784),)

# 准备量化配置
qconfig_mapping = get_default_qconfig_mapping("fbgemm")

# 准备模型
model_prepared = prepare_fx(model, qconfig_mapping, example_inputs)

# 校准
with torch.no_grad():
    for _ in range(100):
        model_prepared(torch.randn(32, 784))

# 转换
model_quantized = convert_fx(model_prepared)

print("FX Graph Mode Quantization completed!")
```

### 5. 量化精度评估

```python
import torch
import torch.nn as nn
import numpy as np

def evaluate_quantization(original_model, quantized_model, test_data):
    """评估量化前后的精度差异"""
    original_model.eval()

    original_outputs = []
    quantized_outputs = []

    with torch.no_grad():
        for x in test_data:
            orig_out = original_model(x)
            quant_out = quantized_model(x)

            original_outputs.append(orig_out.numpy())
            quantized_outputs.append(quant_out.numpy())

    original_outputs = np.concatenate(original_outputs)
    quantized_outputs = np.concatenate(quantized_outputs)

    # 计算各种误差指标
    mae = np.abs(original_outputs - quantized_outputs).mean()
    mse = ((original_outputs - quantized_outputs) ** 2).mean()

    # 分类准确率对比（如果是分类模型）
    orig_preds = original_outputs.argmax(axis=1)
    quant_preds = quantized_outputs.argmax(axis=1)
    agreement = (orig_preds == quant_preds).mean()

    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Prediction Agreement: {agreement * 100:.2f}%")

    return {
        'mae': mae,
        'mse': mse,
        'agreement': agreement
    }

# 使用示例
# test_data = [torch.randn(32, 784) for _ in range(10)]
# metrics = evaluate_quantization(model, quantized_model, test_data)
```

## 量化方法对比

| 方法 | 精度损失 | 实现难度 | 适用场景 |
|------|----------|----------|----------|
| 动态量化 | 低 | ⭐ 简单 | LSTM, Transformer |
| 静态量化 | 中 | ⭐⭐ 中等 | CNN |
| QAT | 最低 | ⭐⭐⭐ 复杂 | 精度敏感场景 |

## 量化效果参考

| 模型 | 原始大小 | 量化后 | 加速比 |
|------|----------|--------|--------|
| ResNet-50 | 98 MB | 25 MB | 2-4x |
| BERT-base | 438 MB | 110 MB | 2-3x |
| GPT-2 | 500 MB | 125 MB | 2-3x |

## 下一步 Next

[下一章：推理优化技术 →](./03-inference-optimization.md)
