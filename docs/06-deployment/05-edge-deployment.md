# 边缘设备部署 (Edge Deployment)

## 概述 Overview

边缘部署将模型运行在资源受限的设备上，如手机、嵌入式系统、IoT 设备。本章介绍移动端和嵌入式部署技术。

## 代码实现 Implementation

### 1. PyTorch Mobile 导出

```python
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

class MobileNet(nn.Module):
    """轻量级移动端模型"""
    def __init__(self, num_classes=10):
        super().__init__()

        # Depthwise Separable Convolution
        self.features = nn.Sequential(
            # 标准卷积
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            # Depthwise + Pointwise
            self._make_dw_block(32, 64, 1),
            self._make_dw_block(64, 128, 2),
            self._make_dw_block(128, 128, 1),
            self._make_dw_block(128, 256, 2),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(256, num_classes)

    def _make_dw_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 创建和导出模型
model = MobileNet()
model.eval()

# 转换为 TorchScript
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 移动端优化
optimized_model = optimize_for_mobile(traced_model)

# 保存为移动端格式
optimized_model._save_for_lite_interpreter("model_mobile.ptl")

print("Mobile model exported!")
```

### 2. 模型轻量化

```python
import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    """SE 注意力模块（轻量化）"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class EfficientBlock(nn.Module):
    """高效卷积块"""
    def __init__(self, in_ch, out_ch, expand_ratio=4, stride=1):
        super().__init__()
        hidden_ch = in_ch * expand_ratio

        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []

        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True)
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride, 1,
                      groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True)
        ])

        # SE
        layers.append(SqueezeExcitation(hidden_ch))

        # Project
        layers.extend([
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

# 构建轻量模型
class TinyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        self.blocks = nn.Sequential(
            EfficientBlock(16, 24, 1, 1),
            EfficientBlock(24, 24, 4, 2),
            EfficientBlock(24, 40, 4, 2),
            EfficientBlock(40, 80, 4, 2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(80, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

model = TinyNet()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 3. 知识蒸馏

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """知识蒸馏损失"""

    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Hard loss (与真实标签)
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft loss (与教师输出)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)

        # 组合
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss

def train_with_distillation(teacher, student, train_loader, epochs=10):
    """使用知识蒸馏训练学生模型"""
    teacher.eval()
    student.train()

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    criterion = DistillationLoss(temperature=4.0, alpha=0.7)

    for epoch in range(epochs):
        total_loss = 0

        for x, y in train_loader:
            optimizer.zero_grad()

            # 教师推理
            with torch.no_grad():
                teacher_logits = teacher(x)

            # 学生推理
            student_logits = student(x)

            # 蒸馏损失
            loss = criterion(student_logits, teacher_logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    return student

# 使用示例
# teacher = LargeModel()  # 大模型
# student = TinyNet()     # 小模型
# student = train_with_distillation(teacher, student, train_loader)
```

### 4. ONNX Runtime Mobile

```python
import torch
import torch.nn as nn
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# 导出模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = SimpleModel()
model.eval()

# 导出 ONNX
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=13
)

# 量化为 INT8（减小模型大小）
quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QUInt8
)

print("ONNX model exported and quantized!")

# 验证量化模型
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model_quantized.onnx")
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
output = session.run(None, {'input': input_data})
print(f"Output shape: {output[0].shape}")
```

### 5. TensorFlow Lite 转换

```python
import torch
import torch.nn as nn

# 首先导出为 ONNX，然后转换为 TFLite
# pip install onnx-tf tensorflow

def convert_to_tflite(model, example_input, output_path):
    """PyTorch -> ONNX -> TensorFlow -> TFLite"""

    # 1. PyTorch -> ONNX
    torch.onnx.export(
        model, example_input, "temp.onnx",
        input_names=['input'],
        output_names=['output'],
        opset_version=13
    )

    # 2. ONNX -> TensorFlow
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load("temp.onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("temp_tf")

    # 3. TensorFlow -> TFLite
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model("temp_tf")

    # 优化选项
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {output_path}")

    # 清理临时文件
    import os
    os.remove("temp.onnx")
    import shutil
    shutil.rmtree("temp_tf")

# 使用示例
# model = TinyNet()
# model.eval()
# convert_to_tflite(model, torch.randn(1, 3, 224, 224), "model.tflite")
```

### 6. 性能测试工具

```python
import torch
import time
import numpy as np

def benchmark_model(model, input_shape, num_runs=100, warmup=10, device='cpu'):
    """基准测试模型性能"""
    model = model.to(device)
    model.eval()

    input_data = torch.randn(input_shape).to(device)

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_data)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    latencies = []

    with torch.inference_mode():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(input_data)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    results = {
        'mean_ms': latencies.mean(),
        'std_ms': latencies.std(),
        'min_ms': latencies.min(),
        'max_ms': latencies.max(),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
    }

    print(f"=== Benchmark Results ({device}) ===")
    print(f"Mean: {results['mean_ms']:.3f} ms")
    print(f"Std:  {results['std_ms']:.3f} ms")
    print(f"P50:  {results['p50_ms']:.3f} ms")
    print(f"P95:  {results['p95_ms']:.3f} ms")
    print(f"P99:  {results['p99_ms']:.3f} ms")

    return results

def profile_model(model, input_shape):
    """分析模型计算量和参数量"""
    from torch.profiler import profile, ProfilerActivity

    model.eval()
    input_data = torch.randn(input_shape)

    # 参数量
    params = sum(p.numel() for p in model.parameters())

    # FLOPs 估算（使用 profiler）
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True
    ) as prof:
        with torch.inference_mode():
            model(input_data)

    # 获取 FLOPs
    events = prof.key_averages()
    total_flops = sum(e.flops for e in events if e.flops > 0)

    print(f"=== Model Profile ===")
    print(f"Parameters: {params:,}")
    print(f"FLOPs: {total_flops:,}")
    print(f"Size (MB): {params * 4 / 1024 / 1024:.2f}")

    return {
        'params': params,
        'flops': total_flops,
        'size_mb': params * 4 / 1024 / 1024
    }

# 使用示例
model = TinyNet()
benchmark_model(model, (1, 3, 224, 224), device='cpu')
profile_model(model, (1, 3, 224, 224))
```

## 边缘部署平台对比

| 平台 | 框架支持 | 性能 | 适用设备 |
|------|----------|------|----------|
| PyTorch Mobile | PyTorch | 良好 | iOS/Android |
| ONNX Runtime | 多框架 | 优秀 | 通用 |
| TensorFlow Lite | TF/ONNX | 优秀 | iOS/Android/嵌入式 |
| NCNN | 多框架 | 最佳 | Android/嵌入式 |
| Core ML | TF/ONNX | 优秀 | iOS/macOS |

## 下一步 Next

恭喜完成部署篇！接下来进入实战项目。

[下一章：手写数字识别 →](../07-practice/01-mnist-classifier.md)
