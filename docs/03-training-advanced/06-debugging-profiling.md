# 调试与性能分析 (Debugging and Profiling)

## 概述 Overview

调试深度学习代码和分析性能瓶颈是提高开发效率的关键技能。

## 代码实现 Implementation

### 1. 常见调试技巧

```python
import torch

# === 检查张量状态 ===
def debug_tensor(tensor, name="tensor"):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min/Max: {tensor.min():.4f}/{tensor.max():.4f}")
    print(f"  Mean/Std: {tensor.mean():.4f}/{tensor.std():.4f}")
    print(f"  Has NaN: {torch.isnan(tensor).any()}")
    print(f"  Has Inf: {torch.isinf(tensor).any()}")

# === 检测异常 ===
torch.autograd.set_detect_anomaly(True)  # 检测梯度异常

try:
    loss.backward()
except RuntimeError as e:
    print(f"Backward error: {e}")

# === 打印模型结构 ===
from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))

# === 检查梯度流 ===
def check_grad_flow(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm={param.grad.norm():.4f}")
        else:
            print(f"{name}: no gradient")
```

### 2. PyTorch Profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# === 基本性能分析 ===
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    for i, (data, target) in enumerate(train_loader):
        if i >= 10:
            break
        with record_function("forward"):
            output = model(data.cuda())
        with record_function("backward"):
            loss = output.sum()
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()

# 打印统计
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 导出 Chrome Trace
prof.export_chrome_trace("trace.json")

# === 使用 schedule ===
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    with_stack=True
) as prof:
    for step, (data, target) in enumerate(train_loader):
        if step >= 20:
            break
        output = model(data.cuda())
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()  # 告诉 profiler 一个 step 结束
```

### 3. 内存分析

```python
import torch

# === 显存统计 ===
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 重置统计
torch.cuda.reset_peak_memory_stats()

# === 内存快照 ===
torch.cuda.memory._record_memory_history()
# ... 运行代码 ...
snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._dump_snapshot(snapshot, "memory_snapshot.pickle")

# === 查找内存泄漏 ===
def find_memory_leaks():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            print(f"Tensor: {obj.shape}, {obj.device}")
```

### 4. 常见问题诊断

```python
# === 损失为 NaN ===
# 原因：学习率太大、数值不稳定
# 解决：降低学习率、使用梯度裁剪、检查输入数据

# === 显存不足 ===
# 解决：减小 batch_size、使用梯度累积、混合精度、梯度检查点

# === 训练速度慢 ===
# 检查：数据加载是否是瓶颈
import time

start = time.time()
for i, batch in enumerate(train_loader):
    if i >= 100:
        break
data_time = time.time() - start

start = time.time()
for i, batch in enumerate(train_loader):
    if i >= 100:
        break
    output = model(batch[0].cuda())
total_time = time.time() - start

print(f"Data loading: {data_time:.2f}s")
print(f"Total: {total_time:.2f}s")
print(f"Data loading ratio: {data_time/total_time:.1%}")
```

## 小结 Summary

| 工具 | 用途 |
|------|------|
| `torch.autograd.set_detect_anomaly` | 检测梯度异常 |
| `torch.profiler` | 性能分析 |
| `torch.cuda.memory_*` | 显存监控 |
| `torchinfo.summary` | 模型结构 |

## 下一步 Next

恭喜完成训练进阶篇！接下来学习 Attention 和 Transformer。

[下一章：注意力机制直觉理解 →](../04-attention-transformer/01-attention-intuition.md)
