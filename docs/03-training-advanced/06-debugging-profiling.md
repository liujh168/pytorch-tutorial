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

## 深入理解：CPU 版 Profiler 完整示例

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# 完整可运行的 Profiler 示例（CPU/CUDA 通用）
model  = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 10))
data   = torch.randn(32, 256)
target = torch.randint(0, 10, (32,))
opt    = torch.optim.Adam(model.parameters())
crit   = nn.CrossEntropyLoss()

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,    # True 会显示调用栈，但更慢
) as prof:
    for step in range(5):
        with record_function("forward_pass"):
            output = model(data)

        with record_function("loss_compute"):
            loss = crit(output, target)

        with record_function("backward_pass"):
            opt.zero_grad()
            loss.backward()

        with record_function("optimizer_step"):
            opt.step()

        prof.step()

# 按 CPU 时间排序，显示前 10 项
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# 导出可在 Chrome 浏览器中查看的 trace（输入 chrome://tracing）
prof.export_chrome_trace("trace.json")
print("Chrome trace 已导出到 trace.json")
```

## 深入理解：训练中的常见问题速查

```python
import torch
import torch.nn as nn
import math

def diagnose_training_issues(model, dataloader, optimizer, criterion, device='cpu'):
    """训练过程中的问题诊断助手"""
    model.train()

    for step, (X, y) in enumerate(dataloader):
        if step >= 3: break
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()

        # ① 检测 Loss 是否 NaN/Inf
        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print(f"⚠️ Step {step}: loss={loss.item()} — 检查输入数据或降低学习率")

        # ② 检测梯度异常
        bad_grads = []
        for name, param in model.named_parameters():
            if param.grad is None: continue
            if torch.isnan(param.grad).any():
                bad_grads.append(f"{name}(NaN)")
            elif param.grad.norm() > 100:
                bad_grads.append(f"{name}(norm={param.grad.norm():.1f})")
        if bad_grads:
            print(f"⚠️ 梯度异常: {', '.join(bad_grads)}")
        else:
            total_norm = sum(p.grad.norm()**2 for p in model.parameters()
                            if p.grad is not None) ** 0.5
            print(f"✓ Step {step}: loss={loss.item():.4f}, grad_norm={total_norm:.3f}")

        # ③ 检测权重是否更新
        weights_before = {n: p.clone() for n, p in model.named_parameters()}
        optimizer.step()
        frozen_params = [n for n, p in model.named_parameters()
                        if torch.allclose(p, weights_before[n])]
        if frozen_params:
            print(f"⚠️ 以下参数未更新（可能被冻结）: {frozen_params}")

# 常见问题与解决方案（速查表）:
issues = {
    "Loss 不降":    ["学习率太小/太大", "数据标签错误", "模型容量不足"],
    "Loss NaN":     ["学习率过大", "输入含 NaN", "log(0) 数值不稳定"],
    "过拟合":       ["数据量不足", "模型太大", "需要正则化"],
    "欠拟合":       ["模型太小", "学习率太小", "训练轮数不足"],
    "GPU 显存不足": ["减小 batch_size", "梯度累积", "混合精度", "梯度检查点"],
    "训练速度慢":   ["num_workers 不足", "数据预处理未缓存", "模型有 CPU-GPU 数据拷贝"],
}
print("\n常见训练问题速查:")
for prob, solutions in issues.items():
    print(f"  {prob}: {', '.join(solutions)}")
```

## 小结 Summary

| 工具 | 用途 | 何时使用 |
|------|------|----------|
| `torch.autograd.set_detect_anomaly` | 检测梯度异常 | 出现 NaN 时 |
| `torch.profiler` | 性能分析 | 训练太慢时 |
| `torch.cuda.memory_*` | 显存监控 | 显存 OOM 时 |
| `record_function` | 自定义 profiler 标签 | 精确定位瓶颈 |
| `debug_tensor()` | 张量状态检查 | 输出异常时 |

## 下一步 Next

恭喜完成训练进阶篇！接下来学习 Attention 和 Transformer。

[下一章：注意力机制直觉理解 →](../04-attention-transformer/01-attention-intuition.md)
