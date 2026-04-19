# GPU 训练与多卡并行 (GPU Training and Distributed Training)

## 概述 Overview

GPU 训练是深度学习的标配。本章介绍单卡 GPU 训练、多卡数据并行（DDP）和模型并行（FSDP）。

**难度级别**：⭐ 重点章节

## 代码实现 Implementation

### 1. 单 GPU 训练

```python
import torch
import torch.nn as nn

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模型移动到 GPU
model = nn.Linear(10, 5).to(device)

# 数据移动到 GPU
x = torch.randn(32, 10).to(device)

# 训练
output = model(x)

# DataLoader 配合 GPU
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))
loader = DataLoader(dataset, batch_size=32, pin_memory=True)  # pin_memory 加速传输

for data, target in loader:
    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
    output = model(data)
```

### 2. DataParallel (DP)

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)

# 简单的数据并行（不推荐用于多卡训练）
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to('cuda')

# 数据会自动分发到各 GPU
x = torch.randn(64, 10).to('cuda')
output = model(x)

# 访问原始模型
original_model = model.module if isinstance(model, nn.DataParallel) else model
```

### 3. DistributedDataParallel (DDP) - 推荐

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, epochs=10):
    setup(rank, world_size)

    # 创建模型
    model = torch.nn.Linear(10, 5).to(rank)
    model = DDP(model, device_ids=[rank])

    # 数据
    dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 10),
        torch.randint(0, 5, (1000,))
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 重要！确保每个 epoch 数据打乱不同

        for data, target in loader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 保存模型（只在主进程）
    if rank == 0:
        torch.save(model.module.state_dict(), 'model.pth')

    cleanup()

# 启动
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

# 或使用 torchrun 启动：
# torchrun --nproc_per_node=4 train.py
```

### 4. FSDP (Fully Sharded Data Parallel)

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

def train_fsdp(rank, world_size):
    setup(rank, world_size)

    # 自动包装策略：参数超过 100M 的层会被单独分片
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100_000_000
    )

    model = YourLargeModel()
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=rank
    )

    # 训练循环...

    # 保存模型
    if rank == 0:
        # FSDP 需要特殊的保存方式
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state_dict = model.state_dict()
            torch.save(state_dict, 'model.pth')

    cleanup()
```

### 5. 显存优化

```python
import torch
import gc

# 清理缓存
torch.cuda.empty_cache()
gc.collect()

# 监控显存
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 使用 inference_mode（比 no_grad 更高效）
with torch.inference_mode():
    output = model(x)

# 梯度检查点
from torch.utils.checkpoint import checkpoint
x = checkpoint(layer, x, use_reentrant=False)

# 混合精度
from torch.cuda.amp import autocast
with autocast():
    output = model(x)
```

## 小结 Summary

| 方法 | 适用场景 | 扩展性 |
|------|----------|--------|
| 单 GPU | 小模型、快速实验 | 单卡 |
| DataParallel | 简单多卡（不推荐） | 单机多卡 |
| DDP | 标准多卡训练 | 多机多卡 |
| FSDP | 超大模型训练 | 模型分片 |

## 下一步 Next

[下一章：调试与性能分析 →](./06-debugging-profiling.md)
