# 训练循环最佳实践 (Training Loop Best Practices)

## 概述 Overview

训练循环是深度学习的核心。一个良好设计的训练循环不仅能正确训练模型，还能方便监控、调试和复现实验。本章将介绍编写高质量训练代码的最佳实践。

完成本章后，你将：

- 掌握训练循环的标准结构
- 学会监控和记录训练过程
- 了解模型保存和恢复策略
- 能够实现早停和学习率调度

**难度级别**：🟡 进阶级

## 前置知识 Prerequisites

- [01-nn-module](./01-nn-module.md) - nn.Module 深入理解
- [02-data-loading](./02-data-loading.md) - 数据加载与预处理

## 核心概念 Core Concepts

### 训练循环的基本结构

```python
for epoch in range(num_epochs):
    # 1. 训练阶段
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 2. 验证阶段
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            ...

    # 3. 日志和检查点
    log_metrics()
    save_checkpoint()
```

## 代码实现 Implementation

### 1. 完整的训练模板

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device='cuda',
        scheduler=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # 反向传播
            loss.backward()

            # 梯度裁剪（可选）
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def fit(self, num_epochs, early_stopping_patience=None):
        """完整训练流程"""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_loss, val_accuracy = self.validate()

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)

            # 打印信息
            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}s)')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2%}')
            print(f'  Learning Rate: {lr:.6f}')

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        return self.history

    def save_checkpoint(self, path):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)

# === 使用示例 ===
# model = YourModel()
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
#
# trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler=scheduler)
# history = trainer.fit(num_epochs=100, early_stopping_patience=10)
```

### 2. 使用 TensorBoard 监控

```python
import torch
from torch.utils.tensorboard import SummaryWriter

class TensorBoardTrainer:
    def __init__(self, model, log_dir='runs/experiment'):
        self.model = model
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def log_scalars(self, train_loss, val_loss, val_acc, epoch):
        """记录标量指标"""
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)

    def log_histograms(self, epoch):
        """记录参数分布"""
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param, epoch)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    def log_learning_rate(self, lr, epoch):
        """记录学习率"""
        self.writer.add_scalar('Learning_Rate', lr, epoch)

    def log_model_graph(self, sample_input):
        """记录模型结构"""
        self.writer.add_graph(self.model, sample_input)

    def log_images(self, images, tag, epoch):
        """记录图像"""
        self.writer.add_images(tag, images, epoch)

    def log_text(self, text, tag, epoch):
        """记录文本"""
        self.writer.add_text(tag, text, epoch)

    def close(self):
        self.writer.close()

# 使用
# tensorboard --logdir=runs
# 然后访问 http://localhost:6006
```

### 3. 梯度累积

```python
import torch
import torch.nn as nn

def train_with_gradient_accumulation(
    model,
    train_loader,
    criterion,
    optimizer,
    accumulation_steps=4,
    device='cuda'
):
    """使用梯度累积训练，模拟更大的 batch size"""
    model.train()
    optimizer.zero_grad()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 损失归一化
        loss = loss / accumulation_steps

        # 反向传播（梯度累积）
        loss.backward()

        # 每 accumulation_steps 步更新一次参数
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # 处理最后不完整的累积
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

# 等效于 batch_size = 32 * 4 = 128
# train_loader = DataLoader(dataset, batch_size=32)
# train_with_gradient_accumulation(model, train_loader, criterion, optimizer, accumulation_steps=4)
```

### 4. 混合精度训练

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(
    model,
    train_loader,
    criterion,
    optimizer,
    device='cuda'
):
    """使用混合精度训练（float16 + float32）"""
    model.train()
    scaler = GradScaler()  # 梯度缩放器

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # 使用 autocast 自动选择精度
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # 缩放损失并反向传播
        scaler.scale(loss).backward()

        # 取消缩放并更新参数
        scaler.step(optimizer)

        # 更新缩放因子
        scaler.update()

# === 更完整的混合精度训练类 ===
class AMPTrainer:
    def __init__(self, model, optimizer, criterion, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = GradScaler()

    def train_step(self, data, target):
        data, target = data.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()

        with autocast():
            output = self.model(data)
            loss = self.criterion(output, target)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    @torch.no_grad()
    def eval_step(self, data, target):
        data, target = data.to(self.device), target.to(self.device)

        with autocast():
            output = self.model(data)
            loss = self.criterion(output, target)

        return output, loss.item()
```

### 5. 分布式训练基础

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """初始化分布式环境"""
    dist.init_process_group(
        backend='nccl',  # GPU 用 nccl，CPU 用 gloo
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def train_distributed(rank, world_size, model, dataset, epochs):
    """分布式训练主函数"""
    setup(rank, world_size)

    # 将模型移到对应 GPU 并包装
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # 分布式采样器
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 重要！确保每个 epoch 打乱不同

        for data, target in loader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # 只在主进程打印和保存
        if rank == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
            torch.save(model.module.state_dict(), 'model.pth')

    cleanup()

# 启动分布式训练
# if __name__ == '__main__':
#     world_size = torch.cuda.device_count()
#     mp.spawn(train_distributed, args=(world_size, model, dataset, epochs), nprocs=world_size)
```

### 6. 可复现性设置

```python
import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU

    # 确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在训练开始时调用
set_seed(42)

# === DataLoader 的可复现性 ===
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g
)
```

### 7. 训练异常处理

```python
import torch
import math

class RobustTrainer:
    def __init__(self, model, optimizer, criterion, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_grad_norm = max_grad_norm

    def train_step(self, data, target):
        self.optimizer.zero_grad()

        output = self.model(data)
        loss = self.criterion(output, target)

        # 检查损失是否为 NaN
        if math.isnan(loss.item()):
            print("Warning: NaN loss detected!")
            return None

        loss.backward()

        # 检查梯度是否包含 NaN
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    print("Warning: NaN gradient detected!")
                    return None
                total_norm += p.grad.data.norm(2).item() ** 2

        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm:.4f}")

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )

        self.optimizer.step()

        return loss.item()

# === 梯度异常检测 ===
def check_gradients(model):
    """检查梯度是否正常"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if math.isnan(grad_norm):
                print(f"NaN gradient in {name}")
                return False
            if math.isinf(grad_norm):
                print(f"Inf gradient in {name}")
                return False
            if grad_norm > 1000:
                print(f"Large gradient in {name}: {grad_norm}")
    return True

# === 异常恢复 ===
class CheckpointManager:
    def __init__(self, model, optimizer, save_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, epoch, loss, filename=None):
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'

        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, path)

    def load_latest(self):
        """加载最新的检查点"""
        checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda x: os.path.getmtime(
            os.path.join(self.save_dir, x)
        ))
        return self.load(os.path.join(self.save_dir, latest))

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
```

## 深入理解 Deep Dive

### 训练循环性能分析

```python
import torch
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name}: {elapsed*1000:.2f}ms")

# 分析训练循环各部分耗时
def profile_training_step(model, data, target, criterion, optimizer):
    with timer("Total"):
        with timer("  Forward"):
            output = model(data)

        with timer("  Loss"):
            loss = criterion(output, target)

        with timer("  Backward"):
            optimizer.zero_grad()
            loss.backward()

        with timer("  Optimizer"):
            optimizer.step()

# === 使用 PyTorch Profiler ===
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    for i, (data, target) in enumerate(train_loader):
        if i >= 10:  # 只分析前10个batch
            break

        with record_function("forward"):
            output = model(data.cuda())

        with record_function("loss"):
            loss = criterion(output, target.cuda())

        with record_function("backward"):
            optimizer.zero_grad()
            loss.backward()

        with record_function("optimizer"):
            optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## 常见问题 FAQ

### Q1: 训练损失不下降

```python
# 检查清单：
# 1. 学习率是否合适
# 2. 数据是否正确加载
# 3. 标签是否正确对应
# 4. 损失函数是否匹配任务类型

# 诊断代码
def diagnose_training(model, data, target, criterion):
    model.train()
    output = model(data)

    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    print(f"Target range: [{target.min():.2f}, {target.max():.2f}]")

    loss = criterion(output, target)
    print(f"Loss: {loss.item():.4f}")
```

### Q2: 过拟合

```python
# 解决方案：
# 1. 数据增强
# 2. 正则化（weight_decay）
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 3. Dropout
# 4. 早停
# 5. 减少模型复杂度
```

### Q3: GPU 内存不足

```python
# 解决方案：
# 1. 减小 batch_size
# 2. 使用梯度累积
# 3. 使用混合精度训练
# 4. 使用 gradient checkpointing
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)  # 不保存中间激活
        x = checkpoint(self.layer2, x)
        return x
```

## 小结 Summary

本章要点：

1. **标准训练循环**
   ```python
   for epoch in range(epochs):
       model.train()
       for batch in train_loader:
           optimizer.zero_grad()
           loss = criterion(model(batch), target)
           loss.backward()
           optimizer.step()
   ```

2. **最佳实践**
   - 使用 `model.train()` 和 `model.eval()`
   - 验证时使用 `torch.no_grad()`
   - 保存检查点以便恢复
   - 使用早停防止过拟合

3. **性能优化**
   - 梯度累积：模拟大 batch
   - 混合精度：加速训练
   - 分布式训练：多 GPU

## 延伸阅读 Further Reading

- [PyTorch 训练技巧](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [分布式训练教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

## 下一步 Next

掌握训练循环后，下一章我们将深入学习各种损失函数及其应用场景。

[下一章：损失函数详解 →](./04-loss-functions.md)
