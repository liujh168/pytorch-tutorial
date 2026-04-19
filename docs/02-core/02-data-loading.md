# 数据加载与预处理 (Data Loading and Preprocessing)

## 概述 Overview

高效的数据加载和预处理是深度学习训练的关键。PyTorch 提供了 `torch.utils.data` 模块来处理数据管道。本章将全面介绍数据加载的最佳实践。

完成本章后，你将：

- 理解 Dataset 和 DataLoader 的工作原理
- 掌握数据预处理和增强技术
- 学会处理各种数据格式
- 了解多进程数据加载

**难度级别**：🟡 进阶级

## 前置知识 Prerequisites

- [01-nn-module](./01-nn-module.md) - nn.Module 深入理解
- Python 迭代器和生成器基础

## 核心概念 Core Concepts

### 数据加载流程

```
原始数据 → Dataset → DataLoader → 训练循环
             ↓           ↓
         预处理/变换    批处理/打乱/并行
```

### 核心组件

```python
from torch.utils.data import Dataset, DataLoader

# Dataset: 定义如何获取单个样本
# DataLoader: 批处理、打乱、并行加载
```

## 代码实现 Implementation

### 1. 自定义 Dataset

```python
import torch
from torch.utils.data import Dataset

# === Map-style Dataset（最常用）===
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: 特征数据
            labels: 标签
            transform: 可选的变换函数
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """根据索引获取单个样本"""
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# 使用示例
data = torch.randn(1000, 10)
labels = torch.randint(0, 5, (1000,))
dataset = CustomDataset(data, labels)

print(f"Dataset size: {len(dataset)}")
sample, label = dataset[0]
print(f"Sample shape: {sample.shape}, Label: {label}")

# === 从文件加载的 Dataset ===
import os

class FileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        # 加载文件（示例：假设是文本文件）
        with open(file_path, 'r') as f:
            data = f.read()

        if self.transform:
            data = self.transform(data)

        return data

# === Iterable-style Dataset（流式数据）===
from torch.utils.data import IterableDataset

class StreamDataset(IterableDataset):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # 返回数据迭代器
        for item in self.data_source:
            yield item

# 用于大文件或网络流
class LargeFileDataset(IterableDataset):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, 'r') as f:
            for line in f:
                yield self.process_line(line)

    def process_line(self, line):
        return line.strip()
```

### 2. DataLoader 详解

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# 创建示例数据集
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)

# === 基本 DataLoader ===
loader = DataLoader(
    dataset,
    batch_size=32,      # 批大小
    shuffle=True,       # 是否打乱
    num_workers=0,      # 工作进程数
    drop_last=False     # 是否丢弃不完整的最后一批
)

for batch_X, batch_y in loader:
    print(f"Batch X shape: {batch_X.shape}")  # (32, 10) 或最后一批可能更小
    print(f"Batch y shape: {batch_y.shape}")
    break

# === 重要参数详解 ===
loader = DataLoader(
    dataset,
    batch_size=32,

    # 打乱数据
    shuffle=True,            # 训练时通常为 True

    # 多进程加载
    num_workers=4,           # 0 = 主进程加载
    prefetch_factor=2,       # 每个 worker 预取的 batch 数

    # 内存管理
    pin_memory=True,         # GPU 训练时设为 True，加速传输
    persistent_workers=True, # 保持 worker 存活

    # 批处理
    drop_last=True,          # 丢弃不完整批次，训练时有用
    collate_fn=None,         # 自定义批处理函数

    # 采样器
    sampler=None,            # 自定义采样方式
)

# === 使用 pin_memory 的完整示例 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # 数据会被复制到 CUDA 固定内存
)

for batch_X, batch_y in loader:
    # non_blocking=True 可以实现异步传输
    batch_X = batch_X.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)
    # 训练步骤...
```

### 3. 数据变换 (Transforms)

```python
import torch
from torchvision import transforms

# === 图像变换示例 ===
# 训练时的变换（包含数据增强）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 验证/测试时的变换（无数据增强）
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# === 自定义变换 ===
class AddGaussianNoise:
    """添加高斯噪声"""
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class Normalize:
    """自定义归一化"""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

# 组合变换
custom_transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1),
    Normalize([0.5], [0.5])
])

# === Lambda 变换 ===
transform = transforms.Lambda(lambda x: x / 255.0)

# === 文本变换示例 ===
class TextTransform:
    def __init__(self, vocab, max_len):
        self.vocab = vocab
        self.max_len = max_len

    def __call__(self, text):
        # 分词
        tokens = text.lower().split()
        # 转索引
        indices = [self.vocab.get(t, 0) for t in tokens]  # 0 = unknown
        # 截断或填充
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [0] * (self.max_len - len(indices))
        return torch.tensor(indices)
```

### 4. 自定义 collate_fn

```python
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# === 处理变长序列 ===
class VariableLengthDataset(Dataset):
    def __init__(self):
        # 模拟变长序列数据
        self.data = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6, 7, 8, 9]),
            torch.tensor([10])
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_variable_length(batch):
    """自定义 collate 函数处理变长序列"""
    # 记录原始长度
    lengths = torch.tensor([len(seq) for seq in batch])

    # 填充到相同长度
    padded = pad_sequence(batch, batch_first=True, padding_value=0)

    return padded, lengths

dataset = VariableLengthDataset()
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_variable_length)

for padded, lengths in loader:
    print(f"Padded: {padded}")
    print(f"Lengths: {lengths}")

# === 处理复杂数据结构 ===
class ComplexDataset(Dataset):
    def __init__(self):
        self.data = [
            {'image': torch.randn(3, 32, 32), 'label': 0, 'name': 'img1'},
            {'image': torch.randn(3, 32, 32), 'label': 1, 'name': 'img2'},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_dict(batch):
    """处理字典格式的批处理"""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    names = [item['name'] for item in batch]

    return {
        'images': images,
        'labels': labels,
        'names': names
    }

dataset = ComplexDataset()
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_dict)

for batch in loader:
    print(f"Images shape: {batch['images'].shape}")
    print(f"Labels: {batch['labels']}")
    print(f"Names: {batch['names']}")
```

### 5. 采样器 (Samplers)

```python
import torch
from torch.utils.data import DataLoader, TensorDataset, Sampler
from torch.utils.data.sampler import (
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler
)

# 创建不平衡数据集
X = torch.randn(100, 10)
y = torch.cat([torch.zeros(90), torch.ones(10)])  # 90% 类0，10% 类1
dataset = TensorDataset(X, y)

# === 加权采样（解决类别不平衡）===
# 计算每个样本的权重
class_counts = torch.bincount(y.long())
class_weights = 1.0 / class_counts
sample_weights = class_weights[y.long()]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

loader = DataLoader(dataset, batch_size=10, sampler=sampler)

# 验证采样效果
all_labels = []
for _, labels in loader:
    all_labels.extend(labels.tolist())
print(f"Class distribution: {torch.bincount(torch.tensor(all_labels).long())}")

# === 子集采样 ===
indices = list(range(50))  # 只使用前50个样本
sampler = SubsetRandomSampler(indices)
loader = DataLoader(dataset, batch_size=10, sampler=sampler)

# === 自定义采样器 ===
class BalancedBatchSampler(Sampler):
    """每个 batch 包含相等数量的正负样本"""
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size

        # 按类别分组索引
        self.pos_indices = (labels == 1).nonzero().squeeze()
        self.neg_indices = (labels == 0).nonzero().squeeze()

    def __iter__(self):
        pos_perm = torch.randperm(len(self.pos_indices))
        neg_perm = torch.randperm(len(self.neg_indices))

        half_batch = self.batch_size // 2
        n_batches = min(len(self.pos_indices), len(self.neg_indices)) // half_batch

        for i in range(n_batches):
            pos_batch = self.pos_indices[pos_perm[i*half_batch:(i+1)*half_batch]]
            neg_batch = self.neg_indices[neg_perm[i*half_batch:(i+1)*half_batch]]
            yield torch.cat([pos_batch, neg_batch]).tolist()

    def __len__(self):
        half_batch = self.batch_size // 2
        return min(len(self.pos_indices), len(self.neg_indices)) // half_batch

# 使用自定义采样器
sampler = BalancedBatchSampler(y, batch_size=10)
loader = DataLoader(dataset, batch_sampler=sampler)
```

### 6. 分布式数据加载

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# === 分布式训练的数据加载 ===
def setup_distributed_dataloader(dataset, rank, world_size, batch_size):
    """设置分布式数据加载器"""

    # DistributedSampler 确保每个进程获得不同的数据子集
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,  # 总进程数
        rank=rank,                 # 当前进程 ID
        shuffle=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    return loader, sampler

# 使用示例（在分布式训练中）
# rank = torch.distributed.get_rank()
# world_size = torch.distributed.get_world_size()
# loader, sampler = setup_distributed_dataloader(dataset, rank, world_size, 32)
#
# for epoch in range(num_epochs):
#     sampler.set_epoch(epoch)  # 确保每个 epoch 的打乱不同
#     for batch in loader:
#         ...
```

### 7. 常见数据格式处理

```python
import torch
from torch.utils.data import Dataset
import json
import csv

# === JSON 数据 ===
class JSONDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item['features']), item['label']

# === CSV 数据 ===
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        features = torch.tensor([float(row[f'feature_{i}']) for i in range(10)])
        label = int(row['label'])
        return features, label

# === 图像文件夹 ===
# torchvision 提供了方便的 ImageFolder
from torchvision.datasets import ImageFolder
from torchvision import transforms

# 目录结构应该是:
# root/
#   class1/
#     img1.jpg
#     img2.jpg
#   class2/
#     img3.jpg

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# dataset = ImageFolder('path/to/images', transform=transform)

# === 内存映射大文件 ===
import numpy as np

class MemmapDataset(Dataset):
    def __init__(self, file_path, shape, dtype='float32'):
        # 内存映射，不会全部加载到内存
        self.data = np.memmap(file_path, dtype=dtype, mode='r', shape=shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].copy())
```

## 深入理解 Deep Dive

### DataLoader 的工作原理

```python
# DataLoader 内部流程：
#
# 1. 采样器生成索引
#    sampler → [0, 5, 2, 8, 3, 7, ...]
#
# 2. 批采样器将索引分组
#    batch_sampler → [[0, 5, 2, 8], [3, 7, ...], ...]
#
# 3. 工作进程获取数据
#    worker 0: dataset[0], dataset[5]
#    worker 1: dataset[2], dataset[8]
#
# 4. collate_fn 组合成批
#    collate_fn([sample0, sample5, sample2, sample8]) → batch

# === 多进程工作原理 ===
# - 每个 worker 是独立的进程
# - 数据通过队列传递
# - prefetch_factor 控制预取数量

# 注意事项：
# 1. worker 中的随机种子需要设置
def worker_init_fn(worker_id):
    torch.manual_seed(torch.initial_seed() + worker_id)

loader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)

# 2. 某些对象不能被 pickle（如 lambda）
# 3. Windows 上需要在 if __name__ == '__main__' 中使用
```

### 性能优化

```python
import torch
from torch.utils.data import DataLoader

# === 数据加载瓶颈诊断 ===
import time

def benchmark_dataloader(loader, num_batches=100):
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
    elapsed = time.time() - start
    print(f"Time for {num_batches} batches: {elapsed:.2f}s")
    print(f"Batches per second: {num_batches/elapsed:.2f}")

# 测试不同配置
for num_workers in [0, 2, 4, 8]:
    loader = DataLoader(dataset, batch_size=32, num_workers=num_workers)
    print(f"\nnum_workers={num_workers}")
    benchmark_dataloader(loader)

# === 优化建议 ===
# 1. 使用 SSD 而非 HDD
# 2. 预处理数据保存为 tensor 格式
# 3. 使用合适的 num_workers（通常 = CPU 核数）
# 4. 使用 pin_memory=True（GPU 训练）
# 5. 使用 persistent_workers=True（减少启动开销）
# 6. 预计算并缓存变换结果
```

## 常见问题 FAQ

### Q1: num_workers > 0 时程序卡住

```python
# Windows 上必须在 main 中使用
if __name__ == '__main__':
    loader = DataLoader(dataset, num_workers=4)
    for batch in loader:
        pass

# 或者设置 num_workers=0
```

### Q2: CUDA out of memory with DataLoader

```python
# 问题：数据传输过快导致 GPU 内存堆积
# 解决：减少 prefetch_factor
loader = DataLoader(dataset, num_workers=4, prefetch_factor=1)
```

### Q3: 数据随机性不一致

```python
# 设置随机种子
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g
)
```

## 小结 Summary

本章要点：

1. **Dataset 定义**
   ```python
   class MyDataset(Dataset):
       def __len__(self): ...
       def __getitem__(self, idx): ...
   ```

2. **DataLoader 配置**
   ```python
   loader = DataLoader(
       dataset,
       batch_size=32,
       shuffle=True,
       num_workers=4,
       pin_memory=True
   )
   ```

3. **数据变换**
   ```python
   transform = transforms.Compose([...])
   ```

4. **处理特殊情况**
   - 变长序列：自定义 `collate_fn`
   - 类别不平衡：`WeightedRandomSampler`
   - 分布式训练：`DistributedSampler`

## 延伸阅读 Further Reading

- [PyTorch 数据加载教程](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [torchvision.transforms 文档](https://pytorch.org/vision/stable/transforms.html)

## 下一步 Next

掌握了数据加载后，下一章我们将学习训练循环的最佳实践。

[下一章：训练循环最佳实践 →](./03-training-loop.md)
