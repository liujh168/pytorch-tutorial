# 推理优化技术 (Inference Optimization)

## 概述 Overview

推理优化旨在减少延迟、提高吞吐量、降低资源消耗。本章介绍多种优化技术。

## 代码实现 Implementation

### 1. 基础优化

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = Model()
model.eval()

# === 优化1: 禁用梯度计算 ===
with torch.no_grad():
    output = model(torch.randn(1, 512))

# === 优化2: 使用推理模式 ===
with torch.inference_mode():
    output = model(torch.randn(1, 512))

# === 优化3: 半精度推理 (FP16) ===
model_fp16 = model.half()
input_fp16 = torch.randn(1, 512).half()

with torch.inference_mode():
    output_fp16 = model_fp16(input_fp16)

print(f"FP32 output: {output.dtype}")
print(f"FP16 output: {output_fp16.dtype}")
```

### 2. Torch Compile (PyTorch 2.0+)

```python
import torch
import torch.nn as nn
import time

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attention(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

model = TransformerBlock(512, 8)
model.eval()

# 使用 torch.compile 编译优化
compiled_model = torch.compile(model, mode="reduce-overhead")

# 预热
x = torch.randn(32, 128, 512)
with torch.inference_mode():
    for _ in range(10):
        compiled_model(x)

# 基准测试
def benchmark(model, input_data, num_runs=100):
    start = time.time()
    with torch.inference_mode():
        for _ in range(num_runs):
            model(input_data)
    return (time.time() - start) / num_runs * 1000

time_eager = benchmark(model, x)
time_compiled = benchmark(compiled_model, x)

print(f"Eager mode: {time_eager:.2f} ms")
print(f"Compiled: {time_compiled:.2f} ms")
print(f"Speedup: {time_eager / time_compiled:.2f}x")
```

### 3. CUDA 优化

```python
import torch
import torch.nn as nn

# === CUDA 内存优化 ===

# 1. 使用 CUDA Streams 实现并行
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

x1 = torch.randn(1000, 1000, device='cuda')
x2 = torch.randn(1000, 1000, device='cuda')

with torch.cuda.stream(stream1):
    y1 = torch.matmul(x1, x1)

with torch.cuda.stream(stream2):
    y2 = torch.matmul(x2, x2)

# 同步
torch.cuda.synchronize()

# 2. 使用 CUDA Graph（减少 kernel launch 开销）
model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10)
).cuda()
model.eval()

# 预热和捕获 CUDA Graph
static_input = torch.randn(32, 512, device='cuda')
static_output = torch.empty(32, 10, device='cuda')

# 预热
for _ in range(10):
    with torch.inference_mode():
        model(static_input)

# 捕获 Graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_output = model(static_input)

# 使用 Graph 推理（更快）
def inference_with_graph(new_input):
    static_input.copy_(new_input)
    graph.replay()
    return static_output.clone()

# 测试
new_data = torch.randn(32, 512, device='cuda')
result = inference_with_graph(new_data)
print(f"Output shape: {result.shape}")
```

### 4. 批处理优化

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class BatchOptimizedInference:
    """批处理推理优化"""

    def __init__(self, model, max_batch_size=32, max_wait_time=0.01):
        self.model = model
        self.model.eval()
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.buffer = []

    def add_request(self, input_data):
        """添加推理请求"""
        self.buffer.append(input_data)

        if len(self.buffer) >= self.max_batch_size:
            return self._process_batch()
        return None

    def _process_batch(self):
        """处理批次"""
        if not self.buffer:
            return []

        # 组装批次
        batch = torch.stack(self.buffer)

        with torch.inference_mode():
            outputs = self.model(batch)

        # 清空缓冲
        self.buffer = []

        return outputs

# 使用示例
model = nn.Linear(512, 10)
batch_processor = BatchOptimizedInference(model, max_batch_size=8)

# 模拟请求
for i in range(10):
    x = torch.randn(512)
    result = batch_processor.add_request(x)
    if result is not None:
        print(f"Processed batch, output shape: {result.shape}")
```

### 5. KV Cache 优化（LLM 推理）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionWithKVCache(nn.Module):
    """带 KV Cache 的注意力，用于 LLM 推理加速"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        """
        x: (batch, seq_len, d_model)
        kv_cache: tuple of (cached_k, cached_v) or None
        """
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 使用缓存
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        # 保存新的缓存
        new_kv_cache = (k.detach(), v.detach())

        # 多头注意力
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 因果 mask
        seq_len = k.size(2)
        mask = torch.triu(torch.ones(T, seq_len, device=x.device), diagonal=seq_len-T+1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), new_kv_cache

# 使用示例
attn = AttentionWithKVCache(512, 8)

# 首次推理（处理 prompt）
x = torch.randn(1, 10, 512)  # prompt
output, kv_cache = attn(x)
print(f"Initial output: {output.shape}")

# 后续推理（逐 token 生成，使用缓存）
for i in range(5):
    new_token = torch.randn(1, 1, 512)  # 单个新 token
    output, kv_cache = attn(new_token, kv_cache)
    print(f"Step {i+1}, cache size: {kv_cache[0].shape[1]}")
```

### 6. 模型剪枝

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class PrunableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = PrunableModel()

# 查看原始参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_nonzero(model):
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p != 0).sum().item()
    return nonzero, total

print(f"Total parameters: {count_parameters(model):,}")

# 结构化剪枝（按 L1 范数）
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)  # 剪枝 30%
prune.l1_unstructured(model.fc2, name='weight', amount=0.3)

# 全局剪枝
parameters_to_prune = (
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.4  # 全局剪枝 40%
)

nonzero, total = count_nonzero(model)
print(f"Nonzero parameters: {nonzero:,} / {total:,} ({nonzero/total*100:.1f}%)")

# 移除剪枝 mask，使剪枝永久化
for module, name in parameters_to_prune:
    prune.remove(module, name)
```

## 优化技术对比

| 技术 | 加速比 | 实现难度 | 适用场景 |
|------|--------|----------|----------|
| torch.compile | 1.5-3x | ⭐ 简单 | 通用 |
| 半精度 (FP16) | 1.5-2x | ⭐ 简单 | GPU 推理 |
| CUDA Graph | 2-5x | ⭐⭐ 中等 | 固定输入形状 |
| KV Cache | 10x+ | ⭐⭐ 中等 | LLM 生成 |
| 量化 (INT8) | 2-4x | ⭐⭐ 中等 | 通用 |
| 剪枝 | 1.5-3x | ⭐⭐⭐ 复杂 | 模型压缩 |

## 下一步 Next

[下一章：模型服务化 →](./04-serving.md)
