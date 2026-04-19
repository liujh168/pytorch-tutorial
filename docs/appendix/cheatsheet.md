# PyTorch 常用 API 速查表 (Cheatsheet)

## Tensor 操作

### 创建 Tensor

```python
import torch

# 基本创建
torch.tensor([1, 2, 3])              # 从列表
torch.zeros(3, 4)                     # 全零
torch.ones(3, 4)                      # 全一
torch.empty(3, 4)                     # 未初始化
torch.full((3, 4), 5.0)              # 填充值

# 随机
torch.rand(3, 4)                      # [0, 1) 均匀分布
torch.randn(3, 4)                     # 标准正态分布
torch.randint(0, 10, (3, 4))         # 随机整数

# 序列
torch.arange(0, 10, 2)               # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)              # [0, 0.25, 0.5, 0.75, 1]

# 单位矩阵/对角矩阵
torch.eye(3)                          # 3x3 单位矩阵
torch.diag(torch.tensor([1,2,3]))    # 对角矩阵

# 从 NumPy
torch.from_numpy(np_array)
tensor.numpy()                        # 转回 NumPy
```

### 形状操作

```python
# 查看形状
x.shape                               # 或 x.size()
x.dim()                               # 维度数
x.numel()                             # 元素总数

# 改变形状
x.view(2, 6)                          # 必须连续
x.reshape(2, 6)                       # 更灵活
x.contiguous()                        # 转为连续

# 增减维度
x.unsqueeze(0)                        # 在位置0增加维度
x.squeeze()                           # 移除大小为1的维度
x.expand(3, 4, 5)                     # 扩展（不复制数据）

# 转置
x.T                                   # 2D 转置
x.transpose(0, 1)                     # 交换两个维度
x.permute(2, 0, 1)                    # 重排所有维度

# 拼接/分割
torch.cat([a, b], dim=0)             # 沿维度拼接
torch.stack([a, b], dim=0)           # 新维度堆叠
torch.split(x, 2, dim=0)             # 分割
torch.chunk(x, 3, dim=0)             # 等分
```

### 数学运算

```python
# 基本运算
x + y                                 # 或 torch.add(x, y)
x - y                                 # 或 torch.sub(x, y)
x * y                                 # 逐元素乘法
x / y                                 # 逐元素除法
x ** 2                                # 幂运算

# 矩阵运算
x @ y                                 # 矩阵乘法
torch.matmul(x, y)                   # 矩阵乘法
torch.mm(x, y)                       # 2D 矩阵乘法
torch.bmm(x, y)                      # 批量矩阵乘法

# 规约操作
x.sum()                               # 求和
x.mean()                              # 均值
x.max()                               # 最大值
x.min()                               # 最小值
x.argmax(dim=1)                      # 最大值索引

# 其他
torch.abs(x)                          # 绝对值
torch.sqrt(x)                         # 平方根
torch.exp(x)                          # 指数
torch.log(x)                          # 自然对数
torch.clamp(x, min=0, max=1)         # 裁剪
```

## 神经网络 (nn.Module)

### 常用层

```python
import torch.nn as nn

# 线性层
nn.Linear(in_features, out_features)

# 卷积层
nn.Conv1d(in_ch, out_ch, kernel_size)
nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=0)
nn.ConvTranspose2d(in_ch, out_ch, kernel_size)

# 池化层
nn.MaxPool2d(kernel_size)
nn.AvgPool2d(kernel_size)
nn.AdaptiveAvgPool2d(output_size)

# 归一化
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)
nn.GroupNorm(num_groups, num_channels)

# Dropout
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)

# 嵌入
nn.Embedding(num_embeddings, embedding_dim)

# RNN
nn.LSTM(input_size, hidden_size, num_layers)
nn.GRU(input_size, hidden_size, num_layers)

# Transformer
nn.MultiheadAttention(embed_dim, num_heads)
nn.TransformerEncoderLayer(d_model, nhead)
nn.TransformerEncoder(encoder_layer, num_layers)
```

### 激活函数

```python
import torch.nn.functional as F

# 函数式
F.relu(x)
F.gelu(x)
F.sigmoid(x)
F.tanh(x)
F.softmax(x, dim=-1)
F.log_softmax(x, dim=-1)

# 模块式
nn.ReLU()
nn.GELU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=-1)
nn.LeakyReLU(0.1)
nn.SiLU()  # Swish
```

### 损失函数

```python
# 分类
nn.CrossEntropyLoss()                 # 多分类 (包含 softmax)
nn.BCEWithLogitsLoss()               # 二分类 (包含 sigmoid)
nn.NLLLoss()                          # 负对数似然

# 回归
nn.MSELoss()                          # 均方误差
nn.L1Loss()                           # 平均绝对误差
nn.SmoothL1Loss()                     # Huber 损失

# 其他
nn.KLDivLoss()                        # KL 散度
nn.CosineEmbeddingLoss()             # 余弦相似度损失
```

## 优化器

```python
import torch.optim as optim

# 常用优化器
optim.SGD(params, lr=0.01, momentum=0.9)
optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))
optim.AdamW(params, lr=1e-3, weight_decay=0.01)

# 学习率调度
optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=1000)
```

## 训练模式

```python
# 模式切换
model.train()                         # 训练模式
model.eval()                          # 评估模式

# 梯度控制
with torch.no_grad():                # 不计算梯度
    pass

with torch.inference_mode():         # 推理模式（更快）
    pass

# 梯度操作
optimizer.zero_grad()                # 清零梯度
loss.backward()                      # 反向传播
optimizer.step()                     # 更新参数

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

## 数据加载

```python
from torch.utils.data import Dataset, DataLoader

# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)
```

## 模型保存/加载

```python
# 保存
torch.save(model.state_dict(), 'model.pth')        # 只保存参数
torch.save(model, 'model_full.pth')                # 保存整个模型

# 加载
model.load_state_dict(torch.load('model.pth'))
model = torch.load('model_full.pth')

# Checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

## 设备管理

```python
# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 移动到设备
tensor = tensor.to(device)
model = model.to(device)

# GPU 信息
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.memory_allocated()
torch.cuda.empty_cache()
```

## 混合精度

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in loader:
    optimizer.zero_grad()

    with autocast():                 # 自动混合精度
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## TorchScript

```python
# Tracing
traced = torch.jit.trace(model, example_input)

# Scripting
scripted = torch.jit.script(model)

# 保存
traced.save('model.pt')

# 加载
loaded = torch.jit.load('model.pt')
```

## 返回

[← 返回目录](../README.md)
