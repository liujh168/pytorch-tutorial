# 常见错误与解决方案 (Common Errors)

## 1. 形状错误 (Shape Errors)

### RuntimeError: size mismatch

```python
# 错误示例
linear = nn.Linear(100, 50)
x = torch.randn(32, 200)  # 输入维度是 200，不是 100
output = linear(x)  # RuntimeError!

# 解决方案
# 1. 检查输入维度
print(f"Input shape: {x.shape}")
print(f"Expected: {linear.in_features}")

# 2. 修正 Linear 层
linear = nn.Linear(200, 50)

# 3. 或修正输入
x = torch.randn(32, 100)
```

### RuntimeError: mat1 and mat2 shapes cannot be multiplied

```python
# 错误：矩阵乘法维度不匹配
a = torch.randn(3, 4)
b = torch.randn(5, 6)
c = a @ b  # RuntimeError!

# 解决：确保 a 的列数等于 b 的行数
a = torch.randn(3, 4)
b = torch.randn(4, 6)  # 4 == 4
c = a @ b  # OK: (3, 4) @ (4, 6) = (3, 6)
```

### 卷积层形状计算

```python
# 计算卷积输出大小
def conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    return (input_size + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1

# 示例
input_size = 28
kernel_size = 3
padding = 1
stride = 1

output_size = conv_output_size(input_size, kernel_size, stride, padding)
print(f"Output size: {output_size}")  # 28

# 常见配置
# kernel=3, padding=1 -> 保持尺寸
# kernel=3, padding=0 -> 尺寸减2
# stride=2 -> 尺寸减半
```

## 2. 设备错误 (Device Errors)

### RuntimeError: Expected all tensors to be on the same device

```python
# 错误
model = MyModel().cuda()
x = torch.randn(32, 100)  # 在 CPU 上
output = model(x)  # RuntimeError!

# 解决方案 1：移动数据到 GPU
x = x.cuda()
# 或
x = x.to('cuda')

# 解决方案 2：统一使用 device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
x = torch.randn(32, 100).to(device)
```

### CUDA out of memory

```python
# 常见原因和解决方案

# 1. 减小 batch size
train_loader = DataLoader(dataset, batch_size=16)  # 从 32 减到 16

# 2. 使用梯度累积
accumulation_steps = 4
for i, (data, target) in enumerate(loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 使用混合精度
with torch.cuda.amp.autocast():
    output = model(data)
    loss = criterion(output, target)

# 4. 清理缓存
torch.cuda.empty_cache()

# 5. 使用 gradient checkpointing
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)  # 节省内存
        x = checkpoint(self.layer2, x)
        return x
```

## 3. 梯度错误 (Gradient Errors)

### RuntimeError: Trying to backward through the graph a second time

```python
# 错误
output1 = model(x)
loss1 = criterion(output1, y)
loss1.backward()

output2 = model(x)  # 使用相同的 x
loss2 = criterion(output2, y)
loss2.backward()  # RuntimeError!

# 解决方案 1：使用 retain_graph=True
loss1.backward(retain_graph=True)

# 解决方案 2：分离张量
x = x.detach()
output2 = model(x)
```

### Gradients are None

```python
# 问题：参数梯度为 None

# 原因 1：参数未参与计算
param = nn.Parameter(torch.randn(10))
# 如果 param 没有在 forward 中使用，梯度为 None

# 原因 2：使用了 detach()
x = x.detach()  # 切断梯度流

# 原因 3：在 no_grad 块中
with torch.no_grad():
    output = model(x)
    loss = criterion(output, y)
loss.backward()  # 梯度为 None

# 调试：检查 requires_grad
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}, grad={param.grad}")
```

### 梯度爆炸/消失

```python
# 检测梯度爆炸
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name}: grad_norm = {grad_norm:.4f}")
        if grad_norm > 100:
            print(f"  WARNING: Gradient explosion!")
        if grad_norm < 1e-7:
            print(f"  WARNING: Gradient vanishing!")

# 解决方案：梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 使用更好的初始化
nn.init.xavier_uniform_(layer.weight)
nn.init.kaiming_normal_(layer.weight, mode='fan_out')
```

## 4. 数据类型错误 (Dtype Errors)

### RuntimeError: expected scalar type Float but found Double

```python
# 错误
x = torch.tensor([1.0, 2.0, 3.0])  # 默认 float64
model = nn.Linear(3, 1)  # 默认 float32
output = model(x)  # RuntimeError!

# 解决方案
x = x.float()  # 转换为 float32
# 或
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
```

### RuntimeError: expected scalar type Long but found Float

```python
# 错误：CrossEntropyLoss 需要 Long 类型的标签
labels = torch.tensor([0.0, 1.0, 2.0])  # float
loss = nn.CrossEntropyLoss()(logits, labels)  # RuntimeError!

# 解决方案
labels = labels.long()
# 或
labels = torch.tensor([0, 1, 2], dtype=torch.long)
```

## 5. 模型错误 (Model Errors)

### 模型不学习

```python
# 检查清单

# 1. 检查是否在训练模式
model.train()  # 不是 model.eval()

# 2. 检查梯度是否计算
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# 3. 检查学习率
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

# 4. 检查损失是否下降
print(f"Loss: {loss.item()}")

# 5. 检查数据是否正确
print(f"Data range: [{x.min():.2f}, {x.max():.2f}]")
print(f"Labels: {y.unique()}")

# 6. 尝试过拟合小数据集
small_data = train_data[:100]
# 如果不能过拟合，模型或代码有问题
```

### 验证集性能差

```python
# 1. 确保使用 eval 模式
model.eval()
with torch.no_grad():
    output = model(x)

# 2. 检查是否有 Dropout/BatchNorm
# eval() 会改变它们的行为

# 3. 数据预处理一致性
# 确保训练和验证使用相同的预处理

# 4. 检查数据泄露
# 确保验证数据没有出现在训练集中
```

## 6. 加载错误 (Loading Errors)

### RuntimeError: Error(s) in loading state_dict

```python
# 错误：模型结构不匹配
model = NewModel()
model.load_state_dict(torch.load('old_model.pth'))  # RuntimeError!

# 解决方案 1：strict=False
model.load_state_dict(torch.load('old_model.pth'), strict=False)

# 解决方案 2：手动加载匹配的权重
pretrained = torch.load('old_model.pth')
model_dict = model.state_dict()

# 过滤不匹配的键
pretrained = {k: v for k, v in pretrained.items()
              if k in model_dict and v.shape == model_dict[k].shape}

model_dict.update(pretrained)
model.load_state_dict(model_dict)
```

### ModuleNotFoundError 加载模型时

```python
# 错误：使用 torch.save(model, path) 保存的模型
# 加载时需要能够导入原始类定义

# 解决方案：只保存 state_dict
torch.save(model.state_dict(), 'model.pth')

# 加载
model = MyModel()  # 先定义模型
model.load_state_dict(torch.load('model.pth'))
```

## 7. 内存泄漏 (Memory Leaks)

```python
# 常见原因

# 1. 累积张量历史
losses = []
for x, y in loader:
    loss = criterion(model(x), y)
    losses.append(loss)  # 保留计算图！

# 修复
losses.append(loss.item())  # 只保存数值

# 2. 忘记 detach
hidden_states = []
for x in data:
    h = model.encode(x)
    hidden_states.append(h)  # 保留计算图

# 修复
hidden_states.append(h.detach())

# 3. 在循环中创建张量
for epoch in range(100):
    mask = torch.ones(1000, 1000).cuda()  # 每次都创建新的

# 修复：在循环外创建
mask = torch.ones(1000, 1000).cuda()
for epoch in range(100):
    # 使用 mask
    pass
```

## 调试技巧

```python
# 1. 打印中间结果
class DebugModel(nn.Module):
    def forward(self, x):
        print(f"Input: {x.shape}, {x.dtype}, {x.device}")
        x = self.layer1(x)
        print(f"After layer1: {x.shape}")
        return x

# 2. 使用 hooks
def print_hook(module, input, output):
    print(f"{module.__class__.__name__}: {input[0].shape} -> {output.shape}")

for name, layer in model.named_modules():
    layer.register_forward_hook(print_hook)

# 3. 检查 NaN/Inf
def check_nan(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}!")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}!")
```

## 返回

[← 返回目录](../README.md)
