# 项目3: 从零实现迷你 GPT (Mini GPT from Scratch)

## 概述 Overview

从零开始实现一个可训练的小型 GPT 模型，深入理解语言模型的工作原理。

## 完整代码 Complete Code

### 1. 配置与数据

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    """GPT 配置"""
    vocab_size: int = 256  # 字符级
    block_size: int = 128  # 上下文长度
    n_embd: int = 128      # 嵌入维度
    n_head: int = 4        # 注意力头数
    n_layer: int = 4       # Transformer 层数
    dropout: float = 0.1
    bias: bool = False     # Linear 层是否使用 bias

# 字符级分词器
class CharTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def fit(self, text):
        chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        return len(chars)

    def encode(self, text):
        return [self.char2idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.idx2char[i] for i in indices])

    def __len__(self):
        return len(self.char2idx)

# 数据集
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y

# 准备数据
text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
""" * 100  # 重复以获得更多训练数据

tokenizer = CharTokenizer()
vocab_size = tokenizer.fit(text)

data = tokenizer.encode(text)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

config = GPTConfig(vocab_size=vocab_size)

train_dataset = TextDataset(train_data, config.block_size)
val_dataset = TextDataset(val_data, config.block_size)

print(f"Vocab size: {vocab_size}")
print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
```

### 2. GPT 模型实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """因果自注意力"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V 投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果 mask
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer('mask', mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 分头
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 注意力
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 输出
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    """前馈网络"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MiniGPT(nn.Module):
    """迷你 GPT 模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化
        self.apply(self._init_weights)

        # 特殊初始化
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"Parameters: {self.get_num_params():,}")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size

        # 位置编码
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        # 嵌入
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """生成文本"""
        for _ in range(max_new_tokens):
            # 截断到 block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniGPT(config).to(device)
```

### 3. 训练循环

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def train(model, train_dataset, val_dataset, config, device):
    """训练 GPT"""

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * 10,
        eta_min=1e-5
    )

    best_val_loss = float('inf')

    for epoch in range(10):
        # 训练
        model.train()
        train_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'mini_gpt.pth')
            print("Model saved!")

        # 生成示例
        model.eval()
        prompt = "To be"
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        generated = model.generate(prompt_ids, max_new_tokens=50, temperature=0.8, top_k=40)
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Generated: {generated_text}")

    return model

# 训练
model = train(model, train_dataset, val_dataset, config, device)
```

### 4. 文本生成

```python
import torch

def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    device='cpu'
):
    """使用 Top-k 和 Top-p 采样生成文本"""
    model.eval()
    model = model.to(device)

    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 获取上下文
            context = generated[:, -model.config.block_size:]

            # 前向传播
            logits, _ = model(context)
            logits = logits[:, -1, :] / temperature

            # Top-k 过滤
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0].tolist())

# 生成示例
model.load_state_dict(torch.load('mini_gpt.pth'))

print("=== 生成示例 ===")
for temp in [0.5, 0.8, 1.0]:
    print(f"\nTemperature {temp}:")
    text = generate_text(
        model, tokenizer,
        prompt="To be or not",
        max_new_tokens=100,
        temperature=temp,
        device=device
    )
    print(text)
```

### 5. 模型分析

```python
import matplotlib.pyplot as plt
import numpy as np

def analyze_attention(model, tokenizer, text, device='cpu'):
    """可视化注意力权重"""
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(text)], device=device)

    # 注册 hook 来捕获注意力权重
    attention_weights = []

    def hook_fn(module, input, output):
        # 在 softmax 之后捕获
        pass  # 需要修改模型来返回注意力

    # 简化版：直接计算
    with torch.no_grad():
        B, T = input_ids.size()
        pos = torch.arange(T, device=device).unsqueeze(0)

        tok_emb = model.transformer.wte(input_ids)
        pos_emb = model.transformer.wpe(pos)
        x = tok_emb + pos_emb

        # 获取第一层的注意力
        block = model.transformer.h[0]
        qkv = block.attn.c_attn(block.ln_1(x))
        q, k, v = qkv.split(model.config.n_embd, dim=2)

        n_head = model.config.n_head
        head_dim = model.config.n_embd // n_head

        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=device))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

    # 可视化
    att_np = att[0].cpu().numpy()  # (n_head, T, T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    chars = list(text)

    for i, ax in enumerate(axes.flat):
        if i < n_head:
            im = ax.imshow(att_np[i], cmap='viridis')
            ax.set_title(f"Head {i+1}")
            ax.set_xticks(range(len(chars)))
            ax.set_yticks(range(len(chars)))
            ax.set_xticklabels(chars)
            ax.set_yticklabels(chars)
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('attention_visualization.png')
    plt.close()

# analyze_attention(model, tokenizer, "To be or not to be", device=device)

def count_parameters_by_layer(model):
    """统计各层参数量"""
    stats = {}

    for name, param in model.named_parameters():
        layer_type = name.split('.')[0]
        if layer_type not in stats:
            stats[layer_type] = 0
        stats[layer_type] += param.numel()

    print("=== 参数分布 ===")
    total = sum(stats.values())
    for name, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"{name}: {count:,} ({count/total*100:.1f}%)")

count_parameters_by_layer(model)
```

## 扩展练习

1. **增加模型规模**：尝试更大的 `n_embd`, `n_layer`, `n_head`
2. **使用真实数据**：用 Wikipedia 或书籍数据训练
3. **实现 KV Cache**：加速推理
4. **添加 Flash Attention**：提升训练效率

## 下一步 Next

[下一章：微调开源 LLM →](./04-finetune-llm.md)
