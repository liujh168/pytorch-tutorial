#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 功能说明
# 该脚本实现了一个简单的情感分析模型，基于Transformer编码器架构：

# 数据生成：创建了简单的正面和负面情感文本数据
# 模型架构：
# 包含位置编码（PositionalEncoding）
# 多头注意力机制（MultiHeadAttention）
# 前馈神经网络（PositionWiseFeedForward）
# 编码器层（EncoderLayer）
# 分类器（TransformerClassifier）
# 训练过程：使用交叉熵损失函数和Adam优化器
# 推理测试：对新文本进行情感预测
# 脚本现在可以在本地环境中独立运行，无需网络连接，适合用于学习和测试Transformer模型的基本原理。

# 安装必要的库
# import subprocess
# import sys
# 安装依赖
# print("正在安装依赖...")
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "transformers", "scikit-learn", "pandas"])

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 生成本地假数据
print("正在生成本地数据...")

# 生成一些假的情感分析数据
class FakeDataset:
    def __init__(self):
        # 生成简单的正面和负面句子
        self.train = {
            'sentence': [
                "I love this product!",
                "This is terrible.",
                "The service was amazing.",
                "I hate this movie.",
                "The food was delicious.",
                "The quality is poor.",
                "I'm very happy with this purchase.",
                "This is the worst experience ever."
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0]
        }
        self.validation = {
            'sentence': [
                "This is fantastic!",
                "I'm not satisfied.",
                "The product works great.",
                "This is a waste of money."
            ],
            'label': [1, 0, 1, 0]
        }

dataset = FakeDataset()
print(f"数据生成完成：{len(dataset.train['sentence'])} 条训练数据，{len(dataset.validation['sentence'])} 条验证数据")

# --- 核心组件定义 ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # 调整mask形状以匹配scores: (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        attention = self.scaled_dot_product_attention(Q, K, V, mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attention)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes, max_len=5000, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, src, mask=None):
        encoder_output = self.encoder(src, mask)
        pooled_output = encoder_output.max(dim=1)[0] 
        logits = self.classifier(pooled_output)
        return logits

# 简单的本地分词器
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[UNK]': 3}
        self.vocab_size = 4
        self.frozen = False  # 标记是否冻结词汇表
    
    def tokenize(self, text):
        return text.lower().split()
    
    def encode(self, text, max_length=128):
        tokens = self.tokenize(text)
        input_ids = []
        for token in tokens:
            if token not in self.vocab:
                if self.frozen:
                    # 如果词汇表已冻结，使用[UNK]标记
                    input_ids.append(self.vocab['[UNK]'])
                else:
                    # 否则添加新词汇
                    self.vocab[token] = self.vocab_size
                    self.vocab_size += 1
                    input_ids.append(self.vocab[token])
            else:
                input_ids.append(self.vocab[token])
        
        # 添加CLS和SEP标记
        input_ids = [1] + input_ids + [2]
        
        # 截断或填充到max_length
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        else:
            input_ids += [0] * (max_length - len(input_ids))
        
        attention_mask = [1 if id != 0 else 0 for id in input_ids]
        return input_ids, attention_mask
    
    def freeze(self):
        """冻结词汇表，不再添加新词汇"""
        self.frozen = True

# 初始化分词器
tokenizer = SimpleTokenizer()
MAX_LEN = 128

# 首先处理所有文本，构建完整的词汇表
print("构建词汇表...")
all_texts = dataset.train['sentence'] + dataset.validation['sentence']
for text in all_texts:
    tokenizer.encode(text, max_length=MAX_LEN)
print(f"词汇表构建完成，大小: {tokenizer.vocab_size}")

# 冻结词汇表，推理时不再添加新词汇
tokenizer.freeze()
print("词汇表已冻结")

class TextDataset(Dataset):
    def __init__(self, data):
        self.texts = data['sentence']
        self.labels = data['label']
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids, attention_mask = tokenizer.encode(text, max_length=MAX_LEN)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 创建 DataLoader
train_dataset = TextDataset(dataset.train)
val_dataset = TextDataset(dataset.validation)

BATCH_SIZE = 2
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print("数据预处理完成！")

# 模型超参数
VOCAB_SIZE = tokenizer.vocab_size  # 使用动态构建的词汇表大小
D_MODEL = 64  # 减小模型大小以加快测试
NUM_HEADS = 2
D_FF = 128
NUM_LAYERS = 1
NUM_CLASSES = 2
EPOCHS = 1
LEARNING_RATE = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = TransformerClassifier(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, mask=attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return total_loss / len(loader), correct / total

print("开始训练...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

print("训练完成！")

def predict_sentiment(text):
    model.eval()
    input_ids, attention_mask = tokenizer.encode(text, max_length=MAX_LEN)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, mask=attention_mask)
        _, predicted = torch.max(outputs.data, 1)
        
    sentiment = "积极" if predicted.item() == 1 else "消极"
    return sentiment

test_texts = [
    "This movie is fantastic! I really enjoyed it.",
    "I hate this product. It's terrible.",
    "The service was okay, not great but not bad either."
]

print("\n--- 推理测试 ---")
for text in test_texts:
    result = predict_sentiment(text)
    print(f"文本: {text}\n预测情感: {result}\n")

print("代码运行成功！")
