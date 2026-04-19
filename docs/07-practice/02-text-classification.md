# 项目2: 文本分类 (Text Classification)

## 概述 Overview

使用 Transformer 构建情感分析模型，学习 NLP 任务的完整流程。

## 完整代码 Complete Code

### 1. 数据准备

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

# 示例数据（实际项目应使用真实数据集如 IMDB、SST-2）
train_texts = [
    "This movie is great and I loved it",
    "Terrible film, waste of time",
    "Amazing acting and beautiful cinematography",
    "Boring and predictable story",
    "Best movie I have ever seen",
    "Very disappointing experience",
    "Excellent performance by the cast",
    "Poor direction and bad script",
    # ... 更多数据
]

train_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative

# 文本预处理
def preprocess_text(text):
    """简单的文本预处理"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点
    return text.split()

# 构建词汇表
class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.min_freq = min_freq

    def build(self, texts):
        counter = Counter()
        for text in texts:
            words = preprocess_text(text)
            counter.update(words)

        for word, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text, max_len=None):
        words = preprocess_text(text)
        indices = [self.word2idx.get(w, 1) for w in words]  # 1 是 <unk>

        if max_len:
            if len(indices) < max_len:
                indices += [0] * (max_len - len(indices))  # padding
            else:
                indices = indices[:max_len]

        return indices

    def __len__(self):
        return len(self.word2idx)

# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=64):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        input_ids = self.vocab.encode(text, self.max_len)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 构建词汇表和数据集
vocab = Vocabulary(min_freq=1)
vocab.build(train_texts)

print(f"Vocabulary size: {len(vocab)}")

train_dataset = TextDataset(train_texts, train_labels, vocab, max_len=32)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
```

### 2. Transformer 分类器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    """基于 Transformer 的文本分类器"""

    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        num_classes=2,
        max_len=512,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.d_model = d_model

    def forward(self, input_ids, attention_mask=None):
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # 创建 padding mask
        if attention_mask is None:
            attention_mask = (input_ids != 0)

        # Transformer
        # 注意：PyTorch Transformer 的 mask 是反的
        padding_mask = ~attention_mask

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 取 [CLS] 位置（这里用第一个 token）或平均池化
        # 使用平均池化（忽略 padding）
        mask_expanded = attention_mask.unsqueeze(-1).float()
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        # 分类
        logits = self.classifier(x)

        return logits

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerClassifier(
    vocab_size=len(vocab),
    d_model=128,
    nhead=4,
    num_layers=2,
    num_classes=2
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 3. 训练与评估

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )

    return {
        'loss': total_loss / len(loader),
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 训练
EPOCHS = 20
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )

    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
```

### 4. 使用预训练模型（HuggingFace）

```python
# pip install transformers datasets

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch

# 加载预训练模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 加载数据集
dataset = load_dataset("imdb")

# 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 训练配置
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),  # 示例用少量数据
    eval_dataset=tokenized_datasets["test"].select(range(500)),
)

# 训练
# trainer.train()

# 推理
def predict_sentiment(text, model, tokenizer, device='cpu'):
    model.eval()
    model = model.to(device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    pred = probs.argmax().item()
    confidence = probs.max().item()

    label = "Positive" if pred == 1 else "Negative"
    return label, confidence

# 测试
# result = predict_sentiment("This movie is amazing!", model, tokenizer)
# print(result)
```

### 5. 模型推理服务

```python
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# 全局模型和分词器
model = None
tokenizer = None
vocab = None

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict

@app.on_event("startup")
async def load_model():
    global model, vocab
    # 加载自定义模型
    vocab = Vocabulary()
    vocab.build(train_texts)  # 实际应从文件加载

    model = TransformerClassifier(vocab_size=len(vocab))
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    input_ids = torch.tensor([vocab.encode(request.text, max_len=32)])

    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=-1)

    pred = probs.argmax().item()
    confidence = probs.max().item()

    label = "Positive" if pred == 1 else "Negative"

    return PredictResponse(
        label=label,
        confidence=confidence,
        probabilities={
            "negative": probs[0, 0].item(),
            "positive": probs[0, 1].item()
        }
    )

# 运行: uvicorn server:app --host 0.0.0.0 --port 8000
```

## 项目结构

```
text-classification/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── model.py           # 模型定义
│   ├── dataset.py         # 数据集类
│   ├── train.py           # 训练脚本
│   └── inference.py       # 推理脚本
├── configs/
│   └── config.yaml        # 配置文件
├── checkpoints/           # 模型权重
└── requirements.txt
```

## 预期结果

| 模型 | IMDB 准确率 | 参数量 |
|------|------------|--------|
| 自定义 Transformer | ~85% | ~500K |
| BERT-base | ~93% | 110M |
| DistilBERT | ~91% | 66M |

## 下一步 Next

[下一章：从零实现迷你 GPT →](./03-mini-gpt.md)
