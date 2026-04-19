# 分词器原理 (Tokenization)

## 概述 Overview

分词器将文本转换为模型可处理的 token 序列。本章介绍主流分词算法。

## 核心算法

### 1. BPE (Byte Pair Encoding)

```python
from collections import Counter

def train_bpe(corpus, vocab_size):
    """简化的 BPE 训练"""
    # 初始化：每个字符是一个 token
    vocab = set()
    for word in corpus:
        vocab.update(list(word))

    # 将单词拆分为字符
    word_freqs = Counter(corpus)
    splits = {word: list(word) for word in word_freqs}

    while len(vocab) < vocab_size:
        # 统计相邻 pair 频率
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            chars = splits[word]
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pair_freqs[pair] += freq

        if not pair_freqs:
            break

        # 找到最频繁的 pair
        best_pair = pair_freqs.most_common(1)[0][0]

        # 合并这个 pair
        new_token = ''.join(best_pair)
        vocab.add(new_token)

        # 更新所有单词的分割
        for word in splits:
            chars = splits[word]
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair:
                    new_chars.append(new_token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            splits[word] = new_chars

    return vocab, splits

# 测试
corpus = ['low', 'lower', 'lowest', 'new', 'newer', 'newest']
vocab, splits = train_bpe(corpus * 100, vocab_size=20)
print(f"Vocab: {vocab}")
print(f"Splits: {splits}")
```

### 2. 使用 HuggingFace Tokenizers

```python
from transformers import AutoTokenizer

# 加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained('gpt2')

text = "Hello, how are you today?"

# 编码
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

ids = tokenizer.encode(text)
print(f"Token IDs: {ids}")

# 解码
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")

# 批量处理
batch = tokenizer(
    ["Hello world", "How are you?"],
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
print(f"Input IDs shape: {batch['input_ids'].shape}")
print(f"Attention mask: {batch['attention_mask']}")

# 特殊 token
print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"Vocab size: {tokenizer.vocab_size}")
```

### 3. 训练自定义分词器

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 创建 BPE 分词器
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 训练
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>"]
)

# 从文件训练
# tokenizer.train(["data/corpus.txt"], trainer)

# 或从内存训练
corpus = ["Hello world", "This is a test", "Training tokenizer"]
tokenizer.train_from_iterator(corpus, trainer)

# 保存
# tokenizer.save("my_tokenizer.json")
```

## 分词算法对比

| 算法 | 特点 | 使用模型 |
|------|------|----------|
| BPE | 基于频率合并 | GPT-2, RoBERTa |
| WordPiece | 基于似然合并 | BERT |
| SentencePiece | 语言无关 | T5, LLaMA |
| Unigram | 基于概率 | XLNet |

## 下一步 Next

[下一章：词嵌入与位置嵌入 →](./02-embeddings.md)
