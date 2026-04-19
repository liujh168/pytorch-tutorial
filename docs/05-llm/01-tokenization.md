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

## 深入理解：BPE 算法逐步追踪

```python
from collections import Counter, defaultdict

def bpe_step_by_step(corpus: list[str], n_merges: int = 10):
    """可视化 BPE 合并过程，逐步展示每次合并"""
    # 初始化：单词 → 字符列表，末尾加 </w> 表示词尾
    word_freqs = Counter(corpus)
    vocab = {word: list(word) + ['</w>'] for word in word_freqs}

    print("初始分割:")
    for word, splits in vocab.items():
        print(f"  {word!r:10s} → {splits}")

    merge_rules = []
    for merge_idx in range(n_merges):
        # 统计所有相邻 pair 的频率
        pair_freqs = defaultdict(int)
        for word, splits in vocab.items():
            freq = word_freqs[word]
            for i in range(len(splits) - 1):
                pair_freqs[(splits[i], splits[i+1])] += freq

        if not pair_freqs:
            break

        # 选择频率最高的 pair
        best = max(pair_freqs, key=pair_freqs.get)
        merge_rules.append(best)
        new_token = ''.join(best)

        print(f"\n合并 {merge_idx+1}: {best} → '{new_token}'  (频率={pair_freqs[best]})")

        # 更新所有单词的分割
        for word in vocab:
            splits = vocab[word]
            new_splits = []
            i = 0
            while i < len(splits):
                if i < len(splits) - 1 and (splits[i], splits[i+1]) == best:
                    new_splits.append(new_token)
                    i += 2
                else:
                    new_splits.append(splits[i])
                    i += 1
            vocab[word] = new_splits

        # 打印更新后的分割
        for word, splits in vocab.items():
            print(f"  {word!r:10s} → {splits}")

    return merge_rules


corpus = ['low'] * 5 + ['lower'] * 2 + ['newest'] * 6 + ['widest'] * 3
print("语料库:", Counter(corpus))
print("=" * 50)
merge_rules = bpe_step_by_step(corpus, n_merges=8)
print(f"\n最终合并规则: {merge_rules}")
```

## 深入理解：特殊 Token 处理

```python
from transformers import AutoTokenizer

# 不同模型的特殊 token 设计
tokenizer_gpt2 = AutoTokenizer.from_pretrained('gpt2')

print("=== GPT-2 分词器特殊 Token ===")
print(f"词汇表大小: {tokenizer_gpt2.vocab_size}")
print(f"EOS token:  '{tokenizer_gpt2.eos_token}' (ID={tokenizer_gpt2.eos_token_id})")
print(f"PAD token:  {tokenizer_gpt2.pad_token}  (GPT-2 默认无 PAD)")

# GPT-2 没有 pad_token，批量处理时需要手动添加
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token   # 常用解决方案

# 编码 & 解码示例
text = "深度学习很有趣！"
encoded = tokenizer_gpt2(text, return_tensors='pt', padding=True)
print(f"\n文本: {text!r}")
print(f"Token IDs: {encoded['input_ids'][0].tolist()}")
print(f"Tokens:    {tokenizer_gpt2.convert_ids_to_tokens(encoded['input_ids'][0])}")
print(f"还原文本: {tokenizer_gpt2.decode(encoded['input_ids'][0])!r}")

# 批量处理（自动 padding）
texts = ["Hello, world!", "PyTorch is great for deep learning research."]
batch = tokenizer_gpt2(
    texts,
    padding=True,       # 短序列 pad 到最长
    truncation=True,    # 超过 max_length 时截断
    max_length=20,
    return_tensors='pt',
)
print(f"\n批量编码 input_ids shape: {batch['input_ids'].shape}")
print(f"attention_mask:\n{batch['attention_mask']}")  # 0 表示 padding 位置
```

## 深入理解：中文分词的挑战

```python
# 中文无空格，不同分词器处理方式不同
from transformers import AutoTokenizer

# 字符级：每个汉字是一个 token
# BERT 中文版使用此方式
text_cn = "深度学习改变了人工智能的发展方向"

# 模拟字符级分词
char_tokens = list(text_cn)
print(f"字符级分词 ({len(char_tokens)} tokens): {char_tokens}")

# 子词级（GPT-2 对中文效率低，因为用 BPE on bytes）
tokenizer = AutoTokenizer.from_pretrained('gpt2')
ids = tokenizer.encode(text_cn)
print(f"\nGPT-2 分词 ({len(ids)} tokens): {tokenizer.convert_ids_to_tokens(ids)}")
print("(注意：中文字符被拆成 UTF-8 bytes，效率低)")

# 结论：中文 LLM（如 ChatGLM、Qwen）使用专为中文优化的词表，
# 通常将常用汉字直接作为单个 token
```

## 分词算法对比

| 算法 | 特点 | 使用模型 | 中文支持 |
|------|------|----------|----------|
| BPE | 基于频率合并，Byte-level 变体 | GPT-2, RoBERTa | 字节级，效率低 |
| WordPiece | 基于最大似然合并 | BERT | 有专门中文版 |
| SentencePiece | 语言无关，直接处理 Unicode | T5, LLaMA | 良好 |
| Unigram | 基于概率剪枝 | XLNet, mBART | 良好 |

## 下一步 Next

[下一章：词嵌入与位置嵌入 →](./02-embeddings.md)
