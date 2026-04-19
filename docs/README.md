# PyTorch 深度学习教程

> 从零基础到理解、实现和部署 LLM 的完整学习路径

## 概述 Overview

本教程是一套**深度体系化**的 PyTorch 学习资料，专为有 Python 基础但没有深度学习经验的学习者设计。通过 7 个循序渐进的模块，你将掌握从基础张量操作到实现和部署大语言模型（LLM）所需的全部知识。

### 学习目标

- 掌握 PyTorch 核心概念和 API
- 理解深度学习训练的完整流程
- 深入理解 Attention 和 Transformer 架构
- 能够从零实现一个迷你 GPT 模型
- 掌握模型部署和优化技术

## 环境要求 Environment Setup

```bash
# Python 版本
Python >= 3.8

# 核心依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib

# 可选依赖（特定章节需要）
pip install transformers datasets  # LLM 相关
pip install onnx onnxruntime       # 部署相关
pip install tensorboard            # 可视化
```

### 验证安装

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 学习路径 Learning Path

### 推荐学习顺序

```
┌─────────────────────────────────────────────────────────────────┐
│                        入门阶段 (1-2周)                          │
│  01-basics ──────────────────────────► 02-core                  │
│  Tensor基础 → 自动微分 → nn.Module → 数据加载 → 训练循环          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        进阶阶段 (2-3周)                          │
│  03-training-advanced ─────────────► 04-attention-transformer   │
│  优化器 → 正则化 → 梯度技巧 → GPU训练 → Attention → Transformer   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        专业阶段 (2-3周)                          │
│  05-llm ────────────────────────────► 06-deployment             │
│  分词器 → 语言模型 → GPT架构 → 模型导出 → 量化 → 服务化           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        实战阶段 (持续)                           │
│  07-practice                                                     │
│  MNIST → 文本分类 → Mini-GPT → 微调LLM                           │
└─────────────────────────────────────────────────────────────────┘
```

## 模块总览 Module Overview

### 📘 [01-basics](./01-basics/) - 基础篇

PyTorch 的核心概念，为后续学习打下坚实基础。

| 文档 | 内容 | 预计时间 |
|------|------|----------|
| [01-introduction](./01-basics/01-introduction.md) | PyTorch 简介与环境搭建 | 1h |
| [02-tensors](./01-basics/02-tensors.md) | Tensor 基础操作 | 2h |
| [03-autograd](./01-basics/03-autograd.md) | 自动微分机制详解 | 2h |
| [04-first-nn](./01-basics/04-first-nn.md) | 构建第一个神经网络 | 2h |

### 📗 [02-core](./02-core/) - 核心篇

深入理解 PyTorch 的核心组件和工作流程。

| 文档 | 内容 | 预计时间 |
|------|------|----------|
| [01-nn-module](./02-core/01-nn-module.md) | nn.Module 深入理解 | 2h |
| [02-data-loading](./02-core/02-data-loading.md) | 数据加载与预处理 | 2h |
| [03-training-loop](./02-core/03-training-loop.md) | 训练循环最佳实践 | 2h |
| [04-loss-functions](./02-core/04-loss-functions.md) | 损失函数详解 | 2h |

### 📙 [03-training-advanced](./03-training-advanced/) - 训练进阶篇 ⭐

掌握训练深度学习模型的高级技术。

| 文档 | 内容 | 预计时间 |
|------|------|----------|
| [01-optimizers](./03-training-advanced/01-optimizers.md) | 优化器深度对比 | 3h |
| [02-lr-schedulers](./03-training-advanced/02-lr-schedulers.md) | 学习率调度策略 | 2h |
| [03-regularization](./03-training-advanced/03-regularization.md) | 正则化技术 | 3h |
| [04-gradient-techniques](./03-training-advanced/04-gradient-techniques.md) | 梯度技巧 | 3h |
| [05-gpu-training](./03-training-advanced/05-gpu-training.md) | GPU 训练与多卡并行 | 4h |
| [06-debugging-profiling](./03-training-advanced/06-debugging-profiling.md) | 调试与性能分析 | 2h |

### 📕 [04-attention-transformer](./04-attention-transformer/) - Attention 与 Transformer 篇 ⭐

深入理解现代深度学习的核心架构。

| 文档 | 内容 | 预计时间 |
|------|------|----------|
| [01-attention-intuition](./04-attention-transformer/01-attention-intuition.md) | 注意力机制直觉理解 | 2h |
| [02-self-attention](./04-attention-transformer/02-self-attention.md) | Self-Attention 数学推导与实现 | 3h |
| [03-multi-head-attention](./04-attention-transformer/03-multi-head-attention.md) | Multi-Head Attention 详解 | 3h |
| [04-positional-encoding](./04-attention-transformer/04-positional-encoding.md) | 位置编码详解 | 3h |
| [05-transformer-block](./04-attention-transformer/05-transformer-block.md) | Transformer Block 完整实现 | 4h |
| [06-encoder-decoder](./04-attention-transformer/06-encoder-decoder.md) | Encoder-Decoder 架构 | 3h |
| [07-efficient-attention](./04-attention-transformer/07-efficient-attention.md) | 高效注意力机制 | 3h |

### 📓 [05-llm](./05-llm/) - LLM 篇

理解大语言模型的核心原理。

| 文档 | 内容 | 预计时间 |
|------|------|----------|
| [01-tokenization](./05-llm/01-tokenization.md) | 分词器原理 | 3h |
| [02-embeddings](./05-llm/02-embeddings.md) | 词嵌入与位置嵌入 | 2h |
| [03-language-modeling](./05-llm/03-language-modeling.md) | 语言模型目标函数 | 3h |
| [04-gpt-architecture](./05-llm/04-gpt-architecture.md) | GPT 架构详解 | 4h |
| [05-training-llm](./05-llm/05-training-llm.md) | LLM 训练技术 | 4h |
| [06-generation](./05-llm/06-generation.md) | 文本生成策略 | 3h |

### 📒 [06-deployment](./06-deployment/) - 部署篇 ⭐

将模型从研究环境带到生产环境。

| 文档 | 内容 | 预计时间 |
|------|------|----------|
| [01-model-export](./06-deployment/01-model-export.md) | 模型导出 | 3h |
| [02-quantization](./06-deployment/02-quantization.md) | 模型量化 | 4h |
| [03-inference-optimization](./06-deployment/03-inference-optimization.md) | 推理优化技术 | 3h |
| [04-serving](./06-deployment/04-serving.md) | 模型服务化 | 4h |
| [05-edge-deployment](./06-deployment/05-edge-deployment.md) | 边缘设备部署 | 3h |

### 📔 [07-practice](./07-practice/) - 实战篇

通过完整项目巩固所学知识。

| 文档 | 内容 | 预计时间 |
|------|------|----------|
| [01-mnist-classifier](./07-practice/01-mnist-classifier.md) | 项目1: 手写数字识别 | 4h |
| [02-text-classification](./07-practice/02-text-classification.md) | 项目2: 文本分类 | 5h |
| [03-mini-gpt](./07-practice/03-mini-gpt.md) | 项目3: 从零实现迷你 GPT | 8h |
| [04-finetune-llm](./07-practice/04-finetune-llm.md) | 项目4: 微调开源 LLM | 6h |

### 📎 [appendix](./appendix/) - 附录

快速参考和补充资源。

| 文档 | 内容 |
|------|------|
| [cheatsheet](./appendix/cheatsheet.md) | PyTorch 常用 API 速查表 |
| [common-errors](./appendix/common-errors.md) | 常见错误与解决方案 |
| [resources](./appendix/resources.md) | 学习资源与论文推荐 |

## 文档约定 Conventions

### 代码示例

所有代码示例都可以直接复制运行：

```python
import torch

# 完整可运行的代码
x = torch.randn(3, 4)
print(x.shape)  # torch.Size([3, 4])
```

### 术语规范

- **中文讲解**：概念解释使用中文
- **英文术语保留**：如 Tensor、Gradient、Attention 等
- **API 注释使用英文**：便于与官方文档对照

### 难度标记

- 🟢 入门级：基础概念，适合初学者
- 🟡 进阶级：需要一定基础
- 🔴 高级：深入原理或复杂实现

## 快速开始 Quick Start

1. 克隆或下载本教程
2. 安装依赖环境
3. 从 [01-basics/01-introduction](./01-basics/01-introduction.md) 开始

```bash
# 验证环境
python -c "import torch; print(torch.__version__)"
```

## 反馈与贡献 Feedback

如果你发现文档中的错误或有改进建议，欢迎提出 Issue 或 PR。

## 许可证 License

本教程采用 MIT 许可证。

---

**开始你的 PyTorch 学习之旅吧！** 🚀

[立即开始 → 01-basics/01-introduction](./01-basics/01-introduction.md)
