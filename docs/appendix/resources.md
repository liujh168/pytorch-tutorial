# 学习资源与论文推荐 (Resources)

## 官方资源

### PyTorch 官方

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html) - 最权威的参考
- [PyTorch 教程](https://pytorch.org/tutorials/) - 官方教程，覆盖各种主题
- [PyTorch Examples](https://github.com/pytorch/examples) - 官方示例代码
- [PyTorch Blog](https://pytorch.org/blog/) - 最新特性和最佳实践

### HuggingFace

- [Transformers 文档](https://huggingface.co/docs/transformers) - 预训练模型库
- [HuggingFace Course](https://huggingface.co/course) - 免费 NLP 课程
- [PEFT 文档](https://huggingface.co/docs/peft) - 高效微调方法
- [Datasets](https://huggingface.co/docs/datasets) - 数据集加载库

## 在线课程

### 深度学习基础

| 课程 | 平台 | 难度 | 特点 |
|------|------|------|------|
| [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) | Coursera | ⭐⭐ | 吴恩达，经典入门 |
| [Fast.ai](https://course.fast.ai/) | fast.ai | ⭐⭐ | 实践导向，PyTorch |
| [MIT 6.S191](http://introtodeeplearning.com/) | MIT | ⭐⭐ | 理论扎实 |
| [Stanford CS231n](http://cs231n.stanford.edu/) | Stanford | ⭐⭐⭐ | 计算机视觉 |
| [Stanford CS224n](http://web.stanford.edu/class/cs224n/) | Stanford | ⭐⭐⭐ | NLP |

### LLM 专题

| 课程 | 平台 | 特点 |
|------|------|------|
| [LLM Course](https://github.com/mlabonne/llm-course) | GitHub | 免费，全面 |
| [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) | FSDL | 工程实践 |
| [Andrej Karpathy - Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) | YouTube | 从零实现 GPT |

## 经典论文

### Transformer 架构

```
1. Attention Is All You Need (2017)
   - Transformer 原论文
   - https://arxiv.org/abs/1706.03762

2. BERT: Pre-training of Deep Bidirectional Transformers (2018)
   - 双向预训练
   - https://arxiv.org/abs/1810.04805

3. Language Models are Few-Shot Learners (GPT-3, 2020)
   - 大规模语言模型
   - https://arxiv.org/abs/2005.14165

4. Training language models to follow instructions (InstructGPT, 2022)
   - RLHF
   - https://arxiv.org/abs/2203.02155
```

### 高效训练与推理

```
1. LoRA: Low-Rank Adaptation of Large Language Models (2021)
   - 高效微调
   - https://arxiv.org/abs/2106.09685

2. FlashAttention: Fast and Memory-Efficient Exact Attention (2022)
   - 高效注意力
   - https://arxiv.org/abs/2205.14135

3. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (2022)
   - 量化
   - https://arxiv.org/abs/2208.07339

4. Scaling Laws for Neural Language Models (2020)
   - 缩放定律
   - https://arxiv.org/abs/2001.08361
```

### 位置编码

```
1. RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)
   - RoPE
   - https://arxiv.org/abs/2104.09864

2. Train Short, Test Long: Attention with Linear Biases (ALiBi, 2021)
   - 线性偏置
   - https://arxiv.org/abs/2108.12409
```

## 开源项目

### 模型实现

| 项目 | 描述 | 链接 |
|------|------|------|
| nanoGPT | 最简 GPT 实现 | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) |
| minGPT | 教育用 GPT | [karpathy/minGPT](https://github.com/karpathy/minGPT) |
| LLaMA | Meta 开源 LLM | [meta-llama/llama](https://github.com/meta-llama/llama) |
| Mistral | 高效 7B 模型 | [mistralai/mistral-src](https://github.com/mistralai/mistral-src) |

### 训练框架

| 项目 | 描述 | 链接 |
|------|------|------|
| DeepSpeed | 微软分布式训练 | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) |
| Megatron-LM | NVIDIA 大模型训练 | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| ColossalAI | 大模型训练系统 | [hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI) |
| Lightning | PyTorch 训练框架 | [Lightning-AI/pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) |

### 推理优化

| 项目 | 描述 | 链接 |
|------|------|------|
| vLLM | 高吞吐量推理 | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| text-generation-inference | HF 推理服务 | [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) |
| llama.cpp | CPU 推理 | [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |
| TensorRT-LLM | NVIDIA 优化 | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |

## 书籍推荐

### 入门

| 书名 | 作者 | 特点 |
|------|------|------|
| Deep Learning with PyTorch | Eli Stevens 等 | PyTorch 官方推荐 |
| Dive into Deep Learning | 李沐 等 | 免费在线，代码丰富 |
| 动手学深度学习 | 李沐 等 | 中文版，PyTorch |

### 进阶

| 书名 | 作者 | 特点 |
|------|------|------|
| Deep Learning | Goodfellow 等 | 理论经典 |
| Neural Networks and Deep Learning | Michael Nielsen | 免费在线 |
| Transformers for Natural Language Processing | Denis Rothman | Transformer 专题 |

## 社区资源

### 论坛与讨论

- [PyTorch Forums](https://discuss.pytorch.org/) - 官方论坛
- [Stack Overflow - PyTorch](https://stackoverflow.com/questions/tagged/pytorch) - 问答
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/) - 最新动态
- [Reddit r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) - 本地 LLM

### 博客与教程

- [Lil'Log](https://lilianweng.github.io/) - Lilian Weng 的深度学习博客
- [Jay Alammar](https://jalammar.github.io/) - 可视化解释 Transformer
- [Sebastian Raschka](https://sebastianraschka.com/blog/) - 机器学习教程
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 图解 Transformer

### Newsletter

- [The Batch](https://www.deeplearning.ai/the-batch/) - Andrew Ng 的周报
- [Import AI](https://importai.substack.com/) - AI 新闻周报
- [ML News](https://paperswithcode.com/newsletter) - Papers with Code

## 工具与环境

### 开发环境

| 工具 | 用途 |
|------|------|
| [Jupyter](https://jupyter.org/) | 交互式开发 |
| [VS Code](https://code.visualstudio.com/) | IDE |
| [Google Colab](https://colab.research.google.com/) | 免费 GPU |
| [Kaggle Notebooks](https://www.kaggle.com/notebooks) | 免费 GPU/TPU |

### 实验管理

| 工具 | 用途 |
|------|------|
| [Weights & Biases](https://wandb.ai/) | 实验跟踪 |
| [TensorBoard](https://www.tensorflow.org/tensorboard) | 可视化 |
| [MLflow](https://mlflow.org/) | 模型管理 |

### 数据集

| 网站 | 描述 |
|------|------|
| [Hugging Face Datasets](https://huggingface.co/datasets) | NLP 数据集 |
| [Kaggle Datasets](https://www.kaggle.com/datasets) | 各类数据集 |
| [Papers with Code Datasets](https://paperswithcode.com/datasets) | 论文数据集 |

## 学习路径建议

### 初学者 (1-2 个月)

1. 完成 PyTorch 官方教程
2. 学习 Fast.ai 课程
3. 实现 MNIST 分类器
4. 阅读 "Deep Learning with PyTorch"

### 中级 (2-4 个月)

1. 深入学习 Transformer 架构
2. 阅读 Attention Is All You Need
3. 实现 minGPT
4. 学习 HuggingFace Transformers

### 高级 (4-6 个月)

1. 学习分布式训练 (DDP, FSDP)
2. 研究 LoRA 和量化
3. 阅读最新 LLM 论文
4. 参与开源项目

## 返回

[← 返回目录](../README.md)
