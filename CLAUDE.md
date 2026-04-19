# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This is a PyTorch deep learning tutorial repository (PyTorch深度学习教程) — documentation-first, designed as a self-paced learning path from tensor basics through LLM deployment. Content is primarily Chinese Markdown with English technical terms.

## Environment Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Verify GPU availability (MPS for Apple Silicon, CUDA for NVIDIA):
```bash
python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
```

Launch Jupyter for interactive notebooks:
```bash
jupyter notebook
```

## Repository Structure

```
setup_check.py  # 环境验证脚本（首次配置时运行）
examples/       # 9 个可直接运行的示例脚本
  01_tensor_basics.py       # 张量操作
  02_autograd.py            # 自动微分
  03_nn_module.py           # 自定义模型
  04_data_loading.py        # Dataset / DataLoader
  05_training_loop.py       # 完整训练循环 + 早停
  06_optimizers_comparison.py  # 优化器对比实验
  07_attention_mechanism.py    # 注意力机制手动实现
  08_simple_transformer.py     # 完整 Transformer 编码器
  09_mnist_classifier.py       # 端到端 CNN 分类项目
docs/           # Markdown 课程文件（含练习题）
  01-basics/    # Tensors, autograd, first neural network
  02-core/      # nn.Module, data loading, training loops, loss functions
  03-training-advanced/  # Optimizers, LR scheduling, regularization, GPU
  04-attention-transformer/  # Self-attention, multi-head, positional encoding
  05-llm/       # Tokenization, embeddings, language modeling, GPT
  06-deployment/ # Export, quantization, inference optimization, serving
  07-practice/  # Projects: MNIST, text classification, mini-GPT, fine-tuning
  appendix/     # API cheatsheet, common errors, resources
notebooks/      # Jupyter 交互式练习本
  01_tensor_basics.ipynb       # 基础张量操作
  02_nn_module_training.ipynb  # nn.Module + 训练循环 + TensorBoard
  03_optimizers_schedulers.ipynb  # 优化器对比 + 学习率调度可视化
  04_attention_transformer.ipynb  # 注意力权重可视化 + Transformer
  05_practice_mnist.ipynb         # 完整 MNIST 端到端项目
```

## Learning Path

Recommended progression: `01-basics` → `02-core` → `03-training-advanced` → `04-attention-transformer` → `05-llm` → `06-deployment` → `07-practice`

## Key Dependencies

- **torch 2.0.1** + torchvision + torchaudio
- **transformers 4.30.2** (Hugging Face) for LLM-related content
- **datasets 2.14.0** for data loading examples
- tensorboard for training visualization
