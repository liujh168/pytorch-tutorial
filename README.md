# PyTorch 深度学习教程

> 从零基础到理解、实现和部署 LLM 的完整学习路径

## 环境要求

- **Python**: 3.10+
- **硬件**: Apple M4 Pro, 48GB RAM
- **GPU 加速**: MPS (Metal Performance Shaders)

## 快速开始

### 1. 创建虚拟环境

```bash
cd pytorch
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动 Jupyter

```bash
jupyter notebook
```

浏览器打开后，进入 `notebooks/` 目录，点击 `01_tensor_basics.ipynb` 开始学习。

## 目录结构

```
pytorch/
├── README.md           # 本文件
├── requirements.txt    # Python 依赖
├── docs/               # 学习文档 (39 篇 Markdown)
│   ├── 01-basics/          # 基础篇
│   ├── 02-core/            # 核心篇
│   ├── 03-training-advanced/  # 训练进阶
│   ├── 04-attention-transformer/  # Attention 与 Transformer
│   ├── 05-llm/             # LLM 篇
│   ├── 06-deployment/      # 部署篇
│   ├── 07-practice/        # 实战篇
│   └── appendix/           # 附录
└── notebooks/          # Jupyter Notebooks (交互式练习)
    └── 01_tensor_basics.ipynb
```

## 学习方式

1. **阅读文档** - `docs/` 中的 Markdown 文件提供详细理论讲解
2. **动手实践** - `notebooks/` 中的 Jupyter Notebook 用于交互式练习

### 推荐学习顺序

```
入门阶段 (1-2周)
  docs/01-basics → docs/02-core
  配合 notebooks/01_tensor_basics.ipynb

进阶阶段 (2-3周)
  docs/03-training-advanced → docs/04-attention-transformer

专业阶段 (2-3周)
  docs/05-llm → docs/06-deployment

实战阶段 (持续)
  docs/07-practice
```

## MPS 加速

Apple Silicon (M4) 使用 MPS 后端进行 GPU 加速：

```python
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 MPS 加速")
else:
    device = torch.device("cpu")

# 将 tensor 移动到 MPS
x = torch.randn(1000, 1000, device=device)
```

## 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

预期输出：
- PyTorch 版本 >= 2.0
- MPS 可用：True

## 文档索引

详细的文档目录请查看 [docs/README.md](./docs/README.md)

---

**开始学习 → [notebooks/01_tensor_basics.ipynb](./notebooks/01_tensor_basics.ipynb)**
