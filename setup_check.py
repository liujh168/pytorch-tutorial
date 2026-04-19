"""
PyTorch 学习环境验证脚本
运行方式: python setup_check.py
"""

import sys
import importlib

# ── 颜色输出 ──────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
BOLD  = "\033[1m"
RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")


# ── 1. Python 版本 ────────────────────────────────────────────
header("1. Python 环境")
major, minor = sys.version_info[:2]
if major == 3 and minor >= 10:
    ok(f"Python {major}.{minor} (推荐 3.10+)")
elif major == 3 and minor >= 8:
    warn(f"Python {major}.{minor} (可用，建议升级到 3.10+)")
else:
    fail(f"Python {major}.{minor} (版本过低，请升级到 3.10+)")


# ── 2. PyTorch 核心 ───────────────────────────────────────────
header("2. PyTorch 核心")
try:
    import torch
    ok(f"torch {torch.__version__}")

    # CUDA
    if torch.cuda.is_available():
        ok(f"CUDA 可用 — 设备: {torch.cuda.get_device_name(0)}")
    else:
        warn("CUDA 不可用")

    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        ok("MPS 可用 (Apple Silicon GPU)")
    else:
        warn("MPS 不可用 (非 Apple Silicon 或驱动问题)")

    # 推荐使用的设备
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  → 推荐使用设备: {BOLD}{device}{RESET}")

    # 简单计算测试
    x = torch.randn(3, 3)
    y = torch.matmul(x, x.T)
    ok("基础张量运算测试通过")

except ImportError:
    fail("torch 未安装 — 请运行: pip install torch torchvision torchaudio")

try:
    import torchvision
    ok(f"torchvision {torchvision.__version__}")
except ImportError:
    fail("torchvision 未安装")

try:
    import torchaudio
    ok(f"torchaudio {torchaudio.__version__}")
except ImportError:
    warn("torchaudio 未安装 (仅音频任务需要)")


# ── 3. Transformers & NLP ─────────────────────────────────────
header("3. Transformers & NLP 工具")
for pkg, name in [
    ("transformers", "transformers"),
    ("datasets",     "datasets"),
    ("tokenizers",   "tokenizers"),
]:
    try:
        mod = importlib.import_module(pkg)
        ok(f"{name} {mod.__version__}")
    except ImportError:
        fail(f"{name} 未安装")


# ── 4. 数据处理与可视化 ───────────────────────────────────────
header("4. 数据处理 & 可视化")
for pkg, name in [
    ("numpy",      "numpy"),
    ("pandas",     "pandas"),
    ("matplotlib", "matplotlib"),
    ("sklearn",    "scikit-learn"),
]:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "unknown")
        ok(f"{name} {version}")
    except ImportError:
        fail(f"{name} 未安装")


# ── 5. 训练工具 ───────────────────────────────────────────────
header("5. 训练辅助工具")
for pkg, name in [
    ("tqdm",        "tqdm"),
    ("tensorboard", "tensorboard"),
]:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "unknown")
        ok(f"{name} {version}")
    except ImportError:
        warn(f"{name} 未安装 (可选)")


# ── 6. Jupyter 环境 ───────────────────────────────────────────
header("6. Jupyter 环境")
for pkg, name in [
    ("jupyter",   "jupyter"),
    ("ipykernel", "ipykernel"),
]:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "unknown")
        ok(f"{name} {version}")
    except ImportError:
        warn(f"{name} 未安装 — Notebook 功能不可用")


# ── 汇总 ──────────────────────────────────────────────────────
print(f"\n{BOLD}{'─'*40}{RESET}")
print(f"{BOLD}环境检查完成{RESET}")
print("如有 ✗ 项，请运行: pip install -r requirements.txt")
print(f"{'─'*40}\n")
