def run():
    import sys

    print(">>> 技能 [project_env] 开始执行")

    print(f"Python: {sys.version}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch: 未安装")

    import os
    req = os.path.join(os.path.dirname(__file__), "..", "..", "..", "requirements.txt")
    if os.path.exists(req):
        with open(req, encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"项目依赖: requirements.txt ({count} 个包)")
    else:
        print("项目依赖: 未找到 requirements.txt")

    print(">>> 技能 [project_env] 执行完毕")


if __name__ == "__main__":
    run()
