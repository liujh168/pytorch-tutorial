---
name: project_env
description: 检查本 PyTorch 项目的运行环境是否就绪
---

# 环境检查技能

一句话：检查 Python、PyTorch、CUDA、项目依赖是否就绪。

## 使用示例

**输入：** `检查环境`
**输出：**
```
>>> 技能 [project_env] 开始执行
Python: 3.10.x
PyTorch: 2.4.x
CUDA 可用: True
项目依赖: requirements.txt (12 个包)
>>> 技能 [project_env] 执行完毕
```

## 工作原理（极简解释）

```
用户说"检查环境"
  → AI 匹配到 project_env 技能
  → AI 读取 SKILL.md，理解技能的意图
  → AI 执行 main.py
  → 脚本输出结果
  → AI 将结果呈现给用户
```

## 文件结构

```
project_env/
├── SKILL.md   ← AI 读这个文件来理解技能
└── main.py    ← 技能被调用时执行这个脚本
```
