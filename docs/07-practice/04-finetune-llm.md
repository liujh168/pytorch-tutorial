# 项目4: 微调开源 LLM (Fine-tuning Open Source LLMs)

## 概述 Overview

学习如何使用 LoRA 等高效微调技术对开源 LLM（如 LLaMA、Mistral）进行微调。

## 完整代码 Complete Code

### 1. 环境准备

```python
# 安装依赖
# pip install torch transformers datasets peft accelerate bitsandbytes

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset

# 检查 GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 2. 加载模型和分词器

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 模型名称（使用小型模型演示）
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# 其他选择:
# - "mistralai/Mistral-7B-v0.1"
# - "meta-llama/Llama-2-7b-hf"
# - "Qwen/Qwen-1_8B"

# 4-bit 量化配置（节省显存）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Model loaded: {model_name}")
print(f"Parameters: {model.num_parameters():,}")
```

### 3. 配置 LoRA

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 准备模型用于 k-bit 训练
model = prepare_model_for_kbit_training(model)

# LoRA 配置
lora_config = LoraConfig(
    r=16,                        # LoRA 秩
    lora_alpha=32,               # 缩放因子
    target_modules=[             # 要应用 LoRA 的模块
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 打印可训练参数
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    print(f"All params: {all_params:,}")

print_trainable_parameters(model)
```

### 4. 准备数据集

```python
from datasets import load_dataset

# 加载数据集（示例使用 Alpaca 格式）
# 实际使用时替换为你的数据集
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

# 数据格式化
def format_instruction(sample):
    """格式化为指令格式"""
    if sample.get("input", ""):
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

    return {"text": prompt}

# 格式化数据集
formatted_dataset = dataset.map(format_instruction)

# 分词
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 划分训练集和验证集
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

print(f"Train size: {len(tokenized_dataset['train'])}")
print(f"Val size: {len(tokenized_dataset['test'])}")
```

### 5. 训练

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 数据收集器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 不是 MLM，是 CLM
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./lora-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    report_to="none",  # 或 "wandb"
    optim="paged_adamw_8bit",
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

# 训练
trainer.train()

# 保存 LoRA 权重
model.save_pretrained("./lora-weights")
tokenizer.save_pretrained("./lora-weights")

print("Training completed!")
```

### 6. 推理

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_finetuned_model(base_model_name, lora_path):
    """加载微调后的模型"""
    # 加载基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(base_model, lora_path)

    # 合并权重（可选，用于推理优化）
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text="", max_length=256):
    """生成回复"""
    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取 Response 部分
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response

# 使用示例
# model, tokenizer = load_finetuned_model(model_name, "./lora-weights")
# response = generate_response(model, tokenizer, "Explain what is machine learning.")
# print(response)
```

### 7. 使用 SFTTrainer（推荐）

```python
from trl import SFTTrainer, SFTConfig

# SFTTrainer 更适合指令微调
sft_config = SFTConfig(
    output_dir="./sft-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    max_seq_length=512,
    packing=False,  # 是否打包短样本
)

# 格式化函数
def formatting_func(example):
    if example.get("input", ""):
        text = f"""### Instruction: {example['instruction']}

### Input: {example['input']}

### Response: {example['output']}"""
    else:
        text = f"""### Instruction: {example['instruction']}

### Response: {example['output']}"""
    return text

# 创建 SFTTrainer
sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    formatting_func=formatting_func,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

# 训练
# sft_trainer.train()
```

### 8. 评估与对比

```python
import torch
from tqdm import tqdm

def evaluate_model(model, tokenizer, eval_prompts, max_length=128):
    """评估模型生成质量"""
    model.eval()
    results = []

    for prompt in tqdm(eval_prompts, desc="Evaluating"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "generated": generated
        })

    return results

def compare_models(base_model, finetuned_model, tokenizer, prompts):
    """对比基座模型和微调模型"""
    print("=== Model Comparison ===\n")

    for prompt in prompts:
        print(f"Prompt: {prompt}\n")

        # 基座模型
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
        with torch.no_grad():
            base_output = base_model.generate(**inputs, max_new_tokens=100)
        base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)
        print(f"Base Model: {base_text}\n")

        # 微调模型
        inputs = tokenizer(prompt, return_tensors="pt").to(finetuned_model.device)
        with torch.no_grad():
            ft_output = finetuned_model.generate(**inputs, max_new_tokens=100)
        ft_text = tokenizer.decode(ft_output[0], skip_special_tokens=True)
        print(f"Finetuned Model: {ft_text}\n")

        print("-" * 50 + "\n")

# 测试提示
test_prompts = [
    "### Instruction: Write a short poem about artificial intelligence.\n\n### Response:",
    "### Instruction: Explain the concept of neural networks to a 10-year-old.\n\n### Response:",
]

# compare_models(base_model, finetuned_model, tokenizer, test_prompts)
```

## 项目结构

```
finetune-llm/
├── data/
│   ├── train.jsonl        # 训练数据
│   └── eval.jsonl         # 评估数据
├── configs/
│   └── lora_config.yaml   # LoRA 配置
├── scripts/
│   ├── prepare_data.py    # 数据准备
│   ├── train.py           # 训练脚本
│   └── inference.py       # 推理脚本
├── checkpoints/           # 模型权重
└── requirements.txt
```

## 微调技术对比

| 技术 | 显存需求 | 训练速度 | 效果 |
|------|----------|----------|------|
| 全量微调 | 最高 | 最慢 | 最好 |
| LoRA | 低 | 快 | 很好 |
| QLoRA | 最低 | 中等 | 很好 |
| Prefix Tuning | 低 | 快 | 一般 |

## 下一步 Next

恭喜完成实战篇！请查看附录获取更多资源。

[附录：常用 API 速查表 →](../appendix/cheatsheet.md)
