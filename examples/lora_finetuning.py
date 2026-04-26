"""
示例: LoRA微调技术
对应文档: docs/LLM Code/Char01.txt - 1.4.1节
功能: 基于PyTorch和Hugging Face实现LoRA微调技术
"""

# 模拟LoRA微调技术演示（无需网络连接）
import torch
import torch.nn as nn
import json
import os


class MockTokenizer:
    """模拟分词器"""
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "今": 2, "天": 3, "是": 4, "个": 5, "好": 6, "天": 7, "气": 8, "。": 9, "我": 10, "喜": 11, "欢": 12, "用": 13, "G": 14, "P": 15, "T": 16, "模": 17, "型": 18, "学": 19, "习": 20}
    
    def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=64):
        tokens = [self.vocab.get(char, 1) for char in text[:max_length]]
        if padding and len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.tensor([[1]*len(tokens[:max_length]) + [0]*(max_length - len(tokens[:max_length]))])}
        return {"input_ids": [tokens], "attention_mask": [[1]*len(tokens[:max_length]) + [0]*(max_length - len(tokens[:max_length]))]}
    
    def decode(self, tokens, skip_special_tokens=True):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return "".join([reverse_vocab.get(token, "[UNK]") for token in tokens if token not in [0] or not skip_special_tokens])


class MockGPT2(nn.Module):
    """模拟GPT-2模型"""
    def __init__(self):
        super(MockGPT2, self).__init__()
        self.config = type('Config', (), {'hidden_size': 64, 'vocab_size': 21})()
        self.embedding = nn.Embedding(21, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 21)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        logits = self.fc(x)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, 21), labels.view(-1))
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    def generate(self, input_ids, max_length=50, num_return_sequences=1):
        result = []
        for _ in range(num_return_sequences):
            current = input_ids[0].tolist()
            for _ in range(max_length - len(current)):
                with torch.no_grad():
                    logits = self(input_ids)['logits']
                    next_token = torch.argmax(logits[0, -1, :]).item()
                current.append(next_token)
                input_ids = torch.tensor([current])
            result.append(current)
        return torch.tensor(result)
    
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"hidden_size": 64, "vocab_size": 21}, f)
    
    @classmethod
    def from_pretrained(cls, path):
        model = cls()
        if os.path.exists(os.path.join(path, "pytorch_model.bin")):
            model.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin")))
        return model


class LoRAModule(nn.Module):
    """LoRA模块 - 低秩矩阵分解"""
    def __init__(self, input_dim, output_dim, rank=4):
        super(LoRAModule, self).__init__()
        self.A = nn.Linear(input_dim, rank, bias=False)
        self.B = nn.Linear(rank, output_dim, bias=False)

    def forward(self, x):
        return self.B(self.A(x))


class GPTWithLoRA(nn.Module):
    """将LoRA注入到模型的线性层中"""
    def __init__(self, base_model, rank=4):
        super(GPTWithLoRA, self).__init__()
        self.base_model = base_model
        self.lora_modules = nn.ModuleDict()

        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                input_dim, output_dim = module.in_features, module.out_features
                self.lora_modules[name] = LoRAModule(input_dim, output_dim, rank)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"hidden_size": 64, "vocab_size": 21}, f)


class MockDataset:
    """模拟数据集"""
    def __init__(self, data):
        self.data = data
    
    def map(self, function, batched=False):
        if batched:
            result = function(self.data)
            return MockDataset(result)
        else:
            result = [function(item) for item in self.data]
            return MockDataset(result)


class MockTrainingArguments:
    """模拟训练参数"""
    def __init__(self):
        self.output_dir = "./results"
        self.per_device_train_batch_size = 4
        self.num_train_epochs = 3
        self.logging_dir = "./logs"
        self.save_strategy = "epoch"
        self.logging_steps = 10


class MockTrainer:
    """模拟训练器"""
    def __init__(self, model, args, train_dataset, tokenizer):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
    
    def train(self):
        print("模拟训练过程...")
        # 模拟训练步骤
        for epoch in range(self.args.num_train_epochs):
            print(f"Epoch {epoch+1}/{self.args.num_train_epochs}")
        print("训练完成！")


def prepare_training_data():
    """准备训练数据"""
    data = {
        "text": [
            "今天是个好天气。",
            "我喜欢用GPT模型学习。",
            "微调技术让模型更加灵活。",
            "LoRA 技术是一种高效的微调方法。",
            "通过低秩矩阵分解减少参数量。"
        ]
    }
    return data


def preprocess_function(examples, tokenizer):
    """数据预处理"""
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)


def train_lora_model():
    """训练LoRA模型"""
    tokenizer = MockTokenizer()
    base_model = MockGPT2()
    lora_model = GPTWithLoRA(base_model, rank=4)

    print("LoRA模型结构:", lora_model)

    data = prepare_training_data()
    dataset = MockDataset(data)
    tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    training_args = MockTrainingArguments()

    trainer = MockTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("开始训练 LoRA 微调模型...")
    trainer.train()

    print("保存模型...")
    lora_model.save_pretrained("./lora_finetuned_model")
    
    # 保存分词器配置
    os.makedirs("./lora_finetuned_model", exist_ok=True)
    with open(os.path.join("./lora_finetuned_model", "tokenizer_config.json"), "w") as f:
        json.dump({"vocab_size": 21}, f)

    return "./lora_finetuned_model"


def inference_lora_model(model_path):
    """使用微调后的模型进行推理"""
    # 创建一个新的MockGPT2模型
    finetuned_model = MockGPT2()
    finetuned_tokenizer = MockTokenizer()

    # 直接使用基础模型进行推理（模拟微调效果）
    test_text = "GPT 模型的优点是"
    inputs = finetuned_tokenizer(test_text, return_tensors="pt")
    output = finetuned_model.generate(inputs["input_ids"], max_length=30)
    result = finetuned_tokenizer.decode(output[0], skip_special_tokens=True)

    print("输入文本:", test_text)
    print("生成结果:", result)
    return result


if __name__ == "__main__":
    model_path = train_lora_model()
    inference_lora_model(model_path)