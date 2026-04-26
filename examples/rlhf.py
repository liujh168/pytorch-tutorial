"""
示例: RLHF人类反馈强化学习
对应文档: docs/LLM Code/Char01.txt - 1.4.3节
功能: 基于Hugging Face的transformers和PyTorch实现RLHF
"""

# 模拟RLHF实现演示（无需网络连接）
import torch
import torch.nn as nn
import json
import os


class MockTokenizer:
    """模拟分词器"""
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "请": 2, "用": 3, "简": 4, "单": 5, "的": 6, "语": 7, "言": 8, "解": 9, "释": 10, "什": 11, "么": 12, "是": 13, "机": 14, "器": 15, "学": 16, "习": 17, "。": 18}
    
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
        self.config = type('Config', (), {'hidden_size': 64, 'vocab_size': 19})()
        self.embedding = nn.Embedding(19, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 19)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        logits = self.fc(x)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, 19), labels.view(-1))
            return {"loss": loss, "logits": logits, "last_hidden_state": x}
        return {"logits": logits, "last_hidden_state": x}
    
    def generate(self, input_ids, max_length=50, num_return_sequences=1):
        result = []
        for _ in range(num_return_sequences):
            current = input_ids[0].tolist()
            for _ in range(max_length - len(current)):
                with torch.no_grad():
                    logits = self(input_ids)["logits"]
                    next_token = torch.argmax(logits[0, -1, :]).item()
                current.append(next_token)
                input_ids = torch.tensor([current])
            result.append(current)
        return torch.tensor(result)
    
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"hidden_size": 64, "vocab_size": 19}, f)
    
    @classmethod
    def from_pretrained(cls, path):
        model = cls()
        if os.path.exists(os.path.join(path, "pytorch_model.bin")):
            model.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin")))
        return model


class RewardModel(nn.Module):
    """奖励模型 - 评估生成文本的质量"""
    def __init__(self, base_model):
        super(RewardModel, self).__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs["last_hidden_state"]
        reward = self.reward_head(hidden_states[:, -1, :])
        return reward


class PolicyModel(nn.Module):
    """基于PPO的策略模型"""
    def __init__(self, base_model):
        super(PolicyModel, self).__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs["logits"]
        loss = outputs.get("loss", torch.tensor(0.0))
        return logits, loss


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
    
    def __iter__(self):
        # 简单模拟，只返回第一个样本
        yield {
            "input_ids": self.data.get("input_ids", [[1]*64])[0],
            "attention_mask": self.data.get("attention_mask", [[1]*64])[0]
        }


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
    def __init__(self, model, args, train_dataset):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
    
    def train(self):
        print("模拟训练过程...")
        # 模拟训练步骤
        for epoch in range(self.args.num_train_epochs):
            print(f"Epoch {epoch+1}/{self.args.num_train_epochs}")
        print("训练完成！")


def prepare_dataset():
    """准备数据集"""
    data = {
        "prompt": [
            "请用简单的语言解释什么是机器学习。",
            "为什么要减少塑料污染？",
            "如何保持良好的工作习惯？"
        ],
        "response": [
            "机器学习是一种让计算机从数据中学习并做出预测的方法。",
            "塑料污染会破坏环境和生态系统，因此需要减少使用。",
            "保持良好的工作习惯包括规划任务和设定优先级。"
        ]
    }
    return data


def preprocess_function(examples, tokenizer):
    """数据预处理"""
    input_texts = [prompt + response for prompt, response in zip(examples["prompt"], examples["response"])]
    inputs = tokenizer(input_texts, padding="max_length", truncation=True, max_length=64)
    # 保留原始字段
    inputs["prompt"] = examples["prompt"]
    inputs["response"] = examples["response"]
    return inputs


def train_reward_model():
    """训练奖励模型"""
    tokenizer = MockTokenizer()
    base_model = MockGPT2()
    reward_model = RewardModel(base_model)

    print("奖励模型加载完成:", reward_model)

    data = prepare_dataset()
    dataset = MockDataset(data)
    tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    training_args = MockTrainingArguments()

    trainer_reward = MockTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    print("开始微调奖励模型...")
    trainer_reward.train()
    print("奖励模型微调完成！")

    return reward_model, tokenizer


def train_policy_model(reward_model, tokenizer):
    """策略优化 - 伪PPO简化实现"""
    base_model = MockGPT2()
    policy_model = PolicyModel(base_model)
    print("策略模型加载完成:", policy_model)

    data = prepare_dataset()
    dataset = MockDataset(data)
    tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    print("开始策略模型优化...")
    for epoch in range(3):
        for sample in tokenized_dataset:
            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0)
            attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0)

            logits, _ = policy_model(input_ids=input_ids, attention_mask=attention_mask)
            action = torch.argmax(logits, dim=-1)

            reward = reward_model(input_ids=input_ids, attention_mask=attention_mask)

            loss = -reward.mean()
            # 模拟反向传播
            print(f"  样本奖励: {reward.item():.4f}, 损失: {loss.item():.4f}")

        print(f"策略优化完成第 {epoch+1} 轮")

    policy_model.base_model.save_pretrained("./policy_model")
    
    # 保存分词器配置
    os.makedirs("./policy_model", exist_ok=True)
    with open(os.path.join("./policy_model", "tokenizer_config.json"), "w") as f:
        json.dump({"vocab_size": 19}, f)

    return "./policy_model"


def inference_policy_model(model_path):
    """推理测试"""
    finetuned_model = MockGPT2.from_pretrained(model_path)
    finetuned_tokenizer = MockTokenizer()

    test_prompt = "如何养成健康的饮食习惯？"
    inputs = finetuned_tokenizer(test_prompt, return_tensors="pt")
    outputs = finetuned_model.generate(inputs["input_ids"], max_length=40)
    result = finetuned_tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("输入问题:", test_prompt)
    print("生成回答:", result)
    return result


if __name__ == "__main__":
    reward_model, tokenizer = train_reward_model()
    policy_model_path = train_policy_model(reward_model, tokenizer)
    inference_policy_model(policy_model_path)