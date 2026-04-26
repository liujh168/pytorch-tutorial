"""
示例: P-Tuning实现
对应文档: docs/LLM Code/Char01.txt - 1.4.2节
功能: 通过Hugging Face的transformers和PyTorch实现基于GPT-2模型的文本分类任务
"""

# 模拟P-Tuning实现演示（无需网络连接）
import torch
import torch.nn as nn
import json
import os


class MockTokenizer:
    """模拟分词器"""
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "今": 2, "天": 3, "的": 4, "天": 5, "气": 6, "很": 7, "好": 8, "。": 9, "我": 10, "很": 11, "讨": 12, "厌": 13, "下": 14, "雨": 15, "阳": 16, "光": 17, "明": 18, "媚": 19, "让": 20, "开": 21, "心": 22}
    
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


class MockBERT(nn.Module):
    """模拟BERT模型"""
    def __init__(self, num_labels=2):
        super(MockBERT, self).__init__()
        self.config = type('Config', (), {'hidden_size': 64, 'vocab_size': 23})()
        self.embedding = nn.Embedding(23, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        logits = self.fc(x[:, -1, :])
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    def get_input_embeddings(self):
        return self.embedding
    
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"hidden_size": 64, "vocab_size": 23, "num_labels": 2}, f)
    
    @classmethod
    def from_pretrained(cls, path, num_labels=2):
        model = cls(num_labels=num_labels)
        if os.path.exists(os.path.join(path, "pytorch_model.bin")):
            model.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin")))
        return model


class PTuningPrompt(nn.Module):
    """P-Tuning模块 - 可训练的虚拟标记嵌入"""
    def __init__(self, num_virtual_tokens, embedding_dim):
        super(PTuningPrompt, self).__init__()
        self.virtual_embeddings = nn.Embedding(num_virtual_tokens, embedding_dim)
        self.num_virtual_tokens = num_virtual_tokens

    def forward(self, batch_size):
        return self.virtual_embeddings(torch.arange(self.num_virtual_tokens).expand(batch_size, -1).to(torch.long))


class PTuningModel(nn.Module):
    """带P-Tuning的模型"""
    def __init__(self, base_model, num_virtual_tokens=10):
        super(PTuningModel, self).__init__()
        self.base_model = base_model
        self.embedding_dim = base_model.config.hidden_size
        self.prompt = PTuningPrompt(num_virtual_tokens, self.embedding_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.size(0)
        prompt_embeddings = self.prompt(batch_size)
        input_embeddings = self.base_model.get_input_embeddings()(input_ids)
        inputs_with_prompt = torch.cat([prompt_embeddings, input_embeddings], dim=1)
        extended_attention_mask = torch.cat([
            torch.ones((batch_size, self.prompt.num_virtual_tokens)).to(attention_mask.device),
            attention_mask
        ], dim=1)
        outputs = self.base_model(
            input_ids=input_ids,  # 简化实现，直接使用input_ids
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"hidden_size": 64, "vocab_size": 23, "num_labels": 2}, f)


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
    
    def train_test_split(self, test_size=0.2):
        split_idx = int(len(self.data["text"]) * (1 - test_size))
        train_data = {
            "text": self.data["text"][:split_idx],
            "label": self.data["label"][:split_idx]
        }
        test_data = {
            "text": self.data["text"][split_idx:],
            "label": self.data["label"][split_idx:]
        }
        return {"train": MockDataset(train_data), "test": MockDataset(test_data)}


class MockTrainingArguments:
    """模拟训练参数"""
    def __init__(self):
        self.output_dir = "./results"
        self.evaluation_strategy = "epoch"
        self.learning_rate = 5e-5
        self.per_device_train_batch_size = 4
        self.num_train_epochs = 3
        self.logging_dir = "./logs"
        self.save_strategy = "epoch"
        self.logging_steps = 10


class MockTrainer:
    """模拟训练器"""
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
    
    def train(self):
        print("模拟训练过程...")
        # 模拟训练步骤
        for epoch in range(self.args.num_train_epochs):
            print(f"Epoch {epoch+1}/{self.args.num_train_epochs}")
        print("训练完成！")


def prepare_data():
    """准备训练数据"""
    data = {
        "text": [
            "今天的天气很好。",
            "我很讨厌下雨。",
            "阳光明媚让我开心。",
            "大风让我心情烦躁。",
            "微笑是积极的表现。"
        ],
        "label": [1, 0, 1, 0, 1]
    }
    return data


def preprocess_function(examples, tokenizer):
    """数据预处理"""
    result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
    # 保留原始字段
    result["text"] = examples["text"]
    result["label"] = examples["label"]
    return result


def train_p_tuning_model():
    """训练P-Tuning模型"""
    tokenizer = MockTokenizer()
    base_model = MockBERT(num_labels=2)
    p_tuning_model = PTuningModel(base_model)

    print("P-Tuning模型结构:", p_tuning_model)

    data = prepare_data()
    dataset = MockDataset(data)
    tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    train_dataset = tokenized_dataset.train_test_split(test_size=0.2)["train"]
    eval_dataset = tokenized_dataset.train_test_split(test_size=0.2)["test"]

    training_args = MockTrainingArguments()

    trainer = MockTrainer(
        model=p_tuning_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    print("开始训练带 P-Tuning 的模型...")
    trainer.train()

    print("保存模型...")
    p_tuning_model.save_pretrained("./p_tuning_finetuned_model")
    
    # 保存分词器配置
    os.makedirs("./p_tuning_finetuned_model", exist_ok=True)
    with open(os.path.join("./p_tuning_finetuned_model", "tokenizer_config.json"), "w") as f:
        json.dump({"vocab_size": 23}, f)

    return "./p_tuning_finetuned_model"


def inference_p_tuning_model(model_path):
    """使用微调后的模型进行推理"""
    # 创建一个新的MockBERT模型
    finetuned_model = MockBERT(num_labels=2)
    finetuned_tokenizer = MockTokenizer()

    test_text = "阳光让我感到快乐。"
    inputs = finetuned_tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    outputs = finetuned_model(**inputs)
    predicted_label = outputs["logits"].argmax(dim=1).item()
    label_map = {0: "消极", 1: "积极"}

    print("输入文本:", test_text)
    print("预测情感:", label_map[predicted_label])
    return predicted_label


if __name__ == "__main__":
    model_path = train_p_tuning_model()
    inference_p_tuning_model(model_path)