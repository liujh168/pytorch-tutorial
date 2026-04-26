"""
示例: 大模型相关的Python开发技术
对应文档: docs/LLM Code/Char01.txt - 1.3.1节
功能: 展示如何加载预训练模型、数据处理、模型推理、微调和保存加载
"""

# 模拟大模型开发技术演示（无需网络连接）
import torch
import torch.nn as nn
import json
import os


class MockTokenizer:
    """模拟分词器"""
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "大": 2, "语": 3, "言": 4, "模": 5, "型": 6, "是": 7, "自": 8, "然": 9, "处": 10, "理": 11, "领": 12, "域": 13, "的": 14, "重": 15, "要": 16, "成": 17, "果": 18, "。": 19}
    
    def __call__(self, text, return_tensors="pt", padding=True, truncation=True):
        tokens = [self.vocab.get(char, 1) for char in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([tokens])}
        return {"input_ids": [tokens]}
    
    def decode(self, tokens, skip_special_tokens=True):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return "".join([reverse_vocab.get(token, "[UNK]") for token in tokens if token not in [0] or not skip_special_tokens])


class MockModel(nn.Module):
    """模拟语言模型"""
    def __init__(self):
        super(MockModel, self).__init__()
        self.embedding = nn.Embedding(20, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 20)
    
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        logits = self.fc(x)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, 20), labels.view(-1))
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


class MockClassificationModel(nn.Module):
    """模拟分类模型"""
    def __init__(self, num_labels=2):
        super(MockClassificationModel, self).__init__()
        self.embedding = nn.Embedding(20, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, num_labels)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        logits = self.fc(x[:, -1, :])
        return {"logits": logits}
    
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"num_labels": 2}, f)
    
    @classmethod
    def from_pretrained(cls, path):
        model = cls()
        model.load_state_dict(torch.load(os.path.join(path, "pytorch_model.bin")))
        return model


def load_pretrained_model():
    """加载预训练模型与分词器"""
    tokenizer = MockTokenizer()
    model = MockModel()
    return tokenizer, model


def data_processing():
    """数据处理示例"""
    tokenizer = MockTokenizer()
    text = "大语言模型是自然语言处理领域的重要成果。"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return inputs


def model_inference(inputs):
    """模型推理"""
    tokenizer = MockTokenizer()
    model = MockModel()
    output = model.generate(inputs['input_ids'], max_length=25, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def fine_tune_model():
    """微调大语言模型"""
    train_data = [
        {"text": "我很高兴今天的天气很好", "label": 1},
        {"text": "我觉得今天很糟糕", "label": 0}
    ]
    return train_data


def save_and_load_model():
    """保存与加载模型"""
    model_classification = MockClassificationModel(num_labels=2)
    tokenizer = MockTokenizer()
    model_classification.save_pretrained("./finetuned_model")
    
    finetuned_model = MockClassificationModel.from_pretrained("./finetuned_model")
    finetuned_tokenizer = MockTokenizer()
    return finetuned_model, finetuned_tokenizer


def test_inference(finetuned_model, finetuned_tokenizer):
    """模型推理测试"""
    test_sentence = "今天真是一个好天气"
    inputs = finetuned_tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = finetuned_model(**inputs)
    predicted_label = outputs['logits'].argmax(dim=1).item()
    return predicted_label


if __name__ == "__main__":
    print("=== 加载预训练模型 ===")
    tokenizer, model = load_pretrained_model()
    print("模型加载成功")

    print("\n=== 数据处理 ===")
    inputs = data_processing()
    print("输入数据:", inputs.keys())
    print("输入ID:", inputs['input_ids'].tolist())

    print("\n=== 模型推理 ===")
    generated_text = model_inference(inputs)
    print("生成文本:", generated_text)

    print("\n=== 保存和加载模型 ===")
    finetuned_model, finetuned_tokenizer = save_and_load_model()
    print("模型保存和加载成功")

    print("\n=== 推理测试 ===")
    label_map = {0: "消极", 1: "积极"}
    predicted_label = test_inference(finetuned_model, finetuned_tokenizer)
    print("预测情感:", label_map[predicted_label])