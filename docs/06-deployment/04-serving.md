# 模型服务化 (Model Serving)

## 概述 Overview

将训练好的模型部署为 API 服务，使其可以被其他应用调用。本章介绍 FastAPI 和 TorchServe 两种方案。

## 代码实现 Implementation

### 1. FastAPI 基础服务

```python
# server.py
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List

# 定义模型
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(x)

# 初始化
app = FastAPI(title="PyTorch Model API")

# 加载模型
model = SimpleClassifier()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# 请求/响应模型
class PredictionRequest(BaseModel):
    data: List[float]  # 784 维向量

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: List[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # 预处理
        input_data = torch.tensor(request.data).float().unsqueeze(0)

        if input_data.shape[1] != 784:
            raise HTTPException(400, "Input must be 784 dimensions")

        # 推理
        with torch.inference_mode():
            logits = model(input_data)
            probs = torch.softmax(logits, dim=-1)

        prediction = probs.argmax().item()
        probabilities = probs.squeeze().tolist()

        return PredictionResponse(
            prediction=prediction,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 运行: uvicorn server:app --host 0.0.0.0 --port 8000
```

### 2. FastAPI 批处理服务

```python
# batch_server.py
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from typing import List
import time

app = FastAPI()

# 全局变量
model = None
request_queue = asyncio.Queue()
BATCH_SIZE = 16
MAX_WAIT_TIME = 0.05  # 50ms

class BatchRequest(BaseModel):
    id: str
    data: List[float]

class BatchResponse(BaseModel):
    id: str
    prediction: int

async def batch_processor():
    """后台批处理任务"""
    while True:
        batch = []
        futures = []

        # 收集请求
        try:
            while len(batch) < BATCH_SIZE:
                item = await asyncio.wait_for(
                    request_queue.get(),
                    timeout=MAX_WAIT_TIME
                )
                batch.append(item)
        except asyncio.TimeoutError:
            pass

        if not batch:
            continue

        # 批量推理
        inputs = torch.stack([item["tensor"] for item in batch])

        with torch.inference_mode():
            outputs = model(inputs)
            predictions = outputs.argmax(dim=-1).tolist()

        # 返回结果
        for item, pred in zip(batch, predictions):
            item["future"].set_result(pred)

@app.on_event("startup")
async def startup():
    global model
    model = nn.Linear(784, 10)
    model.eval()

    # 启动批处理器
    asyncio.create_task(batch_processor())

@app.post("/predict")
async def predict(request: BatchRequest):
    tensor = torch.tensor(request.data).float()
    future = asyncio.get_event_loop().create_future()

    await request_queue.put({
        "tensor": tensor,
        "future": future
    })

    prediction = await future

    return BatchResponse(id=request.id, prediction=prediction)
```

### 3. TorchServe 部署

```python
# model_handler.py
import torch
import torch.nn as nn
from ts.torch_handler.base_handler import BaseHandler
import json

class CustomHandler(BaseHandler):
    """TorchServe 自定义 Handler"""

    def initialize(self, context):
        """初始化模型"""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # 加载模型
        self.model = self._load_model(model_dir)
        self.model.eval()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def _load_model(self, model_dir):
        """加载模型权重"""
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        model.load_state_dict(
            torch.load(f"{model_dir}/model.pth", map_location="cpu")
        )
        return model

    def preprocess(self, data):
        """预处理输入"""
        inputs = []
        for row in data:
            input_data = row.get("data") or row.get("body")
            if isinstance(input_data, (bytes, bytearray)):
                input_data = json.loads(input_data.decode("utf-8"))
            inputs.append(torch.tensor(input_data["input"]).float())

        return torch.stack(inputs).to(self.device)

    def inference(self, data):
        """推理"""
        with torch.inference_mode():
            outputs = self.model(data)
            return outputs

    def postprocess(self, inference_output):
        """后处理输出"""
        probs = torch.softmax(inference_output, dim=-1)
        predictions = probs.argmax(dim=-1)

        results = []
        for pred, prob in zip(predictions, probs):
            results.append({
                "prediction": pred.item(),
                "probabilities": prob.tolist()
            })

        return results

# 打包模型:
# torch-model-archiver --model-name my_model \
#     --version 1.0 \
#     --model-file model.py \
#     --serialized-file model.pth \
#     --handler model_handler.py \
#     --export-path model_store

# 启动 TorchServe:
# torchserve --start --model-store model_store --models my_model=my_model.mar
```

### 4. gRPC 服务

```python
# grpc_server.py
import grpc
from concurrent import futures
import torch
import torch.nn as nn

# 假设已有 proto 文件生成的代码
# protoc --python_out=. --grpc_python_out=. service.proto

"""
# service.proto
syntax = "proto3";

service ModelService {
    rpc Predict(PredictRequest) returns (PredictResponse);
}

message PredictRequest {
    repeated float data = 1;
}

message PredictResponse {
    int32 prediction = 1;
    repeated float probabilities = 2;
}
"""

# import service_pb2
# import service_pb2_grpc

class ModelServicer:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def Predict(self, request, context):
        input_data = torch.tensor(list(request.data)).float().unsqueeze(0)

        with torch.inference_mode():
            logits = self.model(input_data)
            probs = torch.softmax(logits, dim=-1)

        prediction = probs.argmax().item()
        probabilities = probs.squeeze().tolist()

        # return service_pb2.PredictResponse(
        #     prediction=prediction,
        #     probabilities=probabilities
        # )
        return {"prediction": prediction, "probabilities": probabilities}

def serve():
    model = nn.Linear(784, 10)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )

    # service_pb2_grpc.add_ModelServiceServicer_to_server(
    #     ModelServicer(model), server
    # )

    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    server.wait_for_termination()

# if __name__ == "__main__":
#     serve()
```

### 5. Docker 部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码和模型
COPY server.py .
COPY model.pth .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  model-server:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.pth
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 6. 负载均衡与扩展

```python
# nginx.conf 示例
"""
upstream model_servers {
    least_conn;
    server model-server-1:8000 weight=1;
    server model-server-2:8000 weight=1;
    server model-server-3:8000 weight=1;
}

server {
    listen 80;

    location /predict {
        proxy_pass http://model_servers;
        proxy_connect_timeout 10s;
        proxy_read_timeout 30s;
    }

    location /health {
        proxy_pass http://model_servers;
    }
}
"""

# Kubernetes Deployment
"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: model-server
        image: my-model-server:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "4Gi"
            cpu: "2"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model-server
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
```

## 服务框架对比

| 特性 | FastAPI | TorchServe | Triton |
|------|---------|------------|--------|
| 易用性 | ⭐⭐⭐ 最简单 | ⭐⭐ 中等 | ⭐ 复杂 |
| 性能 | 良好 | 优秀 | 最佳 |
| 批处理 | 需自实现 | ✅ 内置 | ✅ 内置 |
| 模型管理 | 需自实现 | ✅ 内置 | ✅ 内置 |
| 多框架 | PyTorch | PyTorch | 多框架 |

## 下一步 Next

[下一章：边缘设备部署 →](./05-edge-deployment.md)
