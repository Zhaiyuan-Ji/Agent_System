# rag_backend_only_sparse.py
"""
仅返回稀疏向量的OpenAI兼容后端
- embedding字段返回Milvus兼容的稀疏向量格式
- 原有OpenAI接口格式不变，适配RAG无感知调用
"""
import os
import time
from collections import defaultdict

# 置顶设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from FlagEmbedding import BGEM3FlagModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import logging

logging.set_verbosity_error()

# 配置项
LOCAL_MODEL_PATH = "/mnt/users/ylu/jzy/model/BAAI-bge-m3/"
SERVICE_PORT = 54332
USE_8BIT_QUANT = False
USE_FP16 = True
VERBOSE_LOG = True


# 模型加载
def load_model_with_optimization(model_path):
    start_time = time.time()
    assert torch.cuda.is_available(), "❌ 无可用GPU"
    assert os.path.exists(model_path), f"❌ 模型路径不存在：{model_path}"

    torch.cuda.set_device(0)
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)

    model = BGEM3FlagModel(
        model_path,
        use_fp16=USE_FP16,
        device=f"cuda:{current_device}",
        load_in_8bit=USE_8BIT_QUANT,
        cache_dir=None,
        local_files_only=True,
        trust_remote_code=False
    )
    model.encode(["预热"], return_sparse=True)

    load_duration = time.time() - start_time
    if VERBOSE_LOG:
        memory_used = torch.cuda.memory_allocated(current_device) / 1024 / 1024
        print(f"\n✅ 模型加载完成（仅返回稀疏向量）")
        print(f"   - 绑定物理卡4：{device_name}")
        print(f"   - 模型路径：{model_path}")
        print(f"   - 加载耗时：{load_duration:.2f} 秒")
        print(f"   - GPU显存占用：{memory_used:.2f} MB")
    return model


model = load_model_with_optimization(LOCAL_MODEL_PATH)


# 解析旧版本稀疏向量
def parse_old_version_sparse(sparse_data):
    if isinstance(sparse_data, defaultdict):
        sorted_items = sorted(sparse_data.items(), key=lambda x: x[0])
        indices = [int(item[0]) for item in sorted_items]
        values = [float(item[1]) for item in sorted_items]
        return indices, values
    elif hasattr(sparse_data, 'indices') and hasattr(sparse_data, 'values'):
        return sparse_data.indices.tolist(), sparse_data.values.tolist()
    else:
        raise ValueError(f"不支持的稀疏向量格式：{type(sparse_data)}")


# OpenAI兼容请求体
class OpenAIEmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "text-embedding-3-small"
    encoding_format: str = "float"


# FastAPI服务
app = FastAPI(title="仅稀疏向量服务（OpenAI兼容）", version="1.0")


# 核心：仅返回稀疏向量，embedding字段为Milvus兼容格式
@app.post("/v1/embeddings")
def openai_compatible_embeddings(req: OpenAIEmbeddingRequest):
    try:
        # 处理输入
        if isinstance(req.input, str):
            texts = [req.input.strip()]
        else:
            texts = [t.strip() for t in req.input if t.strip()]

        if not texts:
            raise ValueError("输入内容不能为空")

        # 仅生成稀疏向量（关闭稠密向量生成，提速）
        start_time = time.time()
        embeddings = model.encode(
            texts,
            return_sparse=True,
            return_dense=False  # 关闭稠密向量，仅生成稀疏向量
        )

        # 构造响应：embedding字段=稀疏向量（Milvus兼容格式）
        response_data = []
        for idx, text in enumerate(texts):
            # 解析稀疏向量
            if 'lexical_weights' in embeddings:
                sparse_indices, sparse_values = parse_old_version_sparse(embeddings['lexical_weights'][idx])
            elif 'lexical_sparse' in embeddings:
                sparse_indices, sparse_values = parse_old_version_sparse(embeddings['lexical_sparse'][idx])
            else:
                sparse_indices, sparse_values = [], []

            # 核心：embedding字段返回Milvus兼容的稀疏向量格式
            # （保持OpenAI接口格式，仅替换embedding值为稀疏向量）
            item = {
                "embedding": {
                    "indices": sparse_indices,
                    "values": sparse_values
                },
                "index": idx,
                "object": "embedding"
            }
            response_data.append(item)

        return {
            "data": response_data,
            "model": req.model,
            "object": "list",
            "usage": {
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts),
                "completion_tokens": 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"稀疏向量生成失败：{str(e)}")


# 启动服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, workers=1, log_level="error")