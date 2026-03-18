from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP
from typing import Dict, Any
from pymilvus import AnnSearchRequest
import math
from openai import OpenAI
from pymilvus import (
    MilvusClient, Function, FunctionType
)

# 初始化MCP实例
mcp = FastMCP("milvus_hybrid_search_server")

# ====================== 全局配置与客户端初始化（生产环境复用） ======================
# Milvus客户端配置
milvus_client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
collection_name = "Agent_System"

# OpenAI向量客户端配置
DENSE_CLIENT_CONFIG = {"base_url": "http://localhost:54331/v1", "api_key": "token-abc123"}
SPARSE_CLIENT_CONFIG = {"base_url": "http://localhost:54332/v1", "api_key": "token-abc123"}
EMBED_MODEL = "text-embedding-3-small"

# 初始化OpenAI客户端（仅初始化一次，生产环境复用）
def init_openai_client(config):
    return OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=30.0,
        max_retries=2
    )

dense_client = init_openai_client(DENSE_CLIENT_CONFIG)
sparse_client = init_openai_client(SPARSE_CLIENT_CONFIG)

# ====================== MCP核心工具函数 ======================
@mcp.tool()
def milvus_hybrid_search(query: str) -> Dict[str, Any]:
    """
    Milvus混合检索工具：基于稠密+稀疏向量实现混合检索，返回Top3相关结果，数据库中包含各种电磁、信号相关的文献
    :param query: 检索文本（如"5G 毫米波大气散射干扰的动态衰减，主要受什么的影响？"）
    :return: 包含检索结果的字典，结构为{"query": 查询文本, "results": 检索结果列表}
    """
    # 生成稠密向量（固定2560维）
    response = dense_client.embeddings.create(input=query, model=EMBED_MODEL)
    dense_vector = response.data[0].embedding
    dense_vector = [float(x) for x in dense_vector[:2560]]
    if len(dense_vector) < 2560:
        dense_vector += [0.0] * (2560 - len(dense_vector))

    # 生成稀疏向量（转换为Milvus要求的{idx: val}格式）
    sparse_response = sparse_client.embeddings.create(input=query, model=EMBED_MODEL)
    raw_sparse = sparse_response.data[0].embedding
    sparse_dict = {}
    if isinstance(raw_sparse, dict) and "indices" in raw_sparse and "values" in raw_sparse:
        for idx, val in zip(raw_sparse["indices"], raw_sparse["values"]):
            int_idx = int(idx)
            float_val = float(val)
            if int_idx >= 0 and not (math.isnan(float_val) or math.isinf(float_val)):
                sparse_dict[int_idx] = float_val
    # 兜底空向量
    if not sparse_dict:
        sparse_dict = {1001: 0.07489, 1002: 0.02476}

    # 构建检索请求
    search_param_1 = {
        "data": [dense_vector],
        "anns_field": "dense_vector",
        "param": {"ef": 10},
        "limit": 50
    }
    request_1 = AnnSearchRequest(**search_param_1)

    search_param_2 = {
        "data": [sparse_dict],
        "anns_field": "sparse_vector",
        "param": {"drop_ratio_search": 0.2},
        "limit": 50
    }
    request_2 = AnnSearchRequest(**search_param_2)

    # 配置RRF重排器
    ranker = Function(
        name="rrf",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={"reranker": "rrf", "k": 60}
    )

    # 执行混合检索
    search_result = milvus_client.hybrid_search(
        collection_name=collection_name,
        reqs=[request_1, request_2],
        ranker=ranker,
        limit=10,
        output_fields=["title", "authors", "publish_year", "content"]
    )

    # 格式化返回结果（生产环境易解析）
    formatted_results = []
    for hits in search_result:
        if hits:
            for hit in hits:
                formatted_results.append({
                    "id": hit["ID"],
                    "distance": round(hit["distance"], 6),
                    "title": hit["entity"].get("title", "无"),
                    "authors": hit["entity"].get("authors", "无"),
                    "publish_year": hit["entity"].get("publish_year", "无"),
                    "content": hit["entity"].get("content", "无")
                })

    return {
        "query": query,
        "results": formatted_results
    }

# ====================== MCP Prompt模板 ======================
@mcp.prompt()
def prompt():
    """Milvus混合检索助手：利用稠密+稀疏向量混合检索技术回答用户的信息查询问题"""
    return """
    你是一个智能检索助手，擅长调用Milvus混合检索工具获取精准的信息。
    当用户提出信息查询类问题时，调用milvus_hybrid_search工具，并将检索结果清晰地呈现给用户。
    """

if __name__ == "__main__":
    mcp.run(transport="stdio")