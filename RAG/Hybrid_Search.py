from pymilvus import AnnSearchRequest
import math
from openai import OpenAI
from pymilvus import (
    MilvusClient, DataType, Function, FunctionType
)

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
collection_name = "Agent_System"
# ====================== 基础配置（复用你的） ======================
DENSE_CLIENT_CONFIG = {"base_url": "http://localhost:54331/v1", "api_key": "token-abc123"}
SPARSE_CLIENT_CONFIG = {"base_url": "http://localhost:54332/v1", "api_key": "token-abc123"}
EMBED_MODEL = "text-embedding-3-small"


# 初始化客户端（只需初始化一次）
def init_openai_client(config):
    return OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=30.0,
        max_retries=2
    )


dense_client = init_openai_client(DENSE_CLIENT_CONFIG)
sparse_client = init_openai_client(SPARSE_CLIENT_CONFIG)


# ====================== 核心函数（仅这两个） ======================
def get_query_dense_vector(query: str) -> list:
    """
    输入查询文本，输出稠密向量（适配Milvus混合检索）
    :param query: 检索文本（如"white headphones, quiet and comfortable"）
    :return: 2560维稠密向量列表（float类型）
    """
    try:
        # 调用OpenAI接口生成向量
        response = dense_client.embeddings.create(input=query, model=EMBED_MODEL)
        dense_vector = response.data[0].embedding

        # 确保维度为2560（与你入库的dense_vector字段一致）
        dense_vector = [float(x) for x in dense_vector[:2560]]
        if len(dense_vector) < 2560:
            dense_vector += [0.0] * (2560 - len(dense_vector))

        return dense_vector
    except Exception as e:
        raise Exception(f"生成稠密向量失败：{str(e)}")


def get_query_sparse_vector(query: str) -> dict:
    """
    输入查询文本，输出稀疏向量（适配Milvus混合检索）
    :param query: 检索文本（如"white headphones, quiet and comfortable"）
    :return: 稀疏向量字典 {索引: 值}（匹配SPARSE_FLOAT_VECTOR格式）
    """
    try:
        # 调用OpenAI接口生成原始稀疏向量
        response = sparse_client.embeddings.create(input=query, model=EMBED_MODEL)
        raw_sparse = response.data[0].embedding

        # 转换为Milvus要求的{idx: val}格式
        sparse_dict = {}
        if isinstance(raw_sparse, dict) and "indices" in raw_sparse and "values" in raw_sparse:
            for idx, val in zip(raw_sparse["indices"], raw_sparse["values"]):
                try:
                    int_idx = int(idx)
                    float_val = float(val)
                    if int_idx >= 0 and not (math.isnan(float_val) or math.isinf(float_val)):
                        sparse_dict[int_idx] = float_val
                except:
                    continue

        # 兜底：避免空向量
        if not sparse_dict:
            sparse_dict = {1001: 0.07489, 1002: 0.02476}

        return sparse_dict
    except Exception as e:
        raise Exception(f"生成稀疏向量失败：{str(e)}")


query_text = "5G 毫米波大气散射干扰的动态衰减，主要受什么的影响？"

query_dense_vector = get_query_dense_vector(query_text)
query_sparse_vector = get_query_sparse_vector(query_text)

search_param_1 = {
    "data": [query_dense_vector],
    "anns_field": "dense_vector",
    "param": {"ef": 10},
    "limit": 10
}

request_1 = AnnSearchRequest(**search_param_1)

search_param_2 = {
    "data": [query_sparse_vector],
    "anns_field": "sparse_vector",
    "param": {"drop_ratio_search": 0.2},
    "limit": 10
}
request_2 = AnnSearchRequest(**search_param_2)


reqs = [request_1, request_2]

ranker = Function(
    name="rrf",
    input_field_names=[], # Must be an empty list
    function_type=FunctionType.RERANK,
    params={
        "reranker": "rrf",
        "k": 60  # Optional
    }
)


res = client.hybrid_search(
    collection_name="Agent_System",
    reqs=reqs,
    ranker=ranker,
    limit=3,
    output_fields=["title", "authors", "publish_year", "content"]
)
# 优化结果展示（清晰打印具体内容）
print(f"🔍 查询文本：{query_text}\n")
print(f"📊 混合检索 Top3 结果：\n")
for idx, hits in enumerate(res):
    if hits:  # 确保有结果
        for hit_idx, hit in enumerate(hits, 1):
            print(f"========== 结果 {hit_idx} ==========")
            print(f"ID: {hit['ID']}")
            print(f"相似度距离: {hit['distance']:.6f}")
            print(f"标题: {hit['entity'].get('title', '无')}")
            print(f"作者: {hit['entity'].get('authors', '无')}")
            print(f"发布年份: {hit['entity'].get('publish_year', '无')}")
            print(f"内容片段: {hit['entity'].get('content', '无')[:1000]}..." if len(hit['entity'].get('content', ''))>300 else f"内容片段: {hit['entity'].get('content', '无')}")
            print("-" * 80)

