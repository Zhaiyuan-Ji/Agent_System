from pymilvus import AnnSearchRequest, MilvusClient, Function, FunctionType
import math
from openai import OpenAI

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
collection_name = "Agent_System"

# 基础配置（复用）
DENSE_CLIENT_CONFIG = {"base_url": "http://localhost:54331/v1", "api_key": "token-abc123"}
SPARSE_CLIENT_CONFIG = {"base_url": "http://localhost:54332/v1", "api_key": "token-abc123"}
EMBED_MODEL = "text-embedding-3-small"

# 初始化客户端/向量生成函数（复用，略）
def init_openai_client(config):
    return OpenAI(base_url=config["base_url"], api_key=config["api_key"], timeout=30.0, max_retries=2)
dense_client = init_openai_client(DENSE_CLIENT_CONFIG)
sparse_client = init_openai_client(SPARSE_CLIENT_CONFIG)

def get_query_dense_vector(query: str) -> list:
    try:
        res = dense_client.embeddings.create(input=query, model=EMBED_MODEL)
        vec = res.data[0].embedding[:2560]
        return vec + [0.0]*(2560 - len(vec))
    except Exception as e:
        raise Exception(f"稠密向量生成失败：{e}")

def get_query_sparse_vector(query: str) -> dict:
    try:
        res = sparse_client.embeddings.create(input=query, model=EMBED_MODEL)
        raw = res.data[0].embedding
        sparse = {}
        if isinstance(raw, dict) and "indices" in raw and "values" in raw:
            for idx, val in zip(raw["indices"], raw["values"]):
                try:
                    sparse[int(idx)] = float(val)
                except:
                    continue
        return sparse if sparse else {1001: 0.07489, 1002: 0.02476}
    except Exception as e:
        raise Exception(f"稀疏向量生成失败：{e}")

# ====================== 核心优化：扩大向量检索范围 ======================
# 1. 加载集合
client.load_collection(collection_name)

# 2. 生成向量（5G相关查询）
query_text = "5G 毫米波大气散射干扰的动态衰减，主要受什么的影响？"
dense_vec = get_query_dense_vector(query_text)
sparse_vec = get_query_sparse_vector(query_text)

# 3. 构建混合检索请求（关键：limit从10改为100，扩大候选范围）
req1 = AnnSearchRequest(data=[dense_vec], anns_field="dense_vector", param={"ef": 20}, limit=100)
req2 = AnnSearchRequest(data=[sparse_vec], anns_field="sparse_vector", param={"drop_ratio_search": 0.2}, limit=100)
ranker = Function(name="rrf", input_field_names=[], function_type=FunctionType.RERANK, params={"reranker": "rrf", "k": 60})

# 4. 过滤条件
filter_condition = 'authors like "%程润梦%"'

# 5. 执行混合检索（外层limit仍为3，仅返回符合条件的3条）
res = client.hybrid_search(
    collection_name=collection_name,
    reqs=[req1, req2],
    ranker=ranker,
    limit=3,
    filter=filter_condition,
    output_fields=["title", "authors", "publish_year", "content"],
    consistency_level="Strong"
)

# ====================== 结果遍历 ======================
print(f"🔍 查询文本：{query_text}\n")
print(f"📊 混合检索 Top3 结果（filter：{filter_condition}）：\n")

if res and res[0]:
    for hit_idx, hit in enumerate(res[0], 1):
        print(f"========== 结果 {hit_idx} ==========")
        print(f"ID: {hit['ID']}")
        print(f"相似度距离: {hit['distance']:.6f}")
        print(f"标题: {hit['entity'].get('title', '无')}")
        print(f"作者: {hit['entity'].get('authors', '无')}")
        print(f"发布年份: {hit['entity'].get('publish_year', '无')}")
        content = hit['entity'].get('content', '无')
        print(f"内容片段: {content[:1000]}..." if len(content) > 300 else f"内容片段: {content}")
        print("-" * 80)
else:
    print("❌ 未找到符合过滤条件的结果！")

# 验证索引
print("\n集合索引列表：", client.list_indexes(collection_name))