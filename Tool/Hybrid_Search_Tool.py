from pymilvus import AnnSearchRequest, MilvusClient, Function, FunctionType
from openai import OpenAI
import os
import math

os.environ['PYTHONIOENCODING'] = 'utf-8'

from pydantic import BaseModel, Field
from langchain_core.tools import tool

MILVUS_CLIENT = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
DENSE_CONFIG = {"base_url": "http://localhost:54331/v1", "api_key": "token-abc123"}
SPARSE_CONFIG = {"base_url": "http://localhost:54332/v1", "api_key": "token-abc123"}
EMBED_MODEL = "text-embedding-3-small"

DENSE_CLIENT = OpenAI(base_url=DENSE_CONFIG["base_url"], api_key=DENSE_CONFIG["api_key"], timeout=30.0, max_retries=2)
SPARSE_CLIENT = OpenAI(base_url=SPARSE_CONFIG["base_url"], api_key=SPARSE_CONFIG["api_key"], timeout=30.0, max_retries=2)


def get_dense_vector(text: str) -> list:
    response = DENSE_CLIENT.embeddings.create(input=text, model=EMBED_MODEL)
    vector = response.data[0].embedding
    vector = [float(x) for x in vector[:2560]]
    if len(vector) < 2560:
        vector += [0.0] * (2560 - len(vector))
    return vector


def get_sparse_vector(text: str) -> dict:
    response = SPARSE_CLIENT.embeddings.create(input=text, model=EMBED_MODEL)
    raw = response.data[0].embedding
    sparse_dict = {}
    if isinstance(raw, dict) and "indices" in raw and "values" in raw:
        for idx, val in zip(raw["indices"], raw["values"]):
            try:
                int_idx = int(idx)
                float_val = float(val)
                if int_idx >= 0 and not (math.isnan(float_val) or math.isinf(float_val)):
                    sparse_dict[int_idx] = float_val
            except:
                continue
    if not sparse_dict:
        sparse_dict = {1001: 0.07489, 1002: 0.02476}
    return sparse_dict


class HybridSearchInput(BaseModel):
    query: str = Field(description="用户查询文本")
    limit: int = Field(default=20, description="返回结果数量上限")


@tool(args_schema=HybridSearchInput)
def hybrid_search(query: str, limit: int = 20) -> str:
    """混合搜索工具，通过语义理解和关键词匹配从学术论文数据库中检索最相关的论文。

    使用场景：
    - 用户进行普通的搜索
    - 用户没有明确指定过滤条件

    工作原理：
    - 结合稠密向量（语义理解）和稀疏向量（关键词匹配）进行混合检索
    - 返回与用户查询语义最相关的论文

    注意：如果用户明确要求按作者、年份等条件过滤，请使用过滤搜索工具而非此工具。

    调用本工具后返回内容格式如下：
    [序号] 论文标题
    作者: 作者姓名 | 年份: 年份
    摘要: 论文摘要...
    正文片段: 从论文正文中摘录的相关内容片段...

    重要：
    - 每条结果以 [数字] 开头，如 [1]、[2] 等
    - 论文标题后面紧接着的是 "作者:" 和 "年份:" 字段
    - 正文片段位于 "正文片段:" 之后，可能包含论文中的小节标题（如 "1.1节标题"），但这不等于论文标题
    - 引用时请使用 [序号] 后的论文标题，而不是正文片段中的章节标题
    """
    dense_vector = get_dense_vector(query)
    sparse_vector = get_sparse_vector(query)

    request_1 = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param={"ef": 10},
        limit=limit
    )
    request_2 = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param={"drop_ratio_search": 0.2},
        limit=limit
    )

    ranker = Function(
        name="rrf",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={"reranker": "rrf", "k": 60}
    )

    results = MILVUS_CLIENT.hybrid_search(
        collection_name="Agent_System",
        reqs=[request_1, request_2],
        ranker=ranker,
        limit=10,
        output_fields=["title", "authors", "publish_year", "summary", "content"]
    )

    if not results or not results[0]:
        return "未找到相关结果"

    output = []
    for hits in results:
        for i, hit in enumerate(hits[:limit], 1):
            entity = hit["entity"]
            title = entity.get("title", "无标题")
            authors = entity.get("authors", "未知作者")
            year = entity.get("publish_year", "未知年份")
            summary = entity.get("summary", "无摘要")
            content = entity.get("content", "")[:500] if entity.get("content") else "无正文"
            output.append(
                f"[{i}] {title}\n"
                f"作者: {authors} | 年份: {year}\n"
                f"摘要: {summary}\n"
                f"正文片段: {content}..."
            )

    return "\n\n".join(output)
