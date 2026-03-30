import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pymilvus import MilvusClient
from openai import OpenAI

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

DENSE_CLIENT_CONFIG = {"base_url": "http://localhost:54331/v1", "api_key": "token-abc123"}
EMBED_MODEL = "text-embedding-3-small"

dense_client = OpenAI(base_url=DENSE_CLIENT_CONFIG["base_url"], api_key=DENSE_CLIENT_CONFIG["api_key"], timeout=30.0, max_retries=2)

def get_query_dense_vector(query: str) -> list:
    response = dense_client.embeddings.create(input=query, model=EMBED_MODEL)
    dense_vector = response.data[0].embedding
    dense_vector = [float(x) for x in dense_vector[:2560]]
    if len(dense_vector) < 2560:
        dense_vector += [0.0] * (2560 - len(dense_vector))
    return dense_vector

def search(
    query: str,
    author: str = None,
    year: int = None,
    year_range: tuple[int, int] = None,
    title_keyword: str = None,
    summary_keyword: str = None,
    limit: int = 5
) -> list:
    query_vector = get_query_dense_vector(query)

    filters = []

    if author:
        filters.append(f'authors == "{author}"')

    if year is not None:
        filters.append(f"publish_year == {year}")

    if year_range:
        filters.append(f"publish_year >= {year_range[0]}")
        filters.append(f"publish_year <= {year_range[1]}")

    if title_keyword:
        filters.append(f'title like "%{title_keyword}%"')

    if summary_keyword:
        filters.append(f'summary like "%{summary_keyword}%"')

    filter_expr = " and ".join(filters) if filters else None

    results = client.search(
        collection_name="Agent_System",
        data=[query_vector],
        anns_field="dense_vector",
        filter=filter_expr,
        limit=limit,
        output_fields=["title", "authors", "publish_year", "summary"]
    )

    return results


if __name__ == "__main__":
    query = "5G 毫米波大气散射干扰"

    print("=" * 60)
    print("测试1: 向量搜索")
    print("=" * 60)
    results = search(query, limit=3)
    for hits in results:
        for hit in hits:
            print(f"作者: {hit['entity']['authors']}, 年份: {hit['entity']['publish_year']}")

    print("\n" + "=" * 60)
    print("测试2: INVERTED索引 - 精确匹配作者")
    print("=" * 60)
    results = search(query, author="程润梦", limit=3)
    for hits in results:
        for hit in hits:
            print(f"作者: {hit['entity']['authors']}")

    print("\n" + "=" * 60)
    print("测试3: AUTOINDEX索引 - 精确匹配年份")
    print("=" * 60)
    results = search(query, year=2024, limit=3)
    for hits in results:
        for hit in hits:
            print(f"年份: {hit['entity']['publish_year']}")

    print("\n" + "=" * 60)
    print("测试4: AUTOINDEX索引 - 范围查询")
    print("=" * 60)
    results = search(query, year_range=(2023, 2025), limit=3)
    for hits in results:
        for hit in hits:
            print(f"年份: {hit['entity']['publish_year']}")

    print("\n" + "=" * 60)
    print("测试5: NGRAM索引 - 模糊搜索标题")
    print("=" * 60)
    results = search(query, title_keyword="信号", limit=3)
    for hits in results:
        for hit in hits:
            print(f"标题: {hit['entity']['title'][:40]}...")

    print("\n" + "=" * 60)
    print("测试6: NGRAM索引 - 模糊搜索摘要")
    print("=" * 60)
    results = search(query, summary_keyword="毫米波", limit=3)
    for hits in results:
        for hit in hits:
            print(f"摘要: {hit['entity']['summary'][:40]}...")

    print("\n" + "=" * 60)
    print("测试7: 组合过滤")
    print("=" * 60)
    results = search(query, author="程润梦", year_range=(2023, 2025), limit=3)
    if results and results[0]:
        for hit in results[0]:
            print(f"作者: {hit['entity']['authors']}, 年份: {hit['entity']['publish_year']}")
    else:
        print("无结果")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
