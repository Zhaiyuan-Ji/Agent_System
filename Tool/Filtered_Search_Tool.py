from pymilvus import MilvusClient
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.tools import tool

MILVUS_CLIENT = MilvusClient(uri="http://localhost:19530", token="root:Milvus")


class FilteredSearchInput(BaseModel):
    author: Optional[str] = Field(default=None, description="作者姓名，精确匹配（如：程润梦、杨丽斌）")
    year: Optional[int] = Field(default=None, description="发表年份，精确匹配（如：2024）")
    year_min: Optional[int] = Field(default=None, description="最小年份，范围查询（如：2023）")
    year_max: Optional[int] = Field(default=None, description="最大年份，范围查询（如：2025）")
    title_keyword: Optional[str] = Field(default=None, description="标题关键词，模糊匹配")
    summary_keyword: Optional[str] = Field(default=None, description="摘要关键词，模糊匹配")
    limit: int = Field(default=5, description="返回结果数量上限")


@tool(args_schema=FilteredSearchInput)
def filtered_search(
    author: Optional[str] = None,
    year: Optional[int] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    title_keyword: Optional[str] = None,
    summary_keyword: Optional[str] = None,
    limit: int = 5
) -> str:
    """过滤搜索工具，通过指定条件从学术论文数据库中筛选结果。

    使用场景：
    - 用户明确要求按作者查找论文（如"找程润梦的论文"）
    - 用户要求按年份/年份范围查找论文（如"2024年的论文"）
    - 用户要求在标题或摘要中包含特定关键词（如"标题包含信号的论文"）
    - 用户提出组合过滤条件

    注意：如果用户只是进行普通的主题搜索，不要调用此工具，应使用混合搜索工具。

    可用过滤字段：
    - author: 作者姓名，支持精确匹配
    - year: 发表年份，支持精确匹配
    - year_min / year_max: 年份范围查询
    - title_keyword: 标题关键词，支持模糊匹配
    - summary_keyword: 摘要关键词，支持模糊匹配

    必须提供至少一个过滤条件才能执行查询。

    调用本工具后返回内容格式如下：
    [序号] 论文标题
    作者: 作者姓名 | 年份: 年份
    摘要: 论文摘要...
    正文片段: 从论文正文中摘录的相关内容片段...

    重要：
    - 每条结果以 [数字] 开头，如 [1]、[2] 等
    - 论文标题后面紧接着的是 "作者:" 和 "年份:" 字段
    - 正文片段位于 "正文片段:" 之后，可能包含论文中的小节标题，但这不是论文标题
    - 引用时请使用 [序号] 后的论文标题，而不是正文片段中的章节标题
    """
    filters = []

    if author:
        filters.append(f'authors == "{author}"')

    if year is not None:
        filters.append(f"publish_year == {year}")

    if year_min is not None:
        filters.append(f"publish_year >= {year_min}")

    if year_max is not None:
        filters.append(f"publish_year <= {year_max}")

    if title_keyword:
        filters.append(f'title like "%{title_keyword}%"')

    if summary_keyword:
        filters.append(f'summary like "%{summary_keyword}%"')

    filter_expr = " and ".join(filters) if filters else None

    if not filter_expr:
        return "请至少指定一个过滤条件"

    results = MILVUS_CLIENT.query(
        collection_name="Agent_System",
        filter=filter_expr,
        limit=limit,
        output_fields=["title", "authors", "publish_year", "summary", "content"]
    )

    if not results:
        return "未找到符合条件的结果"

    output = []
    for i, entity in enumerate(results[:limit], 1):
        title = entity.get("title", "无标题")
        authors = entity.get("authors", "未知作者")
        year_val = entity.get("publish_year", "未知年份")
        summary = entity.get("summary", "无摘要")
        content = entity.get("content", "")[:500] if entity.get("content") else "无正文"
        output.append(
            f"[{i}] {title}\n"
            f"作者: {authors} | 年份: {year_val}\n"
            f"摘要: {summary}\n"
            f"正文片段: {content}..."
        )

    return "\n\n".join(output)
