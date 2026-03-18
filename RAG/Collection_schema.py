from pymilvus import MilvusClient, DataType

milvus_client = MilvusClient(
    uri="http://localhost:19530",
)

# 关键：先删除旧集合（必须！否则新Schema不生效）
COLLECTION_NAME = "Agent_System"
if milvus_client.has_collection(collection_name=COLLECTION_NAME):
    print(f"⚠️ 检测到旧集合 {COLLECTION_NAME}，正在删除...")
    milvus_client.drop_collection(collection_name=COLLECTION_NAME)
    print(f"✅ 旧集合删除成功")



# 1. 创建空 Schema
schema = MilvusClient.create_schema(
    enable_dynamic_field=False  # 关闭动态字段，保证数据结构规范（学术场景推荐）
)

# 2. 添加主键字段（唯一标识每篇文章，推荐自动生成ID）
schema.add_field(
    field_name="ID",
    datatype=DataType.INT64,
    is_primary=True,
    auto_id=True,  # Milvus 自动分配ID，避免手动维护冲突
    description="学术文章唯一主键ID"
)

# 3. 稠密向量字段（用于语义检索，适配BERT/LLM类模型）
# 维度建议：根据你的嵌入模型调整（如BERT-base=768，Sentence-BERT=384/768，LLaMA=1024/4096）
schema.add_field(
    field_name="dense_vector",
    datatype=DataType.FLOAT_VECTOR,  # 32位浮点，通用且精度足够
    dim=2560,  # 可根据实际模型调整（核心！需和嵌入输出维度一致）
    description="文章全文/摘要的稠密向量嵌入（语义检索）"
)

# 4. 稀疏向量字段（用于关键词/TF-IDF类检索，混合检索核心）
schema.add_field(
    field_name="sparse_vector",
    datatype=DataType.SPARSE_FLOAT_VECTOR,  # 稀疏浮点向量，适配TF-IDF/关键词权重
    description="文章关键词/TF-IDF的稀疏向量嵌入（关键词检索）"
)

# 5. c：文章名
schema.add_field(
    field_name="title",
    datatype=DataType.VARCHAR,
    max_length=256,  # 学术文章标题通常<200字符，预留冗余
    description="文章标题"
)

# 6. 元数据字段：作者名（支持多作者，用逗号分隔）
schema.add_field(
    field_name="authors",
    datatype=DataType.VARCHAR,
    max_length=512,  # 多作者（如"张三,李四,王五"）足够存储
    description="文章作者，多作者用英文逗号分隔"
)

# 7. 元数据字段：发表年份
schema.add_field(
    field_name="publish_year",
    datatype=DataType.INT32,  # 整型足够（范围-2147483648~2147483647）
    description="文章发表年份（如2026）"
)

# 8. 元数据字段：文章总结/摘要
schema.add_field(
    field_name="summary",
    datatype=DataType.VARCHAR,
    max_length=6000,  # 摘要通常<1500字符，预留冗余
    description="文章核心总结/摘要"
)

# 9. 元数据字段：文章正文（Milvus的VARCHAR最大支持65535字符）
schema.add_field(
    field_name="content",
    datatype=DataType.VARCHAR,
    max_length=65535,  # 单篇学术文章正文通常<5万字，65535字符足够（若超量可拆分存储）
    description="文章完整正文内容"
)

# 可选：验证Schema（调试用）
print("Schema 创建完成，字段列表：")
for field in schema.fields:
    print(f"- {field.name}: {field.dtype} (主键: {field.is_primary})")

########################################     索引配置     ########################################

index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="dense_vector",
    index_type="HNSW",
    index_name="vector_index", # Name of the index to create
    metric_type="COSINE",  # 语义检索优先用余弦相似度，也可选L2/IP
    params={"M": 16, "efConstruction": 70}  # 平衡检索速度和精度的通用参数
)
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX", # Type of the index to create
    index_name="sparse_index", # Name of the index to create
    metric_type="IP",  # 稀疏向量优先用内积相似度
    params={"inverted_index_algo": "DAAT_MAXSCORE"}, # Algorithm used for building and querying the index
)
index_params.add_index(
    field_name="title",
    index_type="NGRAM",
    index_name="title_ngram_index",  # 唯一索引名
    min_gram=2,  # Minimum substring length (e.g., 2-gram: "st")
    max_gram=20  # Maximum substring length (e.g., 3-gram: "sta")
)

index_params.add_index(
    field_name="authors",
    index_type="NGRAM",
    index_name="authors_ngram_index",  # 唯一索引名
    min_gram=2,  # Minimum substring length (e.g., 2-gram: "st")
    max_gram=20  # Maximum substring length (e.g., 3-gram: "sta")
)

index_params.add_index(
    field_name="summary",
    index_type="NGRAM",
    index_name="summary_ngram_index",  # 唯一索引名
    min_gram=2,  # Minimum substring length (e.g., 2-gram: "st")
    max_gram=20  # Maximum substring length (e.g., 3-gram: "sta")
)



milvus_client.create_collection(
    collection_name="Agent_System",
    schema=schema,
    index_params=index_params,
    consistency_level="Strong"
)


# 验证创建结果
print("\n✅ 集合创建完成，验证信息：")
# 1. 验证Schema
coll_schema = milvus_client.describe_collection(COLLECTION_NAME)
for field in coll_schema["fields"]:
    if field["name"] == "sparse_vector":
        print(f"   sparse_vector字段类型码：{field['type']}")
        print(f"   是否为SPARSE_FLOAT_VECTOR：{field['type'] == DataType.SPARSE_FLOAT_VECTOR.value}")

# 2. 验证索引
indexes = milvus_client.list_indexes(COLLECTION_NAME)
print(f"   集合索引列表：{indexes}")

print("\n🎉 集合完整重建成功！")