# AI Agent 学术研究助手

一个基于 LangChain/LangGraph 的智能学术研究 Agent 系统，支持论文检索、混合向量搜索、上下文管理和实时对话流式输出。

> 本项目适合作为 **Agent 开发、RAG 架构、Context 管理** 等技术的学习参考。

***

## 项目概述

本项目实现了一个严谨的学术研究 AI 助手，具备以下核心能力：

### 能做什么

1. **智能论文检索**：当用户提出学术相关问题时，Agent 自动调用检索工具，从论文数据库中查找相关文献
2. **混合向量搜索**：结合稠密向量（语义理解）和稀疏向量（关键词匹配）进行混合检索，返回最相关的论文
3. **条件过滤搜索**：支持按作者、年份、标题关键词等条件精确筛选论文
4. **上下文管理**：通过 Redis 实现对话历史的存储、加载和智能压缩，避免上下文膨胀
5. **实时流式输出**：支持 Server-Sent Events 流式输出，实时显示 Agent 的思考过程
6. **工具调用可视化**：前端实时展示 Agent 调用了哪些工具、传入的参数、返回的结果

### 解决的问题

- **防止幻觉**：Agent 必须先调用检索工具才能回答学术问题，禁止凭空编造参考文献
- **上下文膨胀**：通过智能摘要压缩机制，保持上下文精简（每 3 轮对话压缩一次）
- **工具结果管理**：工具返回的论文数据只在当前轮有效，不跨轮传递，避免上下文爆炸
- **引用准确性**：严格规范引用格式，从工具返回结果中提取论文标题、作者、年份

***

!\[示例图]\(picture/1.png null)

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (React)                          │
│  - 实时流式对话     - 工具调用状态展示    - 上下文预览面板   │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP / SSE
┌────────────────────────────▼────────────────────────────────┐
│                    后端 API (FastAPI)                         │
│  - /api/chat/stream  流式对话接口                             │
│  - /api/context/{id} 上下文预览接口                           │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                 LangChain Agent (LangGraph)                   │
│  - create_agent        Agent 工厂函数                         │
│  - before_model        前置中间件：注入历史上下文              │
│  - after_model         后置中间件：记录工具调用                │
│  - ContextInjectMiddleware  模型调用拦截与消息替换             │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌───────────────┐  ┌─────────────────────┐
│  混合搜索工具   │  │  过滤搜索工具  │  │  Redis 上下文管理器  │
│ Hybrid Search │  │Filtered Search│  │  存储/压缩/摘要       │
└───────┬───────┘  └───────┬───────┘  └─────────────────────┘
        │                  │
        └────────┬─────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Milvus 向量数据库                          │
│  - 稠密向量 (dense_vector)     语义检索                       │
│  - 稀疏向量 (sparse_vector)    关键词检索                     │
│  - 元数据 (title, authors...)  标量过滤                       │
└─────────────────────────────────────────────────────────────┘
```

***

## 项目结构

```
Agent_System/
├── Agent/                      # Agent 核心模块
│   ├── agent.py              # Agent 工厂函数，集成中间件和工具
│   └── System_prompt.py      # 系统提示词配置
│
├── Back_end/                  # 后端服务
│   └── api_server.py         # FastAPI 服务，流式接口
│
├── Context/                   # 上下文管理模块
│   ├── manager.py            # Redis 上下文管理器（存储/压缩/摘要）
│   ├── middleware.py          # LangChain 中间件（注入上下文/记录工具）
│   ├── context_service.py     # 上下文预览服务
│   ├── schema.py              # AgentContext 数据结构
│   └── config.py             # 配置管理
│
├── Tool/                      # 工具模块
│   ├── Hybrid_Search_Tool.py  # 混合搜索工具（稠密+稀疏向量）
│   └── Filtered_Search_Tool.py # 条件过滤搜索工具
│
├── RAG/                       # RAG 数据处理模块
│   ├── Collection_schema.py   # Milvus 集合 Schema 定义
│   └── Loading_in_Milvus.py  # 数据导入脚本
│
├── Front_end/                 # 前端
│   ├── src/
│   │   ├── App.tsx           # 主组件，消息流处理，工具调用展示
│   │   └── App.css           # 样式（工具调用状态、上下文面板等）
│   └── package.json
│
├── start_app.py              # 一键启动脚本
└── README.md
```

***

## 核心模块详解

### 1. Agent 模块 ([agent.py](Agent/agent.py))

使用 LangChain 的 `create_agent` 工厂函数创建 Agent，集成了：

- **模型**：OpenAI 兼容接口（temperature=0.0 保证确定性）
- **工具**：`hybrid_search` 和 `filtered_search`
- **系统提示词**：规范 Agent 的行为（何时调用工具、如何引用）
- **中间件**：注入历史上下文、记录工具调用

```python
agent = create_agent(
    model=model,
    tools=[hybrid_search, filtered_search],
    system_prompt=SYSTEM_PROMPT,
    context_schema=AgentContext,
    middleware=[
        inject_context,          # 前置：注入历史消息
        ContextInjectMiddleware(), # 拦截模型调用
        log_after_model,         # 后置：记录工具调用
    ],
)
```

### 2. 上下文管理 ([Context/](Context/))

**RedisContextManager** 负责对话历史的持久化和压缩：

| 方法                        | 功能                      |
| ------------------------- | ----------------------- |
| `load_messages()`         | 从 Redis 加载历史消息          |
| `load_context_messages()` | 加载上下文（摘要 + 历史消息）        |
| `append_messages()`       | 保存新消息，触发压缩              |
| `compact()`               | 调用 LLM 生成摘要，保留最近 12 条消息 |
| `summarize()`             | 用 GPT 生成简洁摘要            |

**关键设计**：

- **触发阈值**：每 3 轮对话触发一次压缩
- **摘要策略**：工具结果只保留前 100 字符，避免膨胀
- **消息过滤**：加载历史时不包含 `tool` 类型消息，只在当前轮有效

### 3. LangChain 中间件 ([middleware.py](Context/middleware.py))

使用 LangChain 的 `before_model` 和 `after_model` 装饰器实现中间件：

**前置中间件** **`inject_context`**：

1. 从 Redis 加载历史上下文
2. 过滤消息类型（只保留 human/ai/system）
3. 限制消息数量（最多 3 条）
4. 设置线程上下文

**后置中间件** **`log_after_model`**：

1. 提取 AI 消息中的 tool\_calls
2. 记录工具调用日志到 Redis

### 4. 工具模块 ([Tool/](Tool/))

**混合搜索工具**：

```python
hybrid_search(query: str, limit: int = 20)
```

- 生成稠密向量（语义）和稀疏向量（关键词）
- 使用 RRF (Reciprocal Rank Fusion) 融合两种检索结果
- 返回格式：`[序号] 论文标题\n作者: xxx | 年份: xxx\n摘要: ...\n正文片段: ...`

**过滤搜索工具**：

```python
filtered_search(
    author: str = None,
    year: int = None,
    year_min: int = None,
    year_max: int = None,
    title_keyword: str = None,
    summary_keyword: str = None,
    limit: int = 5
)
```

- 支持多条件组合过滤
- 使用 Milvus 的标量过滤语法

### 5. 前端流式处理 ([App.tsx](Front_end/src/App.tsx))

使用 Fetch API + ReadableStream 消费 SSE：

```typescript
// 处理不同类型的事件
if (data.type === 'text') {
  // 文本片段，追加到消息内容
} else if (data.type === 'tool_call') {
  // 工具调用开始，显示工具名称和参数
} else if (data.type === 'tool_result') {
  // 工具返回结果，显示结果预览
} else if (data.type === 'done') {
  // 对话完成，保存线程 ID
}
```

***

## 学习要点

### 1. LangChain Agent 开发

- **create\_agent**：LangChain 的 Agent 工厂函数，统一封装模型、工具、提示词
- **context\_schema**：定义 Agent 运行时状态的数据结构
- **middleware**：在模型调用前后插入自定义逻辑

### 2. LangGraph 流式处理

```python
result = agent.astream(
    {"messages": [current_message], "thread_id": thread_id},
    stream_mode="messages",
    version="v2",
)

async for chunk in result:
    if chunk.get("type") == "messages":
        token, metadata = chunk.get("data")
        node = metadata.get("langgraph_node", "")
        # node == "model": 模型输出
        # node == "tools": 工具执行结果
```

### 3. RAG 与向量检索

- **混合搜索**：稠密向量捕捉语义，稀疏向量捕捉关键词
- **RRF 融合**：reciprocal rank fusion 平衡两种检索方式
- **Milvus Schema**：设计支持混合检索的 Schema（dense\_vector + sparse\_vector + 元数据）

### 4. 上下文压缩策略

- **摘要压缩**：用 LLM 将旧对话压缩成关键信息
- **选择性保留**：系统消息和摘要永久保留，用户/AI 消息限制数量
- **工具结果隔离**：工具返回的大数据只在当前轮有效

### 5. 系统提示词工程

- **规则优先**：明确何时必须调用工具
- **禁止事项**：防止幻觉（不伪造引用）
- **格式规范**：规范引用格式和输出风格

***

## 快速开始

### 环境依赖

- Python 3.10+
- Node.js 18+
- Milvus 向量数据库（localhost:19530）
- Redis（localhost:6379）
- Embedding 服务（稠密向量 + 稀疏向量）
- LLM 服务（OpenAI 兼容接口）

### 安装

```bash
# 克隆项目
git clone https://github.com/Zhaiyuan-Ji/Agent_System.git
cd Agent_System

# 安装 Python 依赖
pip install langchain langchain-openai pymilvus redis fastapi uvicorn

# 安装前端依赖
cd Front_end
npm install
cd ..

# 配置环境变量（修改 Context/config.py 中的 URL 和 API Key）
```

### 启动

```bash
python start_app.py
```

访问 `http://127.0.0.1:5173` 即可使用。

***

## 示例对话

**用户**：帮我找几篇 5G 毫米波传播相关的论文

**Agent**（自动调用工具）：

```
🔧 调用了 1 个工具
✅ hybrid_search
参数: {"query": "5G毫米波传播", "limit": 20}
结果预览: [1] 5G通信网络中信号干扰抑制技术的研究与实现
作者: 杨丽斌 | 年份: 2025
摘要: 本文剖析5G网络同频、邻道、毫米波大气散射三类干扰特征...
```

**Agent**（结合工具结果回答）：

```
根据检索结果，我找到了以下相关论文...[详细回答]...

[本次回答参考如下信息:
5G通信网络中信号干扰抑制技术的研究与实现 | 杨丽斌 | 2025
]
```

***

## 技术栈

| 层级       | 技术                           |
| -------- | ---------------------------- |
| Agent 框架 | LangChain, LangGraph         |
| 模型       | OpenAI 兼容接口 (GPT-4o / GPT-5) |
| 向量数据库    | Milvus                       |
| 缓存/存储    | Redis                        |
| 后端框架     | FastAPI + Uvicorn            |
| 前端框架     | React + TypeScript + Vite    |
| 样式       | 自定义 CSS                      |

***

## License

MIT License
