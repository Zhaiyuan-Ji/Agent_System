# Agent System

基于 LangChain Agent 的 AI 对话系统，支持流式响应、上下文管理和 Redis 缓存。

!\[截图1]\(picture/1.png null)

## 功能特性

- **流式响应**：逐字流式输出，实时显示 AI 回复
- **上下文管理**：基于 Redis 的上下文存储，支持多轮对话
- **上下文压缩**：当消息超过 20 条时自动触发摘要压缩
- **LangChain 中间件**：使用官方中间件系统处理上下文注入
- **多消息类型支持**：支持 human、ai、system、tool 消息类型

## 项目结构

```
Agent_System/
├── Agent/                    # Agent 核心模块
│   ├── agent.py             # Agent 创建和初始化
│   └── langchain_middleware.md  # LangChain 中间件文档
├── Back_end/                 # 后端服务
│   └── api_server.py         # FastAPI 服务接口
├── Context/                  # 上下文管理模块
│   ├── config.py            # 配置文件
│   ├── context_service.py   # 上下文服务
│   ├── manager.py           # Redis 上下文管理器
│   ├── middleware.py        # LangChain 中间件
│   ├── redis_client.py      # Redis 客户端
│   └── schema.py            # 数据模型
├── Front_end/               # 前端界面
│   └── src/
│       ├── App.tsx         # React 主组件
│       └── App.css         # 样式文件
├── MCP/                     # MCP 服务（本地保留）
├── RAG/                     # RAG 相关代码（本地保留）
├── Data/                    # 数据文件（本地保留）
└── .gitignore
```

## 技术栈

- **后端**：Python、FastAPI、LangChain、Redis
- **前端**：React、TypeScript、Vite
- **AI**：OpenAI Compatible API（GPT-5.1）

## 环境要求

- Python 3.10+
- Node.js 18+
- Redis Server

## 配置

在 `Context/config.py` 中修改：

```python
OPENAI_BASE_URL = "http://127.0.0.1:54329/v1"  # API 地址
OPENAI_API_KEY = "token-abc123"                 # API Key
OPENAI_MODEL = "gpt-5.1"                        # 模型名称
```

## 运行

### 后端

```bash
cd Agent_System
python -m uvicorn Back_end.api_server:app --host 127.0.0.1 --port 8000
```

### 前端

```bash
cd Front_end
npm install
npm run dev
```

访问 <http://127.0.0.1:3000>

## API 接口

| 接口                               | 方法     | 说明        |
| -------------------------------- | ------ | --------- |
| `/api/health`                    | GET    | 健康检查      |
| `/api/chat`                      | POST   | 普通对话（非流式） |
| `/api/chat/stream`               | POST   | 流式对话      |
| `/api/context/{thread_id}`       | GET    | 获取上下文预览   |
| `/api/conversations/{thread_id}` | DELETE | 清空会话      |

##

