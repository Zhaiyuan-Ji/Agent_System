from __future__ import annotations

import os
import uuid
from collections import defaultdict
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

MessageRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    message: str
    mode: str


SYSTEM_PROMPT = """
你是一个简洁、可靠、偏产品经理风格的中文 AI 助手。
回答时优先给出直接结论，再补充必要说明。
如果用户的问题不明确，先基于现有信息做合理假设，不要反复追问。
""".strip()

# 用内存保存会话，方便先把平台跑起来。
conversation_store: dict[str, list[ChatMessage]] = defaultdict(list)

# 默认走演示模式，等你准备好模型服务后再切到 openai。
CHAT_MODE = os.getenv("CHAT_MODE", "demo").lower()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "demo-key")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = AsyncOpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

app = FastAPI(title="Agent System API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_thread_id() -> str:
    return f"thread_{uuid.uuid4().hex[:12]}"


def build_demo_reply(user_message: str, history: list[ChatMessage]) -> str:
    recent_topics = [item.content for item in history if item.role == "user"][-3:]
    topic_summary = " / ".join(recent_topics)

    return (
        "当前是演示模式，我已经把这套聊天平台的基础链路接通了。\n\n"
        f"你的问题是：`{user_message}`\n\n"
        "如果你继续接入真实模型服务，后端会把同一个会话里的上下文一起发给模型。\n\n"
        "你现在可以继续这样用它：\n"
        "1. 直接追问同一个主题，验证多轮对话。\n"
        "2. 点击新会话，验证线程切换。\n"
        "3. 后续把 `CHAT_MODE` 改成 `openai`，并配置模型地址。\n\n"
        f"最近会话上下文：{topic_summary}"
    )


async def generate_reply(thread_id: str) -> str:
    history = conversation_store[thread_id]

    if CHAT_MODE != "openai":
        return build_demo_reply(history[-1].content, history)

    # 只带最近几轮上下文，避免原型阶段消息无限增长。
    model_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    model_messages.extend(item.model_dump() for item in history[-12:])

    completion = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=model_messages,
    )

    return completion.choices[0].message.content or "模型没有返回内容。"


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "mode": CHAT_MODE,
        "model": OPENAI_MODEL,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or create_thread_id()
    message = request.message.strip()
    history = conversation_store[thread_id]

    history.append(ChatMessage(role="user", content=message))
    reply = await generate_reply(thread_id)
    history.append(ChatMessage(role="assistant", content=reply))

    return ChatResponse(
        thread_id=thread_id,
        message=reply,
        mode=CHAT_MODE,
    )


@app.delete("/api/conversations/{thread_id}")
async def clear_conversation(thread_id: str) -> dict[str, str]:
    conversation_store.pop(thread_id, None)
    return {"status": "cleared", "thread_id": thread_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("Back_end.api_server:app", host="127.0.0.1", port=8000, reload=False)
