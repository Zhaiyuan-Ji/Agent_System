"""
API Server

提供 FastAPI 服务接口。
"""

from __future__ import annotations

import uuid
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from Agent.agent import agent
from Context import (
    AgentContext,
    get_agent_settings,
    get_prompt_context,
    context_manager,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    message: str
    mode: str


app = FastAPI(title="Agent System API", version="4.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_thread_id() -> str:
    return f"thread_{uuid.uuid4().hex[:12]}"


@app.get("/api/health")
async def health() -> dict[str, str]:
    settings = get_agent_settings()
    return {
        "status": "ok",
        "mode": settings["mode"],
        "model": settings["model"],
        "base_url": settings["base_url"],
    }


@app.get("/api/context/{thread_id}")
async def context_preview(thread_id: str, draft: str = "") -> dict:
    return get_prompt_context(thread_id=thread_id, draft_message=draft)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    from langchain_core.messages import HumanMessage

    thread_id = request.thread_id or create_thread_id()
    current_message = HumanMessage(content=request.message.strip())

    async def event_stream():
        full_content = ""
        is_assistant_responding = False

        try:
            result = agent.astream(
                {"messages": [current_message], "thread_id": thread_id},
                stream_mode="messages",
                version="v2",
            )

            async for chunk in result:
                if chunk.get("type") == "messages":
                    token, metadata = chunk.get("data")
                    node = metadata.get("langgraph_node", "")

                    if node == "tools":
                        tool_content = getattr(token, 'content', '') or ""
                        if tool_content:
                            tool_name = getattr(token, 'name', 'unknown')
                            tool_preview = tool_content[:200] + "..." if len(tool_content) > 200 else tool_content
                            yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'preview': tool_preview})}\n\n"
                        continue

                    if node == "model":
                        if hasattr(token, 'tool_calls') and token.tool_calls:
                            for tc in token.tool_calls:
                                if isinstance(tc, dict):
                                    tool_name = tc.get('name', 'unknown')
                                    tool_args = tc.get('args', {})
                                else:
                                    tool_name = getattr(tc, 'name', 'unknown')
                                    tool_args = getattr(tc, 'args', {})
                                if tool_name:
                                    args_str = json.dumps(tool_args, ensure_ascii=False)[:500]
                                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'args': args_str})}\n\n"

                        if hasattr(token, 'content_blocks') and token.content_blocks:
                            for block in token.content_blocks:
                                block_data = block if isinstance(block, dict) else {"type": getattr(block, 'type', ''), "text": getattr(block, 'text', '')}
                                if block_data.get("type") == "text":
                                    text = block_data.get("text") or ""
                                    if text:
                                        is_assistant_responding = True
                                        full_content += text
                                        yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        if full_content and is_assistant_responding:
            from langchain_core.messages import AIMessage
            await context_manager.append_messages(
                thread_id,
                [current_message, AIMessage(content=full_content)],
            )

        yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id, 'full_content': full_content})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    from langchain_core.messages import HumanMessage, AIMessage

    thread_id = request.thread_id or create_thread_id()
    current_message = HumanMessage(content=request.message.strip())

    result = await agent.ainvoke(
        {"messages": [current_message], "thread_id": thread_id},
        context=AgentContext(thread_id=thread_id),
    )

    messages = result.get("messages", [])
    ai_messages = [m for m in messages if getattr(m, "type", "") == "ai"]
    reply_message = ai_messages[-1] if ai_messages else None
    reply = reply_message.content if reply_message else "模型没有返回内容。"

    await context_manager.append_messages(
        thread_id,
        [current_message, AIMessage(content=reply)],
    )

    settings = get_agent_settings()
    return ChatResponse(
        thread_id=thread_id,
        message=reply,
        mode=settings["mode"],
    )


@app.delete("/api/conversations/{thread_id}")
async def clear_conversation(thread_id: str) -> dict[str, str]:
    await context_manager.clear_thread(thread_id)
    return {"status": "cleared", "thread_id": thread_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("Back_end.api_server:app", host="127.0.0.1", port=8000, reload=False)
