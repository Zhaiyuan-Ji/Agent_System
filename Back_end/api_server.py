"""
FastAPI service for the Agent Workbench.
"""

from __future__ import annotations

import json
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from Agent.agent import agent
from Context import (
    AgentContext,
    EventRecord,
    ToolEvidence,
    context_manager,
    get_agent_settings,
    get_prompt_context,
    refresh_runtime_artifacts,
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


def sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class ReasoningTextFilter:
    """Remove MiniMax inline <think>...</think> text from streamed answer chunks."""

    open_tag = "<think>"
    close_tag = "</think>"

    def __init__(self) -> None:
        self.in_reasoning = False
        self.pending = ""

    def push(self, text: str) -> str:
        combined = self.pending + text
        self.pending = ""
        visible: list[str] = []
        index = 0

        while index < len(combined):
            if self.in_reasoning:
                close_index = combined.find(self.close_tag, index)
                if close_index == -1:
                    self.pending = self._tag_prefix_suffix(combined[index:], self.close_tag)
                    return "".join(visible)
                index = close_index + len(self.close_tag)
                self.in_reasoning = False
                continue

            open_index = combined.find(self.open_tag, index)
            if open_index == -1:
                segment = combined[index:]
                suffix = self._tag_prefix_suffix(segment, self.open_tag)
                visible.append(segment[: len(segment) - len(suffix)] if suffix else segment)
                self.pending = suffix
                return "".join(visible)

            visible.append(combined[index:open_index])
            index = open_index + len(self.open_tag)
            self.in_reasoning = True

        return "".join(visible)

    @staticmethod
    def _tag_prefix_suffix(text: str, tag: str) -> str:
        max_length = min(len(text), len(tag) - 1)
        for length in range(max_length, 0, -1):
            suffix = text[-length:]
            if tag.startswith(suffix):
                return suffix
        return ""


def build_runtime_phase_payload(thread_id: str, phase: str) -> dict:
    phase_event = context_manager.append_event(
        thread_id,
        EventRecord(
            event_type="phase_changed",
            thread_id=thread_id,
            payload={"phase": phase},
        ),
    )
    return {
        "type": "runtime_event",
        "event_type": phase_event.event_type,
        "sequence": phase_event.sequence,
        "payload": phase_event.payload,
    }


def build_runtime_prelude_payloads(thread_id: str, message: str) -> list[dict]:
    phase_payload = build_runtime_phase_payload(thread_id, "preparing_context")
    runtime_artifacts = refresh_runtime_artifacts(
        thread_id=thread_id,
        draft_message=message,
    )
    return [
        phase_payload,
        *[
            {
                "type": "runtime_event",
                "event_type": event.event_type,
                "sequence": event.sequence,
                "payload": event.payload,
            }
            for event in runtime_artifacts.get("events", [])
        ],
        {
            "type": "state_update",
            "sequence": runtime_artifacts["snapshot"].sequence,
            "state": runtime_artifacts["state"].to_dict(),
        },
        {
            "type": "context_assembly",
            "sequence": runtime_artifacts["assembly"].sequence,
            "assembly": runtime_artifacts["assembly"].to_dict(),
            "context_blocks": [block.to_dict() for block in runtime_artifacts["context_blocks"]],
        },
    ]


def build_runtime_artifact_payloads(runtime_artifacts: dict) -> list[dict]:
    return [
        *[
            {
                "type": "runtime_event",
                "event_type": event.event_type,
                "sequence": event.sequence,
                "payload": event.payload,
            }
            for event in runtime_artifacts.get("events", [])
        ],
        {
            "type": "state_update",
            "sequence": runtime_artifacts["snapshot"].sequence,
            "state": runtime_artifacts["state"].to_dict(),
        },
        {
            "type": "context_assembly",
            "sequence": runtime_artifacts["assembly"].sequence,
            "assembly": runtime_artifacts["assembly"].to_dict(),
            "context_blocks": [block.to_dict() for block in runtime_artifacts["context_blocks"]],
        },
    ]


def build_model_call_payloads(thread_id: str, sent_call_ids: set[str]) -> list[dict]:
    payloads = []
    for record in context_manager.load_model_calls(thread_id):
        if record.call_id in sent_call_ids:
            continue
        sent_call_ids.add(record.call_id)
        payloads.append({"type": "model_call", "model_call": record.to_dict()})
    return payloads


def build_runtime_stream_payload(thread_id: str, delta_text: str, cumulative_chars: int) -> dict:
    stream_event = context_manager.append_event(
        thread_id,
        EventRecord(
            event_type="assistant_answer_streamed",
            thread_id=thread_id,
            payload={
                "delta_text": delta_text,
                "delta_chars": len(delta_text),
                "cumulative_chars": cumulative_chars,
            },
        ),
    )
    return {
        "type": "runtime_event",
        "event_type": stream_event.event_type,
        "sequence": stream_event.sequence,
        "payload": stream_event.payload,
    }


def build_tool_called_payloads(thread_id: str, tool_name: str, tool_args: dict) -> list[dict]:
    tool_event = context_manager.append_event(
        thread_id,
        EventRecord(
            event_type="tool_called",
            thread_id=thread_id,
            payload={"tool": tool_name, "args": tool_args},
        ),
    )
    return [
        {
            "type": "runtime_event",
            "event_type": tool_event.event_type,
            "sequence": tool_event.sequence,
            "payload": tool_event.payload,
        },
        build_runtime_phase_payload(thread_id, "awaiting_tool_result"),
        {
            "type": "tool_call",
            "tool": tool_name,
            "args": json.dumps(tool_args, ensure_ascii=False)[:500],
        },
    ]


def build_tool_result_payloads(
    thread_id: str,
    tool_name: str,
    tool_content: str,
    tool_preview: str,
    current_request: str = "",
) -> list[dict]:
    tool_result_event = context_manager.append_event(
        thread_id,
        EventRecord(
            event_type="tool_result_received",
            thread_id=thread_id,
            payload={"tool": tool_name, "preview": tool_preview},
        ),
    )
    context_manager.save_tool_evidence(
        thread_id,
        ToolEvidence(
            evidence_id=f"tool:{thread_id}:{tool_result_event.sequence}",
            thread_id=thread_id,
            tool_name=tool_name,
            content=tool_content,
            preview=tool_preview,
            sequence=tool_result_event.sequence,
        ),
    )
    refreshed_artifacts = refresh_runtime_artifacts(
        thread_id=thread_id,
        draft_message=current_request,
    )
    return [
        {
            "type": "runtime_event",
            "event_type": tool_result_event.event_type,
            "sequence": tool_result_event.sequence,
            "payload": tool_result_event.payload,
        },
        build_runtime_phase_payload(thread_id, "processing_tool_result"),
        {"type": "tool_result", "tool": tool_name, "preview": tool_preview},
        *build_runtime_artifact_payloads(refreshed_artifacts),
    ]


def build_runtime_completion_payloads(thread_id: str, full_content: str) -> list[dict]:
    completed_event = context_manager.append_event(
        thread_id,
        EventRecord(
            event_type="assistant_answer_completed",
            thread_id=thread_id,
            payload={"content_length": len(full_content)},
        ),
    )
    return [
        {
            "type": "runtime_event",
            "event_type": completed_event.event_type,
            "sequence": completed_event.sequence,
            "payload": completed_event.payload,
        },
        build_runtime_phase_payload(thread_id, "idle"),
    ]


def build_runtime_error_payloads(thread_id: str, error_message: str) -> list[dict]:
    error_event = context_manager.append_event(
        thread_id,
        EventRecord(
            event_type="agent_error",
            thread_id=thread_id,
            payload={"message": error_message},
        ),
    )
    return [
        {
            "type": "runtime_event",
            "event_type": error_event.event_type,
            "sequence": error_event.sequence,
            "payload": error_event.payload,
        },
        build_runtime_phase_payload(thread_id, "idle"),
    ]


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
        sent_model_call_ids: set[str] = set()
        reasoning_filter = ReasoningTextFilter()
        context_manager.clear_model_calls(thread_id)
        user_event = context_manager.append_event(
            thread_id,
            EventRecord(
                event_type="user_message_received",
                thread_id=thread_id,
                payload={"content": current_message.content},
            ),
        )
        yield sse_event(
            {
                "type": "runtime_event",
                "event_type": user_event.event_type,
                "sequence": user_event.sequence,
                "payload": user_event.payload,
            }
        )
        for payload in build_runtime_prelude_payloads(thread_id, current_message.content):
            yield sse_event(payload)

        try:
            result = agent.astream(
                {"messages": [current_message], "thread_id": thread_id},
                stream_mode="messages",
                version="v2",
            )

            async for chunk in result:
                for payload in build_model_call_payloads(thread_id, sent_model_call_ids):
                    yield sse_event(payload)

                if chunk.get("type") != "messages":
                    continue

                token, metadata = chunk.get("data")
                node = metadata.get("langgraph_node", "")

                if node == "tools":
                    tool_content = getattr(token, "content", "") or ""
                    if tool_content:
                        tool_name = getattr(token, "name", "unknown")
                        tool_preview = tool_content[:200] + "..." if len(tool_content) > 200 else tool_content
                        for payload in build_tool_result_payloads(
                            thread_id,
                            tool_name,
                            tool_content,
                            tool_preview,
                            current_request=current_message.content,
                        ):
                            yield sse_event(payload)
                        for payload in build_model_call_payloads(thread_id, sent_model_call_ids):
                            yield sse_event(payload)
                    continue

                if node != "model":
                    continue

                if hasattr(token, "tool_calls") and token.tool_calls:
                    for tool_call in token.tool_calls:
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get("name", "unknown")
                            tool_args = tool_call.get("args", {})
                        else:
                            tool_name = getattr(tool_call, "name", "unknown")
                            tool_args = getattr(tool_call, "args", {})
                        if tool_name:
                            for payload in build_tool_called_payloads(thread_id, tool_name, tool_args):
                                yield sse_event(payload)
                            for payload in build_model_call_payloads(thread_id, sent_model_call_ids):
                                yield sse_event(payload)

                if hasattr(token, "content_blocks") and token.content_blocks:
                    for block in token.content_blocks:
                        block_data = (
                            block
                            if isinstance(block, dict)
                            else {"type": getattr(block, "type", ""), "text": getattr(block, "text", "")}
                        )
                        if block_data.get("type") != "text":
                            continue
                        text = reasoning_filter.push(block_data.get("text") or "")
                        if not text:
                            continue
                        if not is_assistant_responding:
                            yield sse_event(build_runtime_phase_payload(thread_id, "streaming_answer"))
                        is_assistant_responding = True
                        full_content += text
                        yield sse_event(
                            build_runtime_stream_payload(
                                thread_id,
                                text,
                                len(full_content),
                            )
                        )
                        yield sse_event({"type": "text", "content": text})

        except Exception as e:
            error_message = str(e)
            for payload in build_model_call_payloads(thread_id, sent_model_call_ids):
                yield sse_event(payload)
            for payload in build_runtime_error_payloads(thread_id, error_message):
                yield sse_event(payload)
            yield sse_event({"type": "error", "content": error_message})

        for payload in build_model_call_payloads(thread_id, sent_model_call_ids):
            yield sse_event(payload)

        if full_content and is_assistant_responding:
            from langchain_core.messages import AIMessage

            await context_manager.append_messages(
                thread_id,
                [current_message, AIMessage(content=full_content)],
            )
            for payload in build_runtime_completion_payloads(thread_id, full_content):
                yield sse_event(payload)

        yield sse_event({"type": "done", "thread_id": thread_id, "full_content": full_content})

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
    from langchain_core.messages import AIMessage, HumanMessage

    thread_id = request.thread_id or create_thread_id()
    current_message = HumanMessage(content=request.message.strip())

    result = await agent.ainvoke(
        {"messages": [current_message], "thread_id": thread_id},
        context=AgentContext(thread_id=thread_id),
    )

    messages = result.get("messages", [])
    ai_messages = [message for message in messages if getattr(message, "type", "") == "ai"]
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
