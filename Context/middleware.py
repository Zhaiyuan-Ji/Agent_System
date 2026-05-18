"""LangChain middleware for context injection and model-call tracing."""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    after_model,
    before_model,
)
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

from .runtime_models import EventRecord, ModelCallRecord, utc_now_iso


class ChatAgentState(AgentState):
    thread_id: NotRequired[str]
    loaded_message_count: NotRequired[int]
    max_messages: NotRequired[int] = 3


class ContextInjectMiddleware(AgentMiddleware[ChatAgentState]):
    state_schema = ChatAgentState

    current_thread_id: str | None = None
    current_managed_messages: list[BaseMessage] = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ExtendedModelResponse:
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ExtendedModelResponse:
        thread_id = _get_thread_id_from_request(request) or ContextInjectMiddleware.current_thread_id
        managed_messages = ContextInjectMiddleware.current_managed_messages
        context_manager = None
        model_call_record: ModelCallRecord | None = None

        try:
            if thread_id and managed_messages:
                request = request.override(messages=[*managed_messages, *request.messages])

            if thread_id:
                from .manager import RedisContextManager

                context_manager = RedisContextManager()
                existing_calls = context_manager.load_model_calls(thread_id)
                call_index = len(existing_calls) + 1
                started_event = context_manager.append_event(
                    thread_id,
                    EventRecord(
                        event_type="llm_call_started",
                        thread_id=thread_id,
                        payload={"call_index": call_index},
                    ),
                )
                model_call_record = _build_started_model_call_record(
                    thread_id=thread_id,
                    call_index=call_index,
                    sequence=started_event.sequence,
                    request=request,
                    context_manager=context_manager,
                )

            response = await handler(request)

            if context_manager and model_call_record:
                completed_record = _complete_model_call_record(model_call_record, response)
                context_manager.save_model_call(thread_id, completed_record)
                context_manager.append_event(
                    thread_id,
                    EventRecord(
                        event_type="llm_call_completed",
                        thread_id=thread_id,
                        payload={
                            "call_id": completed_record.call_id,
                            "call_index": completed_record.call_index,
                            "phase": completed_record.phase,
                            "output_type": completed_record.output.get("type", "unknown"),
                        },
                    ),
                )

            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={"loaded_message_count": len(managed_messages)}),
            )
        except Exception as exc:
            if context_manager and model_call_record and thread_id:
                failed_record = ModelCallRecord(
                    **{
                        **model_call_record.to_dict(),
                        "completed_at": utc_now_iso(),
                        "output": {"type": "error", "summary": str(exc)},
                    }
                )
                context_manager.save_model_call(thread_id, failed_record)
                context_manager.append_event(
                    thread_id,
                    EventRecord(
                        event_type="llm_call_failed",
                        thread_id=thread_id,
                        payload={
                            "call_id": failed_record.call_id,
                            "call_index": failed_record.call_index,
                            "message": str(exc),
                        },
                    ),
                )
            raise
        finally:
            ContextInjectMiddleware.current_thread_id = None
            ContextInjectMiddleware.current_managed_messages = []

    def save_tool_logs(self, thread_id: str, logs: list) -> None:
        try:
            from .manager import RedisContextManager

            context_manager = RedisContextManager()
            log_key = f"tool_logs:{thread_id}"
            context_manager.redis_client.setex(log_key, 3600, json.dumps(logs, ensure_ascii=False))
        except Exception:
            pass


@before_model(state_schema=ChatAgentState)
def inject_context(state: ChatAgentState, runtime: Runtime) -> dict[str, Any] | None:
    from .manager import RedisContextManager

    ContextInjectMiddleware.current_thread_id = None
    ContextInjectMiddleware.current_managed_messages = []

    thread_id = state.get("thread_id")
    if not thread_id:
        return None

    context_manager = RedisContextManager()
    stored_messages = context_manager.load_context_messages(thread_id)

    context_notes: list[BaseMessage] = []
    dialogue_messages: list[BaseMessage] = []
    for message in stored_messages:
        message_type = getattr(message, "type", "")
        if message_type == "system":
            content = (getattr(message, "content", "") or "").strip()
            if content:
                context_notes.append(
                    HumanMessage(
                        id=getattr(message, "id", None),
                        content=f"[Historical Summary]\n{content}",
                    )
                )
        elif message_type in ("human", "ai"):
            dialogue_messages.append(message)

    max_count = state.get("max_messages", 20)
    managed_messages = [*context_notes, *dialogue_messages[-max_count:]]
    if not managed_messages:
        return None

    ContextInjectMiddleware.current_thread_id = thread_id
    ContextInjectMiddleware.current_managed_messages = managed_messages
    return {"loaded_message_count": len(managed_messages)}


def _get_thread_id_from_request(request: ModelRequest) -> str | None:
    state = getattr(request, "state", None) or {}
    if isinstance(state, dict):
        value = state.get("thread_id")
        return str(value) if value else None
    value = getattr(state, "thread_id", None)
    return str(value) if value else None


def _build_started_model_call_record(
    thread_id: str,
    call_index: int,
    sequence: int,
    request: ModelRequest,
    context_manager: Any,
) -> ModelCallRecord:
    phase = _infer_model_call_phase(request.messages)
    latest_snapshot = context_manager.load_latest_snapshot(thread_id)
    state_snapshot = latest_snapshot.state.to_dict() if latest_snapshot else None

    return ModelCallRecord(
        thread_id=thread_id,
        call_id=f"call_{call_index}",
        call_index=call_index,
        sequence=sequence,
        phase=phase,
        purpose=_infer_model_call_purpose(phase),
        input_context=_serialize_model_call_context(request),
        state_snapshot=state_snapshot,
        output={"type": "pending", "summary": "model call is running"},
    )


def _complete_model_call_record(record: ModelCallRecord, response: ModelResponse) -> ModelCallRecord:
    message = _extract_latest_model_message(response)
    output, raw_think = _serialize_model_output(message)
    return ModelCallRecord(
        **{
            **record.to_dict(),
            "completed_at": utc_now_iso(),
            "output": output,
            "raw_think": raw_think,
        }
    )


def _serialize_model_call_context(request: ModelRequest) -> list[dict[str, Any]]:
    instruction_sections: list[dict[str, Any]] = []
    dialogue_sections: list[dict[str, Any]] = []
    tool_sections: list[dict[str, Any]] = []
    other_sections: list[dict[str, Any]] = []

    system_message = getattr(request, "system_message", None)
    if system_message:
        instruction_sections.append(
            _build_context_section(
                label="system_prompt",
                kind="system",
                value=_message_content_to_text(getattr(system_message, "content", "")),
                message_type=getattr(system_message, "type", "system"),
            )
        )

    for message in request.messages:
        message_type = getattr(message, "type", "")
        content = _message_content_to_text(getattr(message, "content", ""))

        if message_type == "human" and content.startswith("[Historical Summary]"):
            label = "memory"
            kind = "memory"
        elif message_type == "human":
            label = "user_request" if _is_latest_human_message(message, request.messages) else "recent_dialogue"
            kind = "user" if label == "user_request" else "dialogue"
        elif message_type == "tool":
            label = "tool_message"
            kind = "tool"
        elif message_type == "ai":
            label = "recent_dialogue"
            kind = "dialogue"
        elif message_type == "system":
            label = "system_prompt"
            kind = "system"
        else:
            label = message_type or "message"
            kind = "memory"

        section = _build_context_section(
            label=label,
            kind=kind,
            value=content,
            message_type=message_type,
            id=getattr(message, "id", None),
            name=getattr(message, "name", None),
        )

        if kind == "tool":
            tool_sections.append(section)
        elif kind in ("dialogue", "user", "memory"):
            dialogue_sections.append(section)
        elif kind == "system":
            instruction_sections.append(section)
        else:
            other_sections.append(section)

    groups: list[dict[str, Any]] = []
    if instruction_sections:
        groups.append(_build_context_group("instruction_layer", "Input: Instructions", "system", instruction_sections))
    if dialogue_sections:
        groups.append(_build_context_group("dialogue_layer", "Input: Conversation", "dialogue", dialogue_sections))
    if tool_sections:
        groups.append(_build_context_group("tool_layer", "Input: Tool Evidence", "tool", tool_sections))
    if other_sections:
        groups.append(_build_context_group("runtime_layer", "Input: Runtime Context", "memory", other_sections))
    return groups


def _build_context_section(**item: Any) -> dict[str, Any]:
    value = _message_content_to_text(item.get("value", ""))
    label = str(item.get("label") or "context")
    return {
        **item,
        "label": label,
        "value": value,
        "preview": value[:80],
        "estimated_tokens": max(1, len(value) // 4) if value else 0,
        "selected": True,
    }


def _build_context_group(
    group_id: str,
    title: str,
    kind: str,
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    estimated_tokens = sum(int(section.get("estimated_tokens") or 0) for section in sections)
    return {
        "group_id": group_id,
        "label": title,
        "kind": kind,
        "summary": f"{len(sections)} sections / ~{estimated_tokens} tokens",
        "estimated_tokens": estimated_tokens,
        "selected": True,
        "sections": sections,
    }


def _is_latest_human_message(target: BaseMessage, messages: list[BaseMessage]) -> bool:
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            return message is target
    return False


def _infer_model_call_phase(messages: list[BaseMessage]) -> str:
    if any(getattr(message, "type", "") == "tool" for message in messages):
        return "answering"
    if any(getattr(message, "tool_calls", None) for message in messages):
        return "processing_tool_result"
    return "planning"


def _infer_model_call_purpose(phase: str) -> str:
    if phase == "answering":
        return "读取工具结果、状态和上下文，生成后续回答。"
    if phase == "processing_tool_result":
        return "根据已有工具调用轨迹决定下一步。"
    return "理解用户请求并决定是否需要工具。"


def _extract_latest_model_message(response: ModelResponse) -> BaseMessage | None:
    result = getattr(response, "result", None) or []
    ai_messages = [message for message in result if getattr(message, "type", "") == "ai"]
    if ai_messages:
        return ai_messages[-1]
    return result[-1] if result else None


def _serialize_model_output(message: BaseMessage | None) -> tuple[dict[str, Any], str | None]:
    if message is None:
        return {"type": "empty", "summary": "model returned no message"}, None

    content = _message_content_to_text(getattr(message, "content", ""))
    raw_think = _extract_raw_think(content)
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        serialized_tool_calls = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                serialized_tool_calls.append(
                    {
                        "name": tool_call.get("name", "unknown"),
                        "args": tool_call.get("args", {}),
                    }
                )
            else:
                serialized_tool_calls.append(
                    {
                        "name": getattr(tool_call, "name", "unknown"),
                        "args": getattr(tool_call, "args", {}),
                    }
                )
        names = ", ".join(item["name"] for item in serialized_tool_calls)
        return {
            "type": "tool_call",
            "summary": f"模型请求调用工具：{names}",
            "tool_calls": serialized_tool_calls,
        }, raw_think

    visible_content = _remove_think_blocks(content).strip()
    return {
        "type": "answer",
        "summary": visible_content[:240] if visible_content else "模型返回了空回答。",
        "answer": visible_content,
    }, raw_think


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False, default=str)


def _extract_raw_think(text: str) -> str | None:
    open_tag = "<think>"
    close_tag = "</think>"
    start = text.find(open_tag)
    end = text.find(close_tag, start + len(open_tag))
    if start == -1 or end == -1:
        return None
    return text[start + len(open_tag) : end].strip() or None


def _remove_think_blocks(text: str) -> str:
    open_tag = "<think>"
    close_tag = "</think>"
    result = []
    index = 0
    while index < len(text):
        start = text.find(open_tag, index)
        if start == -1:
            result.append(text[index:])
            break
        result.append(text[index:start])
        end = text.find(close_tag, start + len(open_tag))
        if end == -1:
            break
        index = end + len(close_tag)
    return "".join(result)


@after_model(state_schema=ChatAgentState)
def log_after_model(state: ChatAgentState, runtime: Runtime) -> dict[str, Any] | None:
    thread_id = state.get("thread_id")
    messages = state.get("messages", [])

    ContextInjectMiddleware.current_thread_id = None
    ContextInjectMiddleware.current_managed_messages = []

    if not thread_id:
        return None

    tool_calls = []
    tool_results = []

    for message in messages:
        message_type = getattr(message, "type", "")

        if message_type == "ai":
            for tool_call in getattr(message, "tool_calls", []) or []:
                if isinstance(tool_call, dict):
                    name = tool_call.get("name", "unknown")
                    args = tool_call.get("args", {})
                else:
                    name = getattr(tool_call, "name", "unknown")
                    args = getattr(tool_call, "args", {})
                tool_calls.append({"type": "tool_call", "tool": name, "args": args})

        elif message_type == "tool":
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool": getattr(message, "name", ""),
                    "result": getattr(message, "content", ""),
                }
            )

    if tool_calls:
        try:
            from Context.redis_client import redis_client

            redis_client.setex(f"tool_logs:{thread_id}", 3600, json.dumps(tool_calls, ensure_ascii=False))
        except Exception:
            pass

    if tool_results:
        try:
            from Context.redis_client import redis_client

            redis_client.setex(f"tool_results:{thread_id}", 3600, json.dumps(tool_results, ensure_ascii=False))
        except Exception:
            pass

    return None
