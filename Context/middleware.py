"""
Context Middleware Module

提供 LangChain 中间件，用于在模型调用前后注入历史上下文。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from langchain.agents.middleware import (
    before_model,
    after_model,
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ExtendedModelResponse,
)
from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


class ChatAgentState(AgentState):
    thread_id: NotRequired[str]
    loaded_message_count: NotRequired[int]
    max_messages: NotRequired[int] = 20


def filter_messages(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """过滤消息类型，只移除 thinking，保留 human、ai、system、tool"""
    filtered = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type in ("human", "ai", "system", "tool", "tool_calls"):
            filtered.append(msg)
    return filtered


def manage_context_window(
    messages: list[BaseMessage],
    max_count: int = 20
) -> list[BaseMessage]:
    if len(messages) <= max_count:
        return messages

    system_msgs = [m for m in messages if getattr(m, "type", "") == "system"]
    other_msgs = [m for m in messages if getattr(m, "type", "") != "system"]

    kept_msgs = other_msgs[-max_count + len(system_msgs):]

    return [*system_msgs, *kept_msgs]


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
        thread_id = ContextInjectMiddleware.current_thread_id
        managed_messages = ContextInjectMiddleware.current_managed_messages

        if DEBUG_MODE:
            print(f"\n[awrap_model_call] thread_id from middleware: {thread_id}")
            print(f"  managed_messages count: {len(managed_messages)}")

        if thread_id and managed_messages:
            new_messages = [*managed_messages, *request.messages]
            request = request.override(messages=new_messages)

            if DEBUG_MODE:
                print(f"  after inject:")
                for i, msg in enumerate(request.messages):
                    content = getattr(msg, "content", str(msg))[:40]
                    msg_type = getattr(msg, "type", "unknown")
                    print(f"    [{i}] {msg_type}: {content}")

        response = await handler(request)

        if DEBUG_MODE:
            print(f"  model response received")

        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"loaded_message_count": len(managed_messages)}),
        )


@before_model(state_schema=ChatAgentState)
def inject_context(state: ChatAgentState, runtime: Runtime) -> dict[str, Any] | None:
    from .manager import RedisContextManager

    thread_id = state.get("thread_id")

    if DEBUG_MODE:
        print(f"\n[@before_model] inject_context")
        print(f"  thread_id from state: {thread_id}")
        print(f"  current messages: {len(state.get('messages', []))}")

    if thread_id:
        context_manager = RedisContextManager()
        stored_messages = context_manager.load_context_messages(thread_id)
        filtered_messages = filter_messages(stored_messages)
        managed_messages = manage_context_window(filtered_messages)

        if managed_messages:
            ContextInjectMiddleware.current_thread_id = thread_id
            ContextInjectMiddleware.current_managed_messages = managed_messages

            if DEBUG_MODE:
                print(f"  loaded {len(managed_messages)} messages from storage")

            return {"loaded_message_count": len(managed_messages)}

    return None


@after_model(state_schema=ChatAgentState)
def log_after_model(state: ChatAgentState, runtime: Runtime) -> dict[str, Any] | None:
    ContextInjectMiddleware.current_thread_id = None
    ContextInjectMiddleware.current_managed_messages = []

    if DEBUG_MODE:
        messages = state.get("messages", [])
        print(f"\n[@after_model] log_after_model called")
        print(f"  total messages: {len(messages)}")
        last_msg = messages[-1] if messages else None
        if last_msg:
            print(f"  last message: {getattr(last_msg, 'type', 'unknown')} - {getattr(last_msg, 'content', '')[:50]}")
    return None
