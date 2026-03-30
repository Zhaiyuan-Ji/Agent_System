"""
Context Middleware Module

提供 LangChain 中间件，用于在模型调用前后注入历史上下文。
"""

from __future__ import annotations

import os
import json
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
        thread_id = ContextInjectMiddleware.current_thread_id
        managed_messages = ContextInjectMiddleware.current_managed_messages

        if thread_id and managed_messages:
            new_messages = [*managed_messages, *request.messages]
            request = request.override(messages=new_messages)

        response = await handler(request)

        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"loaded_message_count": len(managed_messages)}),
        )

    def save_tool_logs(self, thread_id: str, logs: list) -> None:
        try:
            from .manager import RedisContextManager
            context_manager = RedisContextManager()
            log_key = f"tool_logs:{thread_id}"
            log_data = json.dumps(logs, ensure_ascii=False)
            context_manager.redis_client.setex(log_key, 3600, log_data)
        except Exception:
            pass


@before_model(state_schema=ChatAgentState)
def inject_context(state: ChatAgentState, runtime: Runtime) -> dict[str, Any] | None:
    from .manager import RedisContextManager

    thread_id = state.get("thread_id")

    if thread_id:
        context_manager = RedisContextManager()
        stored_messages = context_manager.load_context_messages(thread_id)

        filtered_messages = []
        for msg in stored_messages:
            msg_type = getattr(msg, "type", "")
            if msg_type in ("human", "ai", "system"):
                filtered_messages.append(msg)

        max_count = state.get("max_messages", 20)
        if len(filtered_messages) > max_count:
            system_msgs = [m for m in filtered_messages if getattr(m, "type", "") == "system"]
            other_msgs = [m for m in filtered_messages if getattr(m, "type", "") != "system"]
            kept_msgs = other_msgs[-max_count + len(system_msgs):]
            managed_messages = [*system_msgs, *kept_msgs]
        else:
            managed_messages = filtered_messages

        if managed_messages:
            ContextInjectMiddleware.current_thread_id = thread_id
            ContextInjectMiddleware.current_managed_messages = managed_messages

            return {"loaded_message_count": len(managed_messages)}

    return None


@after_model(state_schema=ChatAgentState)
def log_after_model(state: ChatAgentState, runtime: Runtime) -> dict[str, Any] | None:
    thread_id = state.get("thread_id")
    messages = state.get("messages", [])

    ContextInjectMiddleware.current_thread_id = None
    ContextInjectMiddleware.current_managed_messages = []

    if thread_id:
        tool_calls = []
        tool_results = []

        for i, msg in enumerate(messages):
            msg_type = getattr(msg, "type", "")

            if msg_type == "ai":
                tc_list = getattr(msg, "tool_calls", [])
                if tc_list:
                    for tc in tc_list:
                        if isinstance(tc, dict):
                            name = tc.get("name", "unknown")
                            args = tc.get("args", {})
                        else:
                            name = getattr(tc, "name", "unknown")
                            args = getattr(tc, "args", {})
                        tool_calls.append({
                            "type": "tool_call",
                            "tool": name,
                            "args": args
                        })

            elif msg_type == "tool":
                name = getattr(msg, "name", "")
                content = getattr(msg, "content", "")
                tool_results.append({
                    "type": "tool_result",
                    "tool": name,
                    "result": content
                })

        if tool_calls:
            try:
                from Context.redis_client import redis_client
                log_key = f"tool_logs:{thread_id}"
                log_data = json.dumps(tool_calls, ensure_ascii=False)
                redis_client.setex(log_key, 3600, log_data)
            except Exception:
                pass

        if tool_results:
            try:
                from Context.redis_client import redis_client
                result_key = f"tool_results:{thread_id}"
                result_data = json.dumps(tool_results, ensure_ascii=False)
                redis_client.setex(result_key, 3600, result_data)
            except Exception:
                pass

    return None
