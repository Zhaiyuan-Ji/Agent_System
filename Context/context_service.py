"""
Context Service Module

提供上下文服务和序列化工具。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from .manager import RedisContextManager
from Agent.System_prompt import SYSTEM_PROMPT

context_manager = RedisContextManager()


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }
    message_type = getattr(message, "type", message.__class__.__name__.lower())

    return {
        "id": getattr(message, "id", None),
        "type": message_type,
        "role": role_map.get(message_type, message_type),
        "content": message.content,
    }


def split_context_sections(thread_id: str, draft_message: str = "") -> dict[str, Any]:
    system_message = SystemMessage(content=SYSTEM_PROMPT, id="system:agent")
    context_messages = context_manager.load_context_messages(thread_id)

    summary_messages = [
        message
        for message in context_messages
        if getattr(message, "type", "") == "system"
        and str(getattr(message, "id", "")).startswith("summary:")
    ]
    history_messages = [message for message in context_messages if message not in summary_messages]

    latest_user_index = -1

    if not draft_message.strip():
        for index in range(len(history_messages) - 1, -1, -1):
            if getattr(history_messages[index], "type", "") == "human":
                latest_user_index = index
                break
    else:
        for index in range(len(history_messages) - 1, -1, -1):
            if getattr(history_messages[index], "type", "") == "human":
                latest_user_index = index
                if history_messages[index].content.strip() == draft_message.strip():
                    break

    current_question_msg = (
        serialize_message(history_messages[latest_user_index]) if latest_user_index >= 0 else None
    )

    current_answer_messages: list[dict[str, Any]] = []
    for index in range(latest_user_index + 1, len(history_messages)):
        if getattr(history_messages[index], "type", "") == "ai":
            current_answer_messages.append(serialize_message(history_messages[index]))
            break

    from .config import CHAT_MODE, OPENAI_MODEL, OPENAI_BASE_URL

    sections: list[dict[str, Any]] = [
        {
            "key": "system",
            "title": "系统提示词",
            "description": "这是你应遵循的指令。",
            "messages": [serialize_message(system_message)],
        },
    ]

    if summary_messages:
        sections.append(
            {
                "key": "summary",
                "title": "历史摘要",
                "description": "这是从更早对话中提取的关键信息摘要。",
                "messages": [serialize_message(message) for message in summary_messages],
            }
        )

    if history_messages:
        sections.append(
            {
                "key": "past_context",
                "title": "过去的上下文",
                "description": "这部分是当前问题之前的历史消息。",
                "messages": [serialize_message(message) for message in history_messages],
            }
        )

    return {
        "thread_id": thread_id,
        "mode": CHAT_MODE,
        "model": OPENAI_MODEL,
        "base_url": OPENAI_BASE_URL,
        "current_question": current_question_msg,
        "current_answer": current_answer_messages,
        "sections": sections,
    }


def get_prompt_context(thread_id: str, draft_message: str = "") -> dict[str, Any]:
    return split_context_sections(thread_id=thread_id, draft_message=draft_message)
