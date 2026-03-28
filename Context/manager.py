from __future__ import annotations

import json
import os
from typing import Iterable

from langchain_core.messages import BaseMessage, SystemMessage, message_to_dict, messages_from_dict
from openai import AsyncOpenAI

from Context.redis_client import redis_client

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:54329/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")
SUMMARY_MODEL = os.getenv("CONTEXT_SUMMARY_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.1"))


class RedisContextManager:
    def __init__(
        self,
        recent_message_limit: int = 12,
        compact_trigger_messages: int = 20,
    ) -> None:
        self.recent_message_limit = recent_message_limit
        self.compact_trigger_messages = compact_trigger_messages
        self.summary_client = AsyncOpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
        )

    def _messages_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:messages"

    def _summary_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:summary"

    def load_messages(self, thread_id: str) -> list[BaseMessage]:
        raw_value = redis_client.get(self._messages_key(thread_id))

        if not raw_value:
            return []

        return messages_from_dict(json.loads(raw_value))

    def load_summary(self, thread_id: str) -> str:
        return redis_client.get(self._summary_key(thread_id)) or ""

    def load_context_messages(self, thread_id: str) -> list[BaseMessage]:
        stored_messages = self.load_messages(thread_id)
        summary = self.load_summary(thread_id)
        context_messages: list[BaseMessage] = []

        if summary:
            context_messages.append(
                SystemMessage(
                    id=f"summary:{thread_id}",
                    content=(
                        "下面是当前会话的历史摘要，请把它当作已经确认过的背景继续使用：\n"
                        f"{summary}"
                    ),
                )
            )

        context_messages.extend(stored_messages)
        return context_messages

    async def append_messages(self, thread_id: str, new_messages: Iterable[BaseMessage]) -> None:
        stored_messages = self.load_messages(thread_id)
        appended_messages = [*stored_messages, *list(new_messages)]

        redis_client.set(
            self._messages_key(thread_id),
            json.dumps([message_to_dict(message) for message in appended_messages], ensure_ascii=False),
        )

        if len(appended_messages) > self.compact_trigger_messages:
            await self.compact(thread_id, appended_messages)

    async def compact(self, thread_id: str, messages: list[BaseMessage]) -> None:
        head_messages = messages[:-self.recent_message_limit]
        tail_messages = messages[-self.recent_message_limit:]
        existing_summary = self.load_summary(thread_id)
        summary = await self.summarize(existing_summary, head_messages)

        redis_client.set(self._summary_key(thread_id), summary)
        redis_client.set(
            self._messages_key(thread_id),
            json.dumps([message_to_dict(message) for message in tail_messages], ensure_ascii=False),
        )

    async def summarize(self, existing_summary: str, messages: list[BaseMessage]) -> str:
        transcript = self._format_messages(messages)
        completion = await self.summary_client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是会话压缩助手。请把旧对话压缩成简洁但可继续推理的上下文。"
                        "保留用户目标、已确认事实、关键约束、已完成步骤、未完成事项。"
                        "如果存在工具调用，只保留工具结论，不保留冗长原始输出。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"已有摘要：\n{existing_summary or '无'}\n\n"
                        f"需要压缩的新消息：\n{transcript}\n\n"
                        "请输出新的完整摘要。"
                    ),
                },
            ],
        )

        return completion.choices[0].message.content or existing_summary

    async def clear_thread(self, thread_id: str) -> None:
        redis_client.delete(self._messages_key(thread_id))
        redis_client.delete(self._summary_key(thread_id))

    def _format_messages(self, messages: list[BaseMessage]) -> str:
        lines: list[str] = []

        for message in messages:
            role = getattr(message, "type", message.__class__.__name__)
            lines.append(f"[{role}] {message.content}")

        return "\n".join(lines)
