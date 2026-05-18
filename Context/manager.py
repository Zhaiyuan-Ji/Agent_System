from __future__ import annotations

import json
import os
from typing import Iterable

from langchain_core.messages import BaseMessage, SystemMessage, message_to_dict, messages_from_dict
from openai import AsyncOpenAI

from Context.http_client_factory import create_async_http_client
from Context.redis_client import redis_client
from Context.runtime_models import (
    AssemblyRecord,
    ContextBlock,
    EventRecord,
    ModelCallRecord,
    StateSnapshot,
    ToolEvidence,
)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.minimaxi.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUMMARY_MODEL = os.getenv("CONTEXT_SUMMARY_MODEL", os.getenv("OPENAI_MODEL", "MiniMax-M2.7"))


class RedisContextManager:
    def __init__(
        self,
        recent_message_limit: int = 12,
        compact_trigger_messages: int = 3,
    ) -> None:
        self.recent_message_limit = recent_message_limit
        self.compact_trigger_messages = compact_trigger_messages
        self.summary_client = AsyncOpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            http_client=create_async_http_client(),
        )

    def _messages_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:messages"

    def _summary_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:summary"

    def _events_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:events"

    def _snapshot_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:snapshot"

    def _snapshots_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:snapshots"

    def _assembly_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:assembly"

    def _assemblies_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:assemblies"

    def _blocks_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:blocks"

    def _tool_evidence_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:tool_evidence"

    def _model_calls_key(self, thread_id: str) -> str:
        return f"context:thread:{thread_id}:model_calls"

    def load_messages(self, thread_id: str) -> list[BaseMessage]:
        raw_value = redis_client.get(self._messages_key(thread_id))

        if not raw_value:
            return []

        return messages_from_dict(json.loads(raw_value))

    def load_summary(self, thread_id: str) -> str:
        return redis_client.get(self._summary_key(thread_id)) or ""

    def load_events(self, thread_id: str) -> list[EventRecord]:
        raw_value = redis_client.get(self._events_key(thread_id))
        if not raw_value:
            return []
        return [EventRecord.from_dict(item) for item in json.loads(raw_value)]

    def append_event(self, thread_id: str, event: EventRecord) -> EventRecord:
        events = self.load_events(thread_id)
        next_sequence = len(events) + 1
        event_to_store = EventRecord(
            event_type=event.event_type,
            thread_id=thread_id,
            payload=event.payload,
            sequence=event.sequence or next_sequence,
            created_at=event.created_at,
        )
        events.append(event_to_store)
        redis_client.set(
            self._events_key(thread_id),
            json.dumps([item.to_dict() for item in events], ensure_ascii=False),
        )
        return event_to_store

    def save_snapshot(self, thread_id: str, snapshot: StateSnapshot) -> None:
        snapshots = self.load_snapshots(thread_id)
        snapshots.append(snapshot)
        redis_client.set(
            self._snapshots_key(thread_id),
            json.dumps([item.to_dict() for item in snapshots], ensure_ascii=False),
        )
        redis_client.set(
            self._snapshot_key(thread_id),
            json.dumps(snapshot.to_dict(), ensure_ascii=False),
        )

    def load_snapshots(self, thread_id: str) -> list[StateSnapshot]:
        raw_value = redis_client.get(self._snapshots_key(thread_id))
        if not raw_value:
            latest = self.load_latest_snapshot(thread_id)
            return [latest] if latest else []
        return [StateSnapshot.from_dict(item) for item in json.loads(raw_value)]

    def load_latest_snapshot(self, thread_id: str) -> StateSnapshot | None:
        raw_value = redis_client.get(self._snapshot_key(thread_id))
        if not raw_value:
            return None
        return StateSnapshot.from_dict(json.loads(raw_value))

    def save_assembly_record(self, thread_id: str, record: AssemblyRecord) -> None:
        records = self.load_assembly_records(thread_id)
        records.append(record)
        redis_client.set(
            self._assemblies_key(thread_id),
            json.dumps([item.to_dict() for item in records], ensure_ascii=False),
        )
        redis_client.set(
            self._assembly_key(thread_id),
            json.dumps(record.to_dict(), ensure_ascii=False),
        )

    def load_assembly_records(self, thread_id: str) -> list[AssemblyRecord]:
        raw_value = redis_client.get(self._assemblies_key(thread_id))
        if not raw_value:
            latest = self.load_latest_assembly_record(thread_id)
            return [latest] if latest else []
        return [AssemblyRecord.from_dict(item) for item in json.loads(raw_value)]

    def load_latest_assembly_record(self, thread_id: str) -> AssemblyRecord | None:
        raw_value = redis_client.get(self._assembly_key(thread_id))
        if not raw_value:
            return None
        return AssemblyRecord.from_dict(json.loads(raw_value))

    def save_context_blocks(self, thread_id: str, blocks: list[ContextBlock]) -> None:
        redis_client.set(
            self._blocks_key(thread_id),
            json.dumps([block.to_dict() for block in blocks], ensure_ascii=False),
        )

    def load_context_blocks(self, thread_id: str) -> list[ContextBlock]:
        raw_value = redis_client.get(self._blocks_key(thread_id))
        if not raw_value:
            return []
        return [ContextBlock.from_dict(item) for item in json.loads(raw_value)]

    def save_tool_evidence(self, thread_id: str, evidence: ToolEvidence) -> None:
        records = self.load_tool_evidence(thread_id)
        records.append(evidence)
        redis_client.set(
            self._tool_evidence_key(thread_id),
            json.dumps([item.to_dict() for item in records], ensure_ascii=False),
        )

    def load_tool_evidence(self, thread_id: str) -> list[ToolEvidence]:
        raw_value = redis_client.get(self._tool_evidence_key(thread_id))
        if not raw_value:
            return []
        return [ToolEvidence.from_dict(item) for item in json.loads(raw_value)]

    def save_model_call(self, thread_id: str, record: ModelCallRecord) -> None:
        records = self.load_model_calls(thread_id)
        records.append(record)
        redis_client.set(
            self._model_calls_key(thread_id),
            json.dumps([item.to_dict() for item in records], ensure_ascii=False),
        )

    def load_model_calls(self, thread_id: str) -> list[ModelCallRecord]:
        raw_value = redis_client.get(self._model_calls_key(thread_id))
        if not raw_value:
            return []
        return [ModelCallRecord.from_dict(item) for item in json.loads(raw_value)]

    def clear_model_calls(self, thread_id: str) -> None:
        redis_client.delete(self._model_calls_key(thread_id))

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
        redis_client.delete(self._events_key(thread_id))
        redis_client.delete(self._snapshot_key(thread_id))
        redis_client.delete(self._snapshots_key(thread_id))
        redis_client.delete(self._assembly_key(thread_id))
        redis_client.delete(self._assemblies_key(thread_id))
        redis_client.delete(self._blocks_key(thread_id))
        redis_client.delete(self._tool_evidence_key(thread_id))
        redis_client.delete(self._model_calls_key(thread_id))

    def _format_messages(self, messages: list[BaseMessage]) -> str:
        lines: list[str] = []

        for message in messages:
            role = getattr(message, "type", message.__class__.__name__)
            content = getattr(message, "content", "") or ""

            if role == "tool":
                tool_name = getattr(message, "name", "")
                content_preview = content[:100] + "..." if len(content) > 100 else content
                lines.append(f"[{role}] {tool_name}: {content_preview}")
            else:
                lines.append(f"[{role}] {content}")

        return "\n".join(lines)
