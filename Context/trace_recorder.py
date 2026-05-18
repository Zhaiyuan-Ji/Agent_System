from __future__ import annotations

from typing import Protocol

from .runtime_models import ContextBlock, EventRecord


class EventStore(Protocol):
    def append_event(self, thread_id: str, event: EventRecord) -> EventRecord:
        ...


def record_context_block_events(
    context_manager: EventStore,
    thread_id: str,
    blocks: list[ContextBlock],
) -> list[EventRecord]:
    events: list[EventRecord] = []
    for block in blocks:
        if block.selected:
            event_type = "context_block_selected"
            payload = {
                "block_id": block.block_id,
                "block_type": block.block_type,
                "priority": block.priority,
                "estimated_tokens": block.estimated_tokens,
            }
        elif block.drop_reason:
            event_type = "context_block_dropped"
            payload = {
                "block_id": block.block_id,
                "block_type": block.block_type,
                "reason": block.drop_reason,
                "priority": block.priority,
                "estimated_tokens": block.estimated_tokens,
            }
        else:
            continue

        events.append(
            context_manager.append_event(
                thread_id,
                EventRecord(event_type=event_type, thread_id=thread_id, payload=payload),
            )
        )
    return events
