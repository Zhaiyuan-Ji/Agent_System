"""
Helpers for previewing, assembling, and recording the effective runtime context.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from Agent.System_prompt import SYSTEM_PROMPT
from Context.runtime_models import (
    AssemblyRecord,
    ContextBlock,
    DerivedState,
    EventRecord,
    StateSnapshot,
)

from .context_selector import build_context_blocks
from .manager import RedisContextManager
from .runtime_assembly import assemble_context_blocks
from .state_reducer import build_derived_state, find_latest_human_message
from .trace_recorder import record_context_block_events

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
        "content": getattr(message, "content", "") or "",
    }


def serialize_state(state: DerivedState | None) -> dict[str, Any] | None:
    return state.to_dict() if state else None


def serialize_context_block(block: ContextBlock) -> dict[str, Any]:
    return block.to_dict()


def serialize_event(event: EventRecord) -> dict[str, Any]:
    return event.to_dict()


def serialize_assembly_record(record: AssemblyRecord | None) -> dict[str, Any] | None:
    return record.to_dict() if record else None


def refresh_runtime_artifacts(
    thread_id: str,
    draft_message: str = "",
    token_budget: int = 120000,
) -> dict[str, Any]:
    context_messages = context_manager.load_context_messages(thread_id)
    summary_text = context_manager.load_summary(thread_id)
    tool_evidence = context_manager.load_tool_evidence(thread_id)
    latest_request = draft_message.strip() or find_latest_human_message(context_messages)

    derived_state = build_derived_state(
        thread_id=thread_id,
        latest_request=latest_request,
        context_messages=context_messages,
        summary_text=summary_text,
        tool_evidence=tool_evidence,
    )

    state_event = context_manager.append_event(
        thread_id,
        EventRecord(
            event_type="state_updated",
            thread_id=thread_id,
            payload={
                "current_goal": derived_state.current_goal,
                "working_memory_items": len(derived_state.working_memory),
                "open_loops": len(derived_state.open_loops),
                "active_evidence_refs": len(derived_state.active_evidence_refs),
            },
        ),
    )

    snapshot = StateSnapshot(thread_id=thread_id, sequence=state_event.sequence, state=derived_state)
    context_manager.save_snapshot(thread_id, snapshot)

    candidate_blocks = build_context_blocks(
        thread_id,
        draft_message,
        context_messages,
        summary_text,
        tool_evidence,
    )
    selected_blocks, dropped_blocks = assemble_context_blocks(candidate_blocks, token_budget=token_budget)
    selected_ids = {block.block_id for block in selected_blocks}
    dropped_by_id = {item["block_id"]: item["reason"] for item in dropped_blocks}

    persisted_blocks = [
        ContextBlock(
            **{
                **block.to_dict(),
                "selected": block.block_id in selected_ids,
                "drop_reason": dropped_by_id.get(block.block_id),
            }
        )
        for block in candidate_blocks
    ]

    context_manager.save_context_blocks(thread_id, persisted_blocks)
    block_events = record_context_block_events(context_manager, thread_id, persisted_blocks)

    assembly_event = context_manager.append_event(
        thread_id,
        EventRecord(
            event_type="assembly_completed",
            thread_id=thread_id,
            payload={
                "selected_blocks": [block.block_id for block in selected_blocks],
                "dropped_blocks": dropped_blocks,
            },
        ),
    )

    estimated_total_tokens = sum(block.estimated_tokens for block in selected_blocks)
    assembly_record = AssemblyRecord(
        thread_id=thread_id,
        sequence=assembly_event.sequence,
        selected_blocks=[block.block_id for block in selected_blocks],
        dropped_blocks=dropped_blocks,
        token_budget=token_budget,
        estimated_total_tokens=estimated_total_tokens,
        payload_preview=" | ".join(block.title for block in selected_blocks),
        context_blocks=[block.to_dict() for block in persisted_blocks],
    )
    context_manager.save_assembly_record(thread_id, assembly_record)

    return {
        "state": derived_state,
        "snapshot": snapshot,
        "context_blocks": persisted_blocks,
        "assembly": assembly_record,
        "events": [state_event, *block_events, assembly_event],
    }


def split_context_sections(thread_id: str, draft_message: str = "") -> dict[str, Any]:
    system_message = SystemMessage(content=SYSTEM_PROMPT, id="system:agent")
    context_messages = context_manager.load_context_messages(thread_id)
    snapshots = context_manager.load_snapshots(thread_id)
    latest_snapshot = context_manager.load_latest_snapshot(thread_id)
    latest_state = latest_snapshot.state if latest_snapshot else None
    latest_assembly = context_manager.load_latest_assembly_record(thread_id)
    runtime_assemblies = context_manager.load_assembly_records(thread_id)
    model_calls = context_manager.load_model_calls(thread_id)
    context_blocks = context_manager.load_context_blocks(thread_id)
    runtime_events = context_manager.load_events(thread_id)
    tool_evidence = context_manager.load_tool_evidence(thread_id)

    summary_messages = [
        message
        for message in context_messages
        if getattr(message, "type", "") == "system"
        and str(getattr(message, "id", "")).startswith("summary:")
    ]
    history_messages = [message for message in context_messages if message not in summary_messages]

    latest_user_index = -1
    latest_user_target = draft_message.strip()
    for index in range(len(history_messages) - 1, -1, -1):
        if getattr(history_messages[index], "type", "") != "human":
            continue
        latest_user_index = index
        if latest_user_target and history_messages[index].content.strip() == latest_user_target:
            break
        if not latest_user_target:
            break

    current_question_msg = (
        serialize_message(history_messages[latest_user_index]) if latest_user_index >= 0 else None
    )

    current_answer_messages: list[dict[str, Any]] = []
    for index in range(latest_user_index + 1, len(history_messages)):
        if getattr(history_messages[index], "type", "") == "ai":
            current_answer_messages.append(serialize_message(history_messages[index]))
            break

    from .config import CHAT_MODE, OPENAI_BASE_URL, OPENAI_MODEL

    sections: list[dict[str, Any]] = [
        {
            "key": "system",
            "title": "System Prompt",
            "description": "Base behavior rules and retrieval constraints for the current agent.",
            "messages": [serialize_message(system_message)],
        },
    ]

    if latest_state:
        sections.append(
            {
                "key": "state",
                "title": "Derived State",
                "description": "Structured state reduced from runtime events and context history.",
                "messages": [
                    {
                        "id": f"state:{thread_id}",
                        "type": "state",
                        "role": "state",
                        "content": latest_state.to_dict(),
                    }
                ],
            }
        )

    if context_blocks:
        sections.append(
            {
                "key": "context_blocks",
                "title": "Context Blocks",
                "description": "Structured blocks considered during context assembly.",
                "messages": [
                    {
                        "id": block.block_id,
                        "type": "context_block",
                        "role": "context_block",
                        "content": block.to_dict(),
                    }
                    for block in context_blocks
                ],
            }
        )

    if tool_evidence:
        sections.append(
            {
                "key": "tool_evidence",
                "title": "Tool Evidence",
                "description": "Structured evidence captured from tool results.",
                "messages": [
                    {
                        "id": evidence.evidence_id,
                        "type": "tool_evidence",
                        "role": "tool",
                        "content": evidence.to_dict(),
                    }
                    for evidence in tool_evidence
                ],
            }
        )

    if summary_messages:
        sections.append(
            {
                "key": "summary",
                "title": "Historical Summary",
                "description": "Compressed background retained from earlier conversation turns.",
                "messages": [serialize_message(message) for message in summary_messages],
            }
        )

    if history_messages:
        sections.append(
            {
                "key": "past_context",
                "title": "Recent Messages",
                "description": "Recent message window retained for the current thread.",
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
        "state": serialize_state(latest_state),
        "assembly": serialize_assembly_record(latest_assembly),
        "runtime_assemblies": [record.to_dict() for record in runtime_assemblies],
        "model_calls": [record.to_dict() for record in model_calls],
        "context_blocks": [serialize_context_block(block) for block in context_blocks],
        "tool_evidence": [evidence.to_dict() for evidence in tool_evidence],
        "runtime_events": [serialize_event(event) for event in runtime_events],
        "runtime_snapshots": [snapshot.to_dict() for snapshot in snapshots],
        "sections": sections,
    }


def get_prompt_context(thread_id: str, draft_message: str = "") -> dict[str, Any]:
    return split_context_sections(thread_id=thread_id, draft_message=draft_message)
