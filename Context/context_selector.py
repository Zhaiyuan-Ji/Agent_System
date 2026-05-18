from __future__ import annotations

from langchain_core.messages import BaseMessage

from Agent.System_prompt import SYSTEM_PROMPT

from .runtime_models import ContextBlock, ToolEvidence
from .state_reducer import (
    extract_message_evidence_refs,
    find_latest_human_message,
    format_tool_evidence_refs,
)


def estimate_tokens(text: str, maximum: int) -> int:
    if not text:
        return 0
    rough = max(len(text) // 4, 1)
    return min(rough, maximum)


def build_context_blocks(
    thread_id: str,
    draft_message: str,
    context_messages: list[BaseMessage],
    summary_text: str,
    tool_evidence: list[ToolEvidence],
) -> list[ContextBlock]:
    latest_request = draft_message.strip() or find_latest_human_message(context_messages)
    recent_messages = context_messages[-3:]
    blocks: list[ContextBlock] = [
        ContextBlock(
            block_id="system:agent",
            block_type="system_rules",
            title="System Rules",
            content=SYSTEM_PROMPT,
            source="system",
            priority=100,
            estimated_tokens=estimate_tokens(SYSTEM_PROMPT, 32),
        )
    ]

    if latest_request:
        blocks.extend(
            [
                ContextBlock(
                    block_id=f"goal:{thread_id}",
                    block_type="task_goal",
                    title="Current Goal",
                    content=latest_request,
                    source="state",
                    priority=90,
                    estimated_tokens=estimate_tokens(latest_request, 16),
                ),
                ContextBlock(
                    block_id=f"request:{thread_id}",
                    block_type="current_user_request",
                    title="Current User Request",
                    content=latest_request,
                    source="message",
                    priority=85,
                    estimated_tokens=estimate_tokens(latest_request, 16),
                ),
            ]
        )

    if summary_text:
        blocks.append(
            ContextBlock(
                block_id=f"summary:{thread_id}",
                block_type="historical_summary",
                title="Historical Summary",
                content=summary_text,
                source="summary",
                priority=40,
                estimated_tokens=estimate_tokens(summary_text, 24),
            )
        )

    if recent_messages:
        recent_text = "\n".join(
            f"[{getattr(message, 'type', '')}] {(getattr(message, 'content', '') or '').strip()}"
            for message in recent_messages
        )
        blocks.append(
            ContextBlock(
                block_id=f"recent:{thread_id}",
                block_type="recent_dialogue",
                title="Recent Dialogue",
                content=recent_text,
                source="messages",
                priority=60,
                estimated_tokens=estimate_tokens(recent_text, 24),
            )
        )

    evidence_refs = [
        *extract_message_evidence_refs(context_messages),
        *format_tool_evidence_refs(tool_evidence),
    ][-5:]
    if evidence_refs:
        blocks.append(
            ContextBlock(
                block_id=f"evidence:{thread_id}",
                block_type="tool_evidence",
                title="Tool Evidence",
                content="\n".join(evidence_refs),
                source="tools",
                priority=80,
                estimated_tokens=estimate_tokens("\n".join(evidence_refs), 32),
                metadata={"refs": evidence_refs},
            )
        )

    return blocks
