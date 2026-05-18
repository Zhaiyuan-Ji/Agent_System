from __future__ import annotations

from langchain_core.messages import BaseMessage

from .runtime_models import DerivedState, ToolEvidence


def find_latest_human_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            return (getattr(message, "content", "") or "").strip()
    return ""


def format_memory_item(message: BaseMessage) -> str:
    message_type = getattr(message, "type", "")
    content = (getattr(message, "content", "") or "").strip()
    if not content:
        return ""
    if message_type == "human":
        return f"User asked: {content[:180]}"
    if message_type == "ai":
        return f"Assistant answered: {content[:180]}"
    if message_type == "tool":
        tool_name = getattr(message, "name", "tool")
        return f"Tool evidence from {tool_name}: {content[:180]}"
    return content[:180]


def extract_message_evidence_refs(messages: list[BaseMessage]) -> list[str]:
    refs: list[str] = []
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        tool_name = getattr(message, "name", "tool")
        content = (getattr(message, "content", "") or "").strip()
        if content:
            refs.append(f"{tool_name}:{content[:120]}")
    return refs[-5:]


def format_tool_evidence_refs(tool_evidence: list[ToolEvidence]) -> list[str]:
    refs: list[str] = []
    for evidence in tool_evidence[-5:]:
        preview = (evidence.preview or evidence.content or "").strip()
        if preview:
            refs.append(f"{evidence.tool_name}:{preview[:160]}")
    return refs


def build_derived_state(
    thread_id: str,
    latest_request: str,
    context_messages: list[BaseMessage],
    summary_text: str,
    tool_evidence: list[ToolEvidence],
) -> DerivedState:
    memory_items = [
        item
        for item in (format_memory_item(message) for message in context_messages[-6:])
        if item
    ]
    evidence_refs = [
        *extract_message_evidence_refs(context_messages),
        *format_tool_evidence_refs(tool_evidence),
    ][-5:]
    open_loops: list[str] = []
    if latest_request:
        open_loops.append("Need produce a cited final answer for the current request.")
    if latest_request and not evidence_refs:
        open_loops.append("Need collect or verify external evidence before finalizing.")

    return DerivedState(
        thread_id=thread_id,
        current_goal=latest_request,
        working_memory=memory_items[-4:],
        confirmed_facts=[summary_text] if summary_text else [],
        active_constraints=["Answer from retrieved evidence and include citation metadata"] if latest_request else [],
        open_loops=open_loops,
        active_evidence_refs=evidence_refs,
        recent_context_summary=summary_text,
    )
