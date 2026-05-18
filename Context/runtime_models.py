from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EventRecord:
    event_type: str
    thread_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventRecord":
        return cls(**data)


@dataclass
class ContextBlock:
    block_id: str
    block_type: str
    title: str
    content: str
    source: str
    priority: int
    freshness: str = "active"
    estimated_tokens: int = 0
    selected: bool = False
    drop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextBlock":
        return cls(**data)


@dataclass
class DerivedState:
    thread_id: str
    current_goal: str = ""
    working_memory: list[str] = field(default_factory=list)
    confirmed_facts: list[str] = field(default_factory=list)
    active_constraints: list[str] = field(default_factory=list)
    open_loops: list[str] = field(default_factory=list)
    active_evidence_refs: list[str] = field(default_factory=list)
    recent_context_summary: str = ""
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DerivedState":
        return cls(**data)


@dataclass
class StateSnapshot:
    thread_id: str
    sequence: int
    state: DerivedState
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateSnapshot":
        return cls(
            thread_id=data["thread_id"],
            sequence=data["sequence"],
            state=DerivedState.from_dict(data["state"]),
            created_at=data.get("created_at", utc_now_iso()),
        )


@dataclass
class ToolEvidence:
    evidence_id: str
    thread_id: str
    tool_name: str
    content: str
    preview: str
    sequence: int
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolEvidence":
        return cls(**data)


@dataclass
class AssemblyRecord:
    thread_id: str
    sequence: int
    selected_blocks: list[str]
    dropped_blocks: list[dict[str, Any]]
    token_budget: int
    estimated_total_tokens: int
    payload_preview: str
    context_blocks: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssemblyRecord":
        return cls(**data)


@dataclass
class ModelCallRecord:
    thread_id: str
    call_id: str
    call_index: int
    sequence: int
    phase: str
    purpose: str
    input_context: list[dict[str, Any]]
    state_snapshot: dict[str, Any] | None = None
    output: dict[str, Any] = field(default_factory=dict)
    raw_think: str | None = None
    started_at: str = field(default_factory=utc_now_iso)
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCallRecord":
        return cls(**data)
