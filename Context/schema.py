from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentContext:
    thread_id: str
    token_budget: int = 120000
    loaded_message_count: int = 0
