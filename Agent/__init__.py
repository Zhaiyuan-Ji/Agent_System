"""
Agent Module

Provides lazy access to the chat agent to avoid import cycles during setup.
"""

from __future__ import annotations

from typing import Any

__all__ = ["agent", "create_chat_agent"]


def create_chat_agent() -> Any:
    from .agent import create_chat_agent as _create_chat_agent

    return _create_chat_agent()


def __getattr__(name: str) -> Any:
    if name == "agent":
        from .agent import agent as _agent

        return _agent

    if name == "create_chat_agent":
        return create_chat_agent

    raise AttributeError(name)
