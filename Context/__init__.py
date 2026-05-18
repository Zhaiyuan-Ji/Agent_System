"""
Lazy exports for the Context package.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentContext",
    "RedisContextManager",
    "SYSTEM_PROMPT",
    "get_agent_settings",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "get_prompt_context",
    "refresh_runtime_artifacts",
    "serialize_message",
    "context_manager",
    "EventRecord",
    "ContextBlock",
    "DerivedState",
    "StateSnapshot",
    "AssemblyRecord",
    "ToolEvidence",
    "ModelCallRecord",
    "build_derived_state",
    "build_context_blocks",
    "record_context_block_events",
    "ChatAgentState",
    "ContextInjectMiddleware",
    "inject_context",
    "log_after_model",
]


def __getattr__(name: str) -> Any:
    if name == "AgentContext":
        from .schema import AgentContext

        return AgentContext

    if name == "RedisContextManager":
        from .manager import RedisContextManager

        return RedisContextManager

    if name in {"get_agent_settings", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"}:
        from . import config as config_module

        return getattr(config_module, name)

    if name in {"get_prompt_context", "refresh_runtime_artifacts", "serialize_message", "context_manager"}:
        from . import context_service as service_module

        return getattr(service_module, name)

    if name in {
        "EventRecord",
        "ContextBlock",
        "DerivedState",
        "StateSnapshot",
        "AssemblyRecord",
        "ToolEvidence",
        "ModelCallRecord",
    }:
        from . import runtime_models as runtime_models_module

        return getattr(runtime_models_module, name)

    if name == "build_derived_state":
        from .state_reducer import build_derived_state

        return build_derived_state

    if name == "build_context_blocks":
        from .context_selector import build_context_blocks

        return build_context_blocks

    if name == "record_context_block_events":
        from .trace_recorder import record_context_block_events

        return record_context_block_events

    if name in {"ChatAgentState", "ContextInjectMiddleware", "inject_context", "log_after_model"}:
        from . import middleware as middleware_module

        return getattr(middleware_module, name)

    if name == "SYSTEM_PROMPT":
        from Agent.System_prompt import SYSTEM_PROMPT

        return SYSTEM_PROMPT

    raise AttributeError(name)
