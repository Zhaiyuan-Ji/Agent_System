"""
Context Module

提供上下文管理和中间件功能。
"""

from .schema import AgentContext
from .manager import RedisContextManager
from .config import (
    get_agent_settings,
    OPENAI_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from .context_service import (
    get_prompt_context,
    serialize_message,
    context_manager,
)
from .middleware import (
    ChatAgentState,
    ContextInjectMiddleware,
    inject_context,
    log_after_model,
)
from Agent.System_prompt import SYSTEM_PROMPT

__all__ = [
    "AgentContext",
    "RedisContextManager",
    "SYSTEM_PROMPT",
    "get_agent_settings",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "get_prompt_context",
    "serialize_message",
    "context_manager",
    "ChatAgentState",
    "ContextInjectMiddleware",
    "inject_context",
    "log_after_model",
]
