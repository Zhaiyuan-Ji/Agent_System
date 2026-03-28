"""
Agent Module

提供 AI Agent 实例。
"""

from __future__ import annotations

from langchain.agents import create_agent 
from langchain.chat_models import init_chat_model

from Context import (
    AgentContext,
    SYSTEM_PROMPT,
    ContextInjectMiddleware,
    inject_context,
    log_after_model,
)
from Context.config import OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL


def create_chat_agent() -> any:
    model = init_chat_model(
        model=OPENAI_MODEL,
        model_provider="openai",
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        temperature=0.7,
    )

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        context_schema=AgentContext,
        middleware=[
            inject_context,
            ContextInjectMiddleware(),
            log_after_model,
        ],
    )

    return agent


agent = create_chat_agent()
