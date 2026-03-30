"""
Context Configuration Module

提供环境配置和系统提示词。
"""

from __future__ import annotations

import os

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:54329/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
CHAT_MODE = os.getenv("CHAT_MODE", "openai").lower()

os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def get_agent_settings() -> dict[str, str]:
    return {
        "mode": CHAT_MODE,
        "model": OPENAI_MODEL,
        "base_url": OPENAI_BASE_URL,
    }
