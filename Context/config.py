"""
Context Configuration Module

提供环境配置和系统提示词。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILES = [
    REPO_ROOT / ".env.local",
    REPO_ROOT / ".env",
]

explicit_env_file = os.getenv("AGENT_SYSTEM_ENV_FILE", "").strip()
if explicit_env_file:
    load_dotenv(explicit_env_file, override=False)
else:
    for env_file in DEFAULT_ENV_FILES:
        if env_file.exists():
            load_dotenv(env_file, override=False)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.minimaxi.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "MiniMax-M2.7")
CHAT_MODE = os.getenv("CHAT_MODE", "openai").lower()

os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def get_agent_settings() -> dict[str, str]:
    return {
        "mode": CHAT_MODE,
        "model": OPENAI_MODEL,
        "base_url": OPENAI_BASE_URL,
    }
