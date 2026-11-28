"""OpenAI-compatible client configuration for Magpie, OpenRouter, or local servers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class Settings:
    """Connection settings for an OpenAI-compatible chat endpoint."""

    api_key: str
    base_url: str
    model: str


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_settings(env_file: Path | None = None) -> Settings:
    """Read OPENAI_* (or MAGPIE_GATEWAY_KEY + OPENAI_BASE_URL) from env / untracked .env."""
    if env_file is None:
        root = Path(__file__).resolve().parents[4]
        _load_dotenv(root / ".env.local")
        _load_dotenv(root / ".env")
    else:
        _load_dotenv(env_file)

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("MAGPIE_GATEWAY_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError(
            "Neither OPENAI_API_KEY nor MAGPIE_GATEWAY_KEY is set. "
            "Export it or copy packages/langchain-lab/.env.example to .env."
        )

    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").strip()
    model = os.environ.get("OPENAI_MODEL", "openrouter/free").strip()
    return Settings(api_key=api_key, base_url=base_url, model=model)


def build_chat_model(settings: Settings | None = None, temperature: float = 0.2) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI client pointed at the configured endpoint."""
    cfg = settings or load_settings()
    return ChatOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=temperature,
    )
