"""Types for browser-based local chat sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return a UTC timestamp string for chat/session events."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class AdapterCatalogEntry:
    """One curated adapter option exposed in the browser UI."""

    key: str
    name: str
    path: str
    description: str = ""


@dataclass(frozen=True)
class ChatAdapterConfig:
    """Adapter configuration for a chat session."""

    key: str
    path: str
    scale: float = 1.0


@dataclass(frozen=True)
class GenerationSettings:
    """Generation parameters snapshot used for one reply."""

    max_new_tokens: int
    temperature: float
    top_p: float


@dataclass
class ChatTurn:
    """One user->assistant exchange in a chat."""

    user_text: str
    assistant_text: str
    timestamp: str
    adapter_config_snapshot: list[ChatAdapterConfig]
    generation_settings_snapshot: GenerationSettings


@dataclass
class ChatSession:
    """In-memory chat session state for one browser process."""

    chat_id: str
    title: str
    created_at: str
    updated_at: str
    created_adapter_config: list[ChatAdapterConfig]
    current_adapter_config: list[ChatAdapterConfig]
    turns: list[ChatTurn] = field(default_factory=list)


@dataclass(frozen=True)
class BrowserChatConfig:
    """Model/runtime configuration for the browser app."""

    base_model: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    prompt_format: str = "auto"
    system_prompt: str = "You are a helpful assistant."
    tone: str = "balanced"
    history_window: int = 24
    seed: int | None = None
