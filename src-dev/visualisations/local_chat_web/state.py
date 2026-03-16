"""In-memory session state for browser local chat."""

from __future__ import annotations

from scripts.visualisations.local_chat_web.types import (
    ChatAdapterConfig,
    ChatSession,
    ChatTurn,
    GenerationSettings,
    utc_now_iso,
)


def clone_adapter_config(adapters: list[ChatAdapterConfig]) -> list[ChatAdapterConfig]:
    """Clone adapter config values to avoid shared mutable state."""
    return [
        ChatAdapterConfig(key=adapter.key, path=adapter.path, scale=float(adapter.scale))
        for adapter in adapters
    ]


class SessionStore:
    """Simple in-memory chat session store scoped to one process."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._chat_order: list[str] = []
        self._active_chat_id: str | None = None
        self._counter = 0

    def list_sessions(self) -> list[ChatSession]:
        """Return sessions in UI display order."""
        return [self._sessions[chat_id] for chat_id in self._chat_order]

    def get_session(self, chat_id: str) -> ChatSession:
        """Return one chat session by id."""
        if chat_id not in self._sessions:
            raise KeyError(f"Unknown chat id: {chat_id}")
        return self._sessions[chat_id]

    def get_active_session(self) -> ChatSession:
        """Return the active chat session."""
        if self._active_chat_id is None:
            raise ValueError("No active chat session")
        return self.get_session(self._active_chat_id)

    def create_chat(
        self,
        *,
        initial_adapters: list[ChatAdapterConfig] | None = None,
        title: str | None = None,
    ) -> ChatSession:
        """Create and select a new chat session."""
        adapters = clone_adapter_config(initial_adapters or [])
        now = utc_now_iso()
        chat_id = f"chat-{self._counter + 1:04d}"
        default_title = f"Chat {self._counter + 1}"

        session = ChatSession(
            chat_id=chat_id,
            title=title or default_title,
            created_at=now,
            updated_at=now,
            created_adapter_config=clone_adapter_config(adapters),
            current_adapter_config=clone_adapter_config(adapters),
        )

        self._counter += 1
        self._sessions[chat_id] = session
        self._chat_order.append(chat_id)
        self._active_chat_id = chat_id
        return session

    def select_chat(self, chat_id: str) -> ChatSession:
        """Switch active chat session."""
        session = self.get_session(chat_id)
        self._active_chat_id = chat_id
        return session

    def set_current_adapters(self, chat_id: str, adapters: list[ChatAdapterConfig]) -> None:
        """Replace the active adapter configuration for a chat."""
        session = self.get_session(chat_id)
        session.current_adapter_config = clone_adapter_config(adapters)
        session.updated_at = utc_now_iso()

    def upsert_adapter(self, chat_id: str, adapter: ChatAdapterConfig) -> ChatSession:
        """Insert or update one adapter in a chat config."""
        session = self.get_session(chat_id)
        updated: list[ChatAdapterConfig] = []
        replaced = False

        for existing in session.current_adapter_config:
            if existing.key == adapter.key:
                updated.append(
                    ChatAdapterConfig(
                        key=adapter.key,
                        path=adapter.path,
                        scale=float(adapter.scale),
                    )
                )
                replaced = True
            else:
                updated.append(existing)

        if not replaced:
            updated.append(
                ChatAdapterConfig(
                    key=adapter.key,
                    path=adapter.path,
                    scale=float(adapter.scale),
                )
            )

        self.set_current_adapters(chat_id, updated)
        return session

    def remove_adapter(self, chat_id: str, adapter_key: str) -> ChatSession:
        """Remove an adapter from chat config by key."""
        session = self.get_session(chat_id)
        filtered = [a for a in session.current_adapter_config if a.key != adapter_key]
        self.set_current_adapters(chat_id, filtered)
        return session

    def set_adapter_scale(self, chat_id: str, adapter_key: str, scale: float) -> ChatSession:
        """Update adapter scaling for one configured adapter."""
        session = self.get_session(chat_id)
        updated: list[ChatAdapterConfig] = []
        found = False

        for adapter in session.current_adapter_config:
            if adapter.key == adapter_key:
                updated.append(
                    ChatAdapterConfig(
                        key=adapter.key,
                        path=adapter.path,
                        scale=float(scale),
                    )
                )
                found = True
            else:
                updated.append(adapter)

        if not found:
            raise KeyError(f"Adapter '{adapter_key}' is not active in chat {chat_id}")

        self.set_current_adapters(chat_id, updated)
        return session

    def append_turn(
        self,
        chat_id: str,
        *,
        user_text: str,
        assistant_text: str,
        generation_settings: GenerationSettings,
    ) -> ChatTurn:
        """Append one exchange and snapshot config/settings used for it."""
        session = self.get_session(chat_id)
        turn = ChatTurn(
            user_text=user_text,
            assistant_text=assistant_text,
            timestamp=utc_now_iso(),
            adapter_config_snapshot=clone_adapter_config(session.current_adapter_config),
            generation_settings_snapshot=GenerationSettings(
                max_new_tokens=generation_settings.max_new_tokens,
                temperature=generation_settings.temperature,
                top_p=generation_settings.top_p,
            ),
        )
        session.turns.append(turn)
        session.updated_at = turn.timestamp

        if len(session.turns) == 1 and session.title.startswith("Chat "):
            trimmed = user_text.strip().replace("\n", " ")
            if trimmed:
                session.title = (trimmed[:48] + "...") if len(trimmed) > 48 else trimmed

        return turn
