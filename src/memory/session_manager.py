"""多对话会话记忆管理。"""

from __future__ import annotations

from agentscope.memory import InMemoryMemory

from src.config import Settings
from src.memory.layered_session_memory import LayeredSessionMemory


class SessionManager:
    """按 session_id 隔离短期对话记忆（分层上下文窗口）。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._sessions: dict[str, LayeredSessionMemory] = {}

    def get_memory(self, session_id: str) -> LayeredSessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = LayeredSessionMemory(self.settings)
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        self._sessions[session_id] = LayeredSessionMemory(self.settings)

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())
