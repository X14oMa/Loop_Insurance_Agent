"""保险主 Agent：两层上下文窗口管理。"""

from __future__ import annotations

from agentscope.agent import ReActAgent

from src.memory.layered_session_memory import LayeredSessionMemory


class InsuranceReActAgent(ReActAgent):
    """扩展 ReActAgent：用分层会话记忆替代内置字符阈值压缩。"""

    async def _compress_memory_if_needed(self) -> None:
        """每轮 ReAct 迭代前检查 93%/window-13k 硬压缩。"""
        if isinstance(self.memory, LayeredSessionMemory):
            if hasattr(self, "_sys_prompt"):
                self.memory.set_agent_context(
                    sys_prompt=self.sys_prompt,
                    model_name=getattr(self.model, "model_name", ""),
                )
            await self.memory.on_iteration_start()
            return
        await super()._compress_memory_if_needed()
