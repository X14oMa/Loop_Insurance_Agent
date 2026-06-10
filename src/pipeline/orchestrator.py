"""Insurance Agent Pipeline：完整工作流编排。"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from agentscope.message import Msg
from agentscope.pipeline import stream_printing_messages

from src.agent import InsuranceAgentFactory
from src.agent.insurance_react_agent import InsuranceReActAgent
from src.compliance import ComplianceMeta, ComplianceValidator
from src.config import Settings
from src.memory import MemoryManager
from src.memory.layered_session_memory import LayeredSessionMemory
from src.memory.session_manager import SessionManager
from src.pipeline.agent_router import TurnConfig, resolve_turn_config
from src.pipeline.streaming import msg_to_stream_events, reset_stream_state
from src.model.chat_model_factory import create_chat_model
from src.model.embedding_factory import create_embedding_model
from src.rag.hybrid_knowledge import HybridKnowledge
from src.rag.document_loader import _sanitize_filename, load_pdf_bytes
from src.rag.parent_store import ParentChunkStore
from src.rag.policy_manifest import PolicyManifest
from src.rag.retrieval_config import RetrievalConfig
from src.rag.tool_response_limits import ToolResponseLimits
from src.rag.vector_store import create_vector_store


class InsuranceAgentPipeline:
    """Insurance Agent 完整 Pipeline。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.knowledge: HybridKnowledge | None = None
        self.memory_manager: MemoryManager | None = None
        self.agent: InsuranceReActAgent | None = None
        self.compliance = ComplianceValidator(settings)
        self.session_manager = SessionManager(settings)
        self.manifest: PolicyManifest | None = None

    async def initialize(self) -> None:
        """初始化知识库、记忆模块和主 Agent（不扫描本地文档目录）。"""
        self.settings.validate()

        embedding_model = create_embedding_model(self.settings)

        vector_store = create_vector_store(self.settings)

        parent_store_path = self.settings.vector_store_path / "parent_chunks.json"
        parent_store = ParentChunkStore(parent_store_path)
        retrieval_config = RetrievalConfig()
        tool_response_limits = ToolResponseLimits(
            max_chunks=self.settings.rag_tool_limit,
            max_chunk_chars=self.settings.rag_max_chunk_display_chars,
            max_total_chars=self.settings.rag_max_tool_response_chars,
        )

        self.knowledge = HybridKnowledge(
            embedding_model=embedding_model,
            embedding_store=vector_store,
            parent_store=parent_store,
            retrieval_config=retrieval_config,
            tool_response_limits=tool_response_limits,
        )
        restored = await self.knowledge.restore_bm25_from_vector_store()

        manifest_path = self.settings.vector_store_path / "policy_manifest.json"
        self.manifest = PolicyManifest(manifest_path)

        self.memory_manager = MemoryManager(self.settings)
        self.agent = InsuranceAgentFactory.create(
            settings=self.settings,
            knowledge=self.knowledge,
            memory_manager=self.memory_manager,
            compliance=self.compliance,
            policy_context_provider=self._format_policy_library_context,
        )

        if restored:
            import logging

            logging.getLogger(__name__).info(
                "已从向量库恢复 %d 个条款片段，跳过启动时 re-index。",
                restored,
            )

    def _format_policy_library_context(self) -> str | None:
        policies = self.list_policies()
        if not policies:
            return (
                "<policy_library>\n"
                "当前知识库为空，尚未上传任何保单 PDF。\n"
                "</policy_library>"
            )

        lines = [
            "当前知识库中已入库的保单（调用 retrieve_knowledge / delegate_subagents 时 doc_id 请使用下列 filename）：",
        ]
        for index, policy in enumerate(policies, start=1):
            lines.append(
                f"{index}. {policy['filename']}（{policy['chunks']} 片段，"
                f"{policy['size_kb']} KB）",
            )
        if len(policies) == 1:
            lines.append(
                "用户提及「这份保单/本保单」且未指定名称时，默认指上述唯一保单。",
            )
        return "<policy_library>\n" + "\n".join(lines) + "\n</policy_library>"

    async def _build_enriched_input(
        self,
        user_input: str,
        *,
        max_retrievals: int | None = None,
    ) -> str:
        sections: list[str] = []

        policy_context = self._format_policy_library_context()
        if policy_context:
            sections.append(policy_context)

        sections.append(
            "<current_turn_only>\n"
            "以下「本轮用户问题」是你唯一需要回答的内容。"
            "会话历史（含 delegate_subagents 的工具结果）仅供理解指代；"
            "除非用户明确提及「刚才/上次/之前」，"
            "否则不要重复回答历史中已经解决的其他问题。\n"
            "</current_turn_only>",
        )

        if max_retrievals is not None:
            sections.append(
                f"<retrieval_budget>本轮最多调用 retrieve_knowledge "
                f"{max_retrievals} 次；多子问题可调用 delegate_subagents（返回已含引用短文）。"
                f"</retrieval_budget>",
            )

        sections.append(f"【本轮用户问题】\n{user_input}")
        return "\n\n".join(sections)

    def _configure_main_agent_for_turn(self, turn: TurnConfig) -> None:
        """按轮次配置主 Agent 检索深度与工具。"""
        if not self.agent or not self.knowledge:
            raise RuntimeError("Pipeline 尚未初始化。")

        InsuranceAgentFactory.apply_main_agent_turn_config(
            self.agent,
            self.knowledge,
            self.settings,
            self.compliance,
            turn.max_retrievals,
            self._format_policy_library_context,
        )

    async def _resolve_turn(self, user_input: str) -> TurnConfig:
        return await resolve_turn_config(user_input, self.settings)

    async def chat(
        self,
        user_input: str,
        session_id: str = "default",
    ) -> tuple[str, ComplianceMeta]:
        if not self.agent or not self.memory_manager or not self.knowledge:
            raise RuntimeError("Pipeline 尚未初始化。")

        self._bind_session(session_id)
        self._sync_session_attachments(session_id)
        plain_user_msg = Msg("User", user_input, "user")
        turn = await self._resolve_turn(user_input)

        self._configure_main_agent_for_turn(turn)
        enriched_input = await self._build_enriched_input(
            user_input,
            max_retrievals=turn.max_retrievals,
        )
        user_msg = Msg("User", enriched_input, "user")
        response = await self.agent(user_msg)

        response, compliance_meta = self.compliance.process(response)
        await self.memory_manager.update_after_response(plain_user_msg, response)
        return response.get_text_content() or "", compliance_meta

    async def chat_stream(
        self,
        user_input: str,
        session_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        if not self.agent or not self.memory_manager or not self.knowledge:
            raise RuntimeError("Pipeline 尚未初始化。")

        self._bind_session(session_id)
        self._sync_session_attachments(session_id)
        plain_user_msg = Msg("User", user_input, "user")
        turn = await self._resolve_turn(user_input)

        self._configure_main_agent_for_turn(turn)
        enriched_input = await self._build_enriched_input(
            user_input,
            max_retrievals=turn.max_retrievals,
        )
        user_msg = Msg("User", enriched_input, "user")

        response_holder: dict[str, Msg] = {}

        async def run_agent() -> Msg:
            result = await self.agent(user_msg)  # type: ignore[misc]
            response_holder["response"] = result
            return result

        thinking_parts: list[str] = []
        reset_stream_state()

        async for msg, last in stream_printing_messages([self.agent], run_agent()):
            for event in msg_to_stream_events(msg, last):
                if event["type"] == "thinking" and event.get("content"):
                    if event.get("stream"):
                        thinking_parts.append(event["content"])
                    elif event["content"] not in thinking_parts:
                        thinking_parts.append(event["content"])
                yield event

        response, compliance_meta = self.compliance.process(
            response_holder["response"],
        )
        await self.memory_manager.update_after_response(plain_user_msg, response)

        yield {
            "type": "done",
            "answer": response.get_text_content() or "",
            "thinking": "\n\n".join(part for part in thinking_parts if part.strip()),
            "compliance": compliance_meta.to_dict(),
        }

    def _sync_session_attachments(self, session_id: str) -> None:
        memory = self.session_manager.get_memory(session_id)
        if not isinstance(memory, LayeredSessionMemory):
            return
        policies = self.list_policies()
        memory.attachments.recent_files = [p["filename"] for p in policies]
        memory.attachments.current_plan = ""
        memory.attachments.active_skills = []
        memory.attachments.async_tasks = []

    def _bind_session(self, session_id: str) -> None:
        if not self.agent:
            raise RuntimeError("Pipeline 尚未初始化。")
        memory = self.session_manager.get_memory(session_id)
        self.agent.memory = memory
        if isinstance(memory, LayeredSessionMemory):
            memory.set_agent_context(
                sys_prompt=self.agent.sys_prompt,
                model_name=self.settings.model_name,
            )

    def clear_session(self, session_id: str) -> None:
        self.session_manager.clear(session_id)

    async def clear_policy_library(self, clear_long_term: bool = True) -> dict[str, int]:
        if not self.knowledge or not self.manifest:
            raise RuntimeError("Pipeline 尚未初始化。")

        policy_count = len(self.manifest.list_all())
        await self.knowledge.clear_vector_store()
        self.manifest.clear()

        deleted_memories = 0
        if clear_long_term and self.memory_manager:
            async with self.memory_manager.long_term:
                deleted_memories = await self.memory_manager.long_term.clear_all()

        return {
            "deleted_files": policy_count,
            "deleted_memories": deleted_memories,
        }

    def list_policies(self) -> list[dict[str, Any]]:
        if not self.manifest:
            return []
        return self.manifest.list_all()

    async def upload_policy_pdf(
        self,
        file_content: bytes,
        filename: str,
    ) -> dict[str, str | int | float]:
        """解析 PDF 并写入向量库，不保存源文件到本地。"""
        if not self.knowledge or not self.manifest:
            raise RuntimeError("Pipeline 尚未初始化。")

        safe_name = _sanitize_filename(filename)
        documents, parent_entries = await load_pdf_bytes(
            file_content,
            safe_name,
            child_size=self.settings.rag_chunk_size,
            child_overlap=self.settings.rag_chunk_overlap,
        )
        if not documents:
            raise ValueError("无法从 PDF 中提取文本，请确认文件未加密且内容可读。")

        existing = self.manifest.get(safe_name)
        if existing:
            await self.knowledge.delete_by_doc_id(safe_name)

        if self.knowledge.parent_store:
            self.knowledge.parent_store.upsert_batch(parent_entries)
        await self.knowledge.add_documents(documents)
        size_kb = round(len(file_content) / 1024, 1)
        self.manifest.add(safe_name, size_kb, len(documents))

        return {
            "filename": safe_name,
            "relative_path": safe_name,
            "chunks": len(documents),
            "size_kb": size_kb,
        }

    async def delete_policy(self, filename: str) -> None:
        if not self.knowledge or not self.manifest:
            raise RuntimeError("Pipeline 尚未初始化。")

        safe_name = _sanitize_filename(filename)
        if not self.manifest.get(safe_name):
            raise FileNotFoundError(f"保单不存在: {safe_name}")

        await self.knowledge.delete_by_doc_id(safe_name)
        self.manifest.remove(safe_name)
