"""项目配置模块。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    """Agent 系统全局配置。"""

    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek").strip().lower(),
    )
    dashscope_api_key: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""),
    )
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""),
    )
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ).strip(),
    )
    user_id: str = field(default_factory=lambda: os.getenv("USER_ID", "default_user"))
    agent_name: str = field(
        default_factory=lambda: os.getenv("AGENT_NAME", "InsuranceAgent"),
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "deepseek-chat"),
    )
    model_context_window: int = field(
        default_factory=lambda: int(os.getenv("MODEL_CONTEXT_WINDOW", "0") or "0"),
    )
    context_soft_ratio: float = field(
        default_factory=lambda: float(os.getenv("CONTEXT_SOFT_RATIO", "0.8")),
    )
    context_hard_ratio: float = field(
        default_factory=lambda: float(os.getenv("CONTEXT_HARD_RATIO", "0.93")),
    )
    context_hard_reserve_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("CONTEXT_HARD_RESERVE_TOKENS", "13000"),
        ),
    )
    max_main_agent_retrievals: int = field(
        default_factory=lambda: int(os.getenv("MAX_MAIN_AGENT_RETRIEVALS", "5")),
    )
    max_main_agent_retrievals_low: int = field(
        default_factory=lambda: int(os.getenv("MAX_MAIN_AGENT_RETRIEVALS_LOW", "3")),
    )
    max_main_agent_retrievals_high: int = field(
        default_factory=lambda: int(os.getenv("MAX_MAIN_AGENT_RETRIEVALS_HIGH", "6")),
    )
    dashscope_base_url: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_BASE_URL", ""),
    )
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "dashscope")
        .strip()
        .lower(),
    )
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
    )
    embedding_dimensions: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
    )
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "").strip(),
    )
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", "").strip(),
    )
    vector_store_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("VECTOR_STORE_PATH", "./data/vector_store"),
        ),
    )
    vector_store_persist: bool = field(
        default_factory=lambda: os.getenv(
            "VECTOR_STORE_PERSIST", "true",
        ).lower() in {"1", "true", "yes"},
    )
    long_term_memory_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("LONG_TERM_MEMORY_PATH", "./data/memory_store"),
        ),
    )
    long_term_auto_profile: bool = field(
        default_factory=lambda: os.getenv(
            "LONG_TERM_AUTO_PROFILE", "true",
        ).lower() in {"1", "true", "yes"},
    )
    profile_model_name: str = field(
        default_factory=lambda: os.getenv("PROFILE_MODEL_NAME", "").strip()
        or os.getenv("DECOMPOSE_MODEL_NAME", "deepseek-chat"),
    )
    rag_chunk_size: int = 256
    rag_chunk_overlap: int = 64
    rag_score_threshold: float = 0.5
    rag_top_k: int = 5
    rag_tool_limit: int = field(
        default_factory=lambda: int(os.getenv("RAG_TOOL_LIMIT", "3")),
    )
    rag_max_chunk_display_chars: int = field(
        default_factory=lambda: int(os.getenv("RAG_MAX_CHUNK_DISPLAY_CHARS", "2000")),
    )
    rag_max_tool_response_chars: int = field(
        default_factory=lambda: int(os.getenv("RAG_MAX_TOOL_RESPONSE_CHARS", "8000")),
    )
    agent_compression_threshold: int = field(
        default_factory=lambda: int(os.getenv("AGENT_COMPRESSION_THRESHOLD", "20000")),
    )
    agent_compression_keep_recent: int = field(
        default_factory=lambda: int(os.getenv("AGENT_COMPRESSION_KEEP_RECENT", "4")),
    )
    compression_model_name: str = field(
        default_factory=lambda: os.getenv("COMPRESSION_MODEL_NAME", "deepseek-chat"),
    )
    subagent_enabled: bool = field(
        default_factory=lambda: os.getenv("SUBAGENT_ENABLED", "true").lower()
        in {"1", "true", "yes"},
    )
    subagent_min_questions: int = field(
        default_factory=lambda: int(os.getenv("SUBAGENT_MIN_QUESTIONS", "2")),
    )
    decompose_model_name: str = field(
        default_factory=lambda: os.getenv("DECOMPOSE_MODEL_NAME", "deepseek-chat"),
    )
    max_subagents: int = field(
        default_factory=lambda: int(os.getenv("MAX_SUBAGENTS", "5")),
    )
    max_subagent_concurrency: int = field(
        default_factory=lambda: int(os.getenv("MAX_SUBAGENT_CONCURRENCY", "3")),
    )
    max_subagent_retrievals: int = field(
        default_factory=lambda: int(os.getenv("MAX_SUBAGENT_RETRIEVALS", "4")),
    )
    max_subagent_retrievals_high: int = field(
        default_factory=lambda: int(os.getenv("MAX_SUBAGENT_RETRIEVALS_HIGH", "5")),
    )
    subagent_tool_max_chars: int = field(
        default_factory=lambda: int(os.getenv("SUBAGENT_TOOL_MAX_CHARS", "3500")),
    )
    max_upload_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_MB", "20")),
    )
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8001")))

    # 合规文案（仅前端展示，不写入 Agent 回答）
    disclaimer_text: str = field(
        default_factory=lambda: os.getenv(
            "DISCLAIMER_TEXT",
            "回答仅供参考，不构成任何保险承诺或法律建议。"
            "具体保障范围、理赔条件及免责条款以您所持有的正式保险合同为准。"
            "如有疑问，请联系您的保险顾问或拨打官方客服热线。",
        ),
    )
    source_notice_text: str = field(
        default_factory=lambda: os.getenv(
            "SOURCE_NOTICE_TEXT",
            "以上回答基于知识库检索结果整理，具体条款内容请以正式保险合同原文为准。",
        ),
    )

    def validate(self) -> None:
        provider = self.llm_provider.strip().lower()
        if provider not in {"deepseek", "dashscope"}:
            raise ValueError(
                f"不支持的 LLM_PROVIDER={self.llm_provider!r}，"
                "请使用 deepseek 或 dashscope。",
            )
        self.llm_provider = provider

        if self.llm_provider == "deepseek":
            if not self.deepseek_api_key:
                raise ValueError(
                    "LLM_PROVIDER=deepseek 时需设置 DEEPSEEK_API_KEY。"
                    "请在 .env 中配置 DeepSeek API Key。",
                )
        elif not self.dashscope_api_key:
            raise ValueError(
                "LLM_PROVIDER=dashscope 时需设置 DASHSCOPE_API_KEY。",
            )

        emb = self.embedding_provider.strip().lower()
        if emb not in {"dashscope", "openai"}:
            raise ValueError(
                f"不支持的 EMBEDDING_PROVIDER={self.embedding_provider!r}，"
                "请使用 dashscope 或 openai。",
            )
        self.embedding_provider = emb

        if emb == "dashscope":
            if not (self.embedding_api_key or self.dashscope_api_key):
                raise ValueError(
                    "EMBEDDING_PROVIDER=dashscope 时需配置 "
                    "DASHSCOPE_API_KEY 或 EMBEDDING_API_KEY。",
                )
        elif not (
            self.embedding_api_key
            or self.deepseek_api_key
            or self.dashscope_api_key
        ):
            raise ValueError(
                "EMBEDDING_PROVIDER=openai 时需配置 EMBEDDING_API_KEY、"
                "DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY 之一。",
            )

        if emb == "openai" and not self.embedding_base_url:
            self.embedding_base_url = (
                self.deepseek_base_url
                if self.llm_provider == "deepseek" and self.deepseek_base_url
                else "https://api.deepseek.com"
            )


def get_settings() -> Settings:
    settings = Settings()
    for path_attr in ("vector_store_path", "long_term_memory_path"):
        path = getattr(settings, path_attr)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
            setattr(settings, path_attr, path)
        path.mkdir(parents=True, exist_ok=True)
    return settings
