"""FastAPI 后端：保单上传与 Agent 对话。"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, get_settings
from src.pipeline import InsuranceAgentPipeline

FRONTEND_DIR = PROJECT_ROOT / "frontend"
_pipeline: InsuranceAgentPipeline | None = None
_init_lock = asyncio.Lock()
# 进程启动 ID：前端用于检测「后端已重启、会话记忆已丢失」
SERVER_BOOT_ID: str = uuid.uuid4().hex


async def get_pipeline() -> InsuranceAgentPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    async with _init_lock:
        if _pipeline is None:
            settings = get_settings()
            pipeline = InsuranceAgentPipeline(settings)
            await pipeline.initialize()
            _pipeline = pipeline
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    await get_pipeline()
    yield
    global _pipeline
    if _pipeline and _pipeline.knowledge:
        client = _pipeline.knowledge.embedding_store.get_client()
        await client.close()
    _pipeline = None


app = FastAPI(
    title="Insurance Agent API",
    description="保险 Agent 保单上传与咨询接口",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default", max_length=64)


class UploadResponse(BaseModel):
    filename: str
    relative_path: str
    chunks: int
    size_kb: float


class HealthResponse(BaseModel):
    status: str
    policies_count: int
    server_boot_id: str


class ClearPolicyRequest(BaseModel):
    clear_long_term: bool = True


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    pipeline = await get_pipeline()
    policies = pipeline.list_policies()
    return HealthResponse(
        status="ok",
        policies_count=len(policies),
        server_boot_id=SERVER_BOOT_ID,
    )


@app.get("/api/ui-config")
async def ui_config() -> dict[str, str]:
    """前端展示的免责声明与来源提示文案。"""
    pipeline = await get_pipeline()
    return pipeline.compliance.ui_config()


@app.get("/api/documents")
async def list_documents() -> dict[str, Any]:
    pipeline = await get_pipeline()
    return {"documents": pipeline.list_policies()}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_policy(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 格式的保单文件。")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制（最大 {settings.max_upload_size_mb} MB）。",
        )

    pipeline = await get_pipeline()
    try:
        result = await pipeline.upload_policy_pdf(content, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(
        filename=str(result["filename"]),
        relative_path=str(result["relative_path"]),
        chunks=int(result["chunks"]),
        size_kb=float(result["size_kb"]),
    )


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str) -> dict[str, str]:
    pipeline = await get_pipeline()
    try:
        await pipeline.delete_policy(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": f"已删除 {filename}"}


@app.delete("/api/documents")
async def clear_policy_library(body: ClearPolicyRequest | None = None) -> dict[str, Any]:
    """清空全部保单文件与知识库索引。"""
    pipeline = await get_pipeline()
    clear_long_term = body.clear_long_term if body else True
    result = await pipeline.clear_policy_library(clear_long_term=clear_long_term)
    return {
        "message": "保单记忆库已清空",
        **result,
    }


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str) -> dict[str, str]:
    """清除指定对话窗口的后端会话记忆。"""
    pipeline = await get_pipeline()
    pipeline.clear_session(session_id)
    return {"message": f"会话 {session_id} 已清除"}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    pipeline = await get_pipeline()

    async def event_generator():
        try:
            async for event in pipeline.chat_stream(
                request.message.strip(),
                session_id=request.session_id,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error_event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    pipeline = await get_pipeline()
    try:
        reply, compliance_meta = await pipeline.chat(
            request.message.strip(),
            session_id=request.session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"对话失败: {exc}") from exc
    return {"reply": reply, "compliance": compliance_meta.to_dict()}


@app.get("/")
async def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="前端页面不存在。")
    return FileResponse(index_path)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
