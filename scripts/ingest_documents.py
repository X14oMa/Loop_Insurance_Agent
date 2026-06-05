"""批量导入：将指定目录下的 PDF 写入向量库（不保留源文件）。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.config import get_settings
from src.pipeline import InsuranceAgentPipeline


async def main(source_dir: Path) -> None:
    settings = get_settings()
    settings.validate()

    if not source_dir.is_dir():
        print(f"目录不存在: {source_dir}", file=sys.stderr)
        sys.exit(1)

    pdf_files = list(source_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"{source_dir} 下没有 PDF 文件。")
        return

    pipeline = InsuranceAgentPipeline(settings)
    await pipeline.initialize()

    total_chunks = 0
    for pdf_path in pdf_files:
        content = pdf_path.read_bytes()
        result = await pipeline.upload_policy_pdf(content, pdf_path.name)
        total_chunks += int(result["chunks"])
        print(f"已入库: {result['filename']} ({result['chunks']} 片段)")

    print(f"\n共导入 {len(pdf_files)} 个文件，{total_chunks} 个片段。")
    print(f"向量库: {settings.vector_store_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从目录批量导入 PDF 到向量库")
    parser.add_argument(
        "directory",
        type=Path,
        help="包含 PDF 的目录路径",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.directory.resolve()))
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        sys.exit(1)
