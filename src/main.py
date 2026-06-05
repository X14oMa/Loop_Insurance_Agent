"""Insurance Agent 系统入口。"""

from __future__ import annotations

import asyncio
import sys

from src.config import get_settings
from src.pipeline import InsuranceAgentPipeline


async def run_interactive() -> None:
    settings = get_settings()
    pipeline = InsuranceAgentPipeline(settings)

    print("正在初始化 Insurance Agent 系统...")
    await pipeline.initialize()
    print("系统就绪。输入问题开始咨询，输入 quit 退出。\n")

    while True:
        try:
            user_input = input("您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q", "退出"}:
            print("再见！")
            break

        print("\n助手: ", end="", flush=True)
        try:
            response, _compliance = await pipeline.chat(user_input)
            print(response)
        except Exception as exc:
            print(f"\n[错误] {exc}")
        print()


def main() -> None:
    try:
        asyncio.run(run_interactive())
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        print("请复制 .env.example 为 .env 并填写 DASHSCOPE_API_KEY。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
