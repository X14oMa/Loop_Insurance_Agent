"""Agent 流式输出解析：增量 delta + SSE 事件。"""

from __future__ import annotations

import json
from typing import Any

from agentscope.message import Msg

# 每个请求周期内的流式快照，用于计算 delta
_stream_snapshots: dict[str, str] = {}


def reset_stream_state() -> None:
    """新一轮 chat_stream 开始前清空。"""
    _stream_snapshots.clear()


def _blocks_to_text(blocks: Any) -> str:
    """将 tool output 块列表转为纯文本；output 也可能是 str。"""
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""

    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "thinking":
                parts.append(str(block.get("thinking", "")))
    return "\n".join(part for part in parts if part)


def _tool_output_to_text(output: Any) -> str:
    """ToolResultBlock.output 可能是 str 或 ContentBlock 列表。"""
    return _blocks_to_text(output)


def _compute_delta(channel: str, stream_id: str, text: str) -> tuple[str, str]:
    """返回 (delta, full_text)。"""
    key = f"{stream_id}:{channel}"
    previous = _stream_snapshots.get(key, "")
    if text.startswith(previous):
        delta = text[len(previous) :]
    else:
        delta = text
    _stream_snapshots[key] = text
    return delta, text


def _stream_event(
    channel: str,
    *,
    stream_id: str,
    text: str,
    last: bool,
) -> dict[str, Any] | None:
    delta, full = _compute_delta(channel, stream_id, text)
    if not delta and not last:
        return None
    if last:
        _stream_snapshots.pop(f"{stream_id}:{channel}", None)
    return {
        "type": channel,
        "stream_id": stream_id,
        "content": full,
        "delta": delta,
        "stream": True,
        "last": last,
    }


def msg_to_stream_events(msg: Msg, last: bool) -> list[dict[str, Any]]:
    """将 AgentScope print 消息转为 SSE 事件。"""
    events: list[dict[str, Any]] = []
    stream_id = str(getattr(msg, "id", "") or id(msg))

    if msg.role == "system":
        for block in msg.get_content_blocks("tool_result"):
            output_text = _tool_output_to_text(block.get("output", []))
            if output_text.strip():
                preview = output_text if len(output_text) <= 800 else (
                    output_text[:800] + "\n...(结果已截断)"
                )
                events.append(
                    {
                        "type": "thinking",
                        "stream_id": f"{stream_id}-tool",
                        "content": f"工具返回：\n{preview}",
                        "delta": f"工具返回：\n{preview}",
                        "stream": False,
                        "last": last,
                    },
                )
        return events

    for block in msg.get_content_blocks("thinking"):
        thinking_text = str(block.get("thinking", ""))
        if thinking_text:
            item = _stream_event(
                "thinking",
                stream_id=stream_id,
                text=thinking_text,
                last=last,
            )
            if item:
                events.append(item)

    tool_uses = msg.get_content_blocks("tool_use")
    if tool_uses:
        for block in tool_uses:
            tool_input = block.get("input", {})
            if isinstance(tool_input, str):
                input_text = tool_input
            else:
                input_text = json.dumps(tool_input, ensure_ascii=False, indent=2)
            content = f"调用工具 `{block['name']}`：\n{input_text}"
            events.append(
                {
                    "type": "thinking",
                    "stream_id": f"{stream_id}-tool-{block.get('id', block.get('name'))}",
                    "content": content,
                    "delta": content,
                    "stream": False,
                    "last": last,
                },
            )

    text = msg.get_text_content()
    if text and msg.role == "assistant" and not tool_uses:
        item = _stream_event(
            "answer",
            stream_id=stream_id,
            text=text,
            last=last,
        )
        if item:
            events.append(item)

    return events
