"""会话附件上下文：硬压缩时注入计划、文件、异步任务等。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AsyncTaskSnapshot:
    """子 Agent 等异步任务快照。"""

    task_id: str
    description: str
    status: str
    detail: str = ""


@dataclass
class SessionContextAttachments:
    """硬压缩第三段「附件」的数据源。"""

    recent_files: list[str] = field(default_factory=list)
    current_plan: str = ""
    active_skills: list[str] = field(default_factory=list)
    async_tasks: list[AsyncTaskSnapshot] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = ["## 附件"]
        if self.recent_files:
            lines.append("### 最近相关文件 / 保单")
            for path in self.recent_files:
                lines.append(f"- {path}")
        else:
            lines.append("### 最近相关文件 / 保单\n- （无）")

        if self.current_plan:
            lines.append(f"### 当前计划\n{self.current_plan}")
        else:
            lines.append("### 当前计划\n- （无）")

        if self.active_skills:
            lines.append("### 激活的技能")
            for skill in self.active_skills:
                lines.append(f"- {skill}")
        else:
            lines.append("### 激活的技能\n- （无）")

        if self.async_tasks:
            lines.append("### 异步任务状态")
            for task in self.async_tasks:
                line = f"- [{task.status}] {task.task_id}: {task.description}"
                if task.detail:
                    line += f" — {task.detail}"
                lines.append(line)
        else:
            lines.append("### 异步任务状态\n- （无）")

        if self.extra:
            lines.append("### 其他上下文")
            for key, value in self.extra.items():
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)
