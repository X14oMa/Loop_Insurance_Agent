"""Generate project presentation PPTX for Loop Insurance Agent.

Outline (25 slides):
  1  Cover
  2  Agenda
  3  Business background
  4  Question types
  5  Solution overview
  6  Core capabilities
  7  Tech stack
  8  System architecture
  9  Project structure
 10  RAG: ingest & chunking
 11  RAG: hybrid retrieval
 12  RAG: limits & table bottleneck
 13  Agent: main ReAct
 14  Agent: SubAgent delegation
 15  Pipeline & TurnConfig
 16  SSE streaming
 17  Memory module
 18  Compliance & frontend
 19  Model factories
 20  Evaluation setup
 21  Baseline results
 22  Memory ablation
 23  Capability profile & Bad Cases
 24  Future work
 25  Summary
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PRIMARY = RGBColor(0x0F, 0x4C, 0x81)
PRIMARY_LIGHT = RGBColor(0x1A, 0x6B, 0xB5)
ACCENT = RGBColor(0x0D, 0x94, 0x88)
DARK = RGBColor(0x1E, 0x29, 0x3B)
GRAY = RGBColor(0x64, 0x74, 0x8B)
LIGHT = RGBColor(0xF8, 0xFA, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0xBF, 0xDB, 0xFE)


def set_bg(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def blank_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_header(slide, title: str, subtitle: str = "") -> None:
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(10), Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = PRIMARY
    bar.line.fill.background()
    tf = bar.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = WHITE
    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(13)
        p2.font.color.rgb = MUTED


def add_section_slide(prs: Presentation, chapter: str, title: str) -> None:
    slide = blank_slide(prs)
    set_bg(slide, PRIMARY)
    c = slide.shapes.add_textbox(Inches(0.8), Inches(2.6), Inches(8.4), Inches(0.6))
    c.text_frame.paragraphs[0].text = chapter
    c.text_frame.paragraphs[0].font.size = Pt(16)
    c.text_frame.paragraphs[0].font.color.rgb = MUTED
    t = slide.shapes.add_textbox(Inches(0.8), Inches(3.2), Inches(8.4), Inches(1.2))
    t.text_frame.paragraphs[0].text = title
    t.text_frame.paragraphs[0].font.size = Pt(36)
    t.text_frame.paragraphs[0].font.bold = True
    t.text_frame.paragraphs[0].font.color.rgb = WHITE


def add_bullets(
    slide,
    items: list[str],
    *,
    left=0.65,
    top=1.25,
    width=8.7,
    height=5.8,
    size=15,
) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, raw in enumerate(items):
        text = raw.strip()
        level = 1 if text.startswith("-") else 0
        if level:
            text = text[1:].strip()
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = level
        p.font.size = Pt(size - 2 if level else size)
        p.font.color.rgb = GRAY if level else DARK
        p.space_after = Pt(6)


def add_two_col(
    slide,
    left: list[str],
    right: list[str],
    *,
    left_title: str = "",
    right_title: str = "",
) -> None:
    if left_title:
        lt = slide.shapes.add_textbox(Inches(0.65), Inches(1.15), Inches(4.3), Inches(0.35))
        lt.text_frame.paragraphs[0].text = left_title
        lt.text_frame.paragraphs[0].font.bold = True
        lt.text_frame.paragraphs[0].font.size = Pt(13)
        lt.text_frame.paragraphs[0].font.color.rgb = PRIMARY
    if right_title:
        rt = slide.shapes.add_textbox(Inches(5.05), Inches(1.15), Inches(4.3), Inches(0.35))
        rt.text_frame.paragraphs[0].text = right_title
        rt.text_frame.paragraphs[0].font.bold = True
        rt.text_frame.paragraphs[0].font.size = Pt(13)
        rt.text_frame.paragraphs[0].font.color.rgb = PRIMARY
    add_bullets(slide, left, left=0.65, top=1.55, width=4.25, height=5.4, size=13)
    add_bullets(slide, right, left=5.05, top=1.55, width=4.25, height=5.4, size=13)


def add_table(
    slide,
    headers: list[str],
    rows: list[list[str]],
    *,
    top=1.3,
    col_widths: list[float] | None = None,
) -> None:
    cols = len(headers)
    tbl_shape = slide.shapes.add_table(
        len(rows) + 1,
        cols,
        Inches(0.55),
        Inches(top),
        Inches(9.0),
        Inches(min(0.42 * (len(rows) + 1), 5.5)),
    )
    table = tbl_shape.table
    if col_widths:
        for j, w in enumerate(col_widths):
            table.columns[j].width = Inches(w)
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = WHITE
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = DARK


def add_flow(slide, lines: list[str], top=1.35) -> None:
    box = slide.shapes.add_textbox(Inches(0.6), Inches(top), Inches(8.8), Inches(5.6))
    tf = box.text_frame
    tf.word_wrap = True
    mono = ("→", "├", "│", "└", "↓")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(12)
        p.font.color.rgb = DARK
        if any(m in line for m in mono):
            p.font.name = "Consolas"
        p.space_after = Pt(2)


def build_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # ── 1 Cover ──
    s = blank_slide(prs)
    set_bg(s, PRIMARY)
    t = s.shapes.add_textbox(Inches(0.75), Inches(2.1), Inches(8.5), Inches(1.2))
    t.text_frame.paragraphs[0].text = "Loop Insurance Agent"
    t.text_frame.paragraphs[0].font.size = Pt(40)
    t.text_frame.paragraphs[0].font.bold = True
    t.text_frame.paragraphs[0].font.color.rgb = WHITE
    sub = s.shapes.add_textbox(Inches(0.75), Inches(3.3), Inches(8.5), Inches(0.8))
    sub.text_frame.paragraphs[0].text = "保险条款智能咨询系统 — 项目演示"
    sub.text_frame.paragraphs[0].font.size = Pt(22)
    sub.text_frame.paragraphs[0].font.color.rgb = MUTED
    tag = s.shapes.add_textbox(Inches(0.75), Inches(4.5), Inches(8.5), Inches(0.6))
    tag.text_frame.paragraphs[0].text = "RAG · ReAct Agent · 混合检索 · SSE 流式 · 量化评测"
    tag.text_frame.paragraphs[0].font.size = Pt(13)
    tag.text_frame.paragraphs[0].font.color.rgb = WHITE
    foot = s.shapes.add_textbox(Inches(0.75), Inches(6.3), Inches(8.5), Inches(0.4))
    foot.text_frame.paragraphs[0].text = "依据 项目报告v1.md  |  核心代码 ~8,600 行"
    foot.text_frame.paragraphs[0].font.size = Pt(11)
    foot.text_frame.paragraphs[0].font.color.rgb = MUTED

    # ── 2 Agenda ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "演示目录", "共 25 页 · 简约风格")
    add_bullets(
        s,
        [
            "一、项目背景与方案（3–5）",
            "二、架构与技术栈（6–9）",
            "三、核心模块实现（10–19）",
            "- RAG 知识库 · Agent/SubAgent · Pipeline · 记忆 · 合规/前端",
            "四、测试结果与分析（20–23）",
            "五、总结与优化方向（24–25）",
        ],
        top=1.35,
        size=16,
    )

    # ── 3 Business ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "业务背景", "保险条款 PDF 咨询的痛点")
    add_bullets(
        s,
        [
            "条款 PDF 篇幅长、层级深：章 / 节 / 条 / 附表，含大量表格",
            "保障计划表、费率表、给付比例表等 — 结构化信息密集",
            "传统关键词搜索：语义理解弱，同义换说法易漏检",
            "纯 LLM 直接作答：易产生幻觉，难以追溯条款依据",
            "本项目路径：RAG 检索条款片段 + ReAct Agent 组织回答",
            "合规层约束引用格式；免责/来源提示由前端统一展示",
        ],
    )

    # ── 4 Question types ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "用户问题类型", "40 题评测题库覆盖六类场景")
    add_table(
        s,
        ["类型", "示例", "评测难点"],
        [
            ["基础事实", "守御一生等待期多久？", "需精确锚定条款"],
            ["条款解释", "补偿原则是什么意思？", "概念 + 条文对照"],
            ["多条件判断", "第 20 天疾病住院赔不赔？", "多段检索与推理"],
            ["风险提示", "犹豫期后退保损失？", "相对表现最好 97.4%"],
            ["模糊问题", "等待期多久？（未指保单）", "应先澄清 — 弱项 77.8%"],
            ["诱导性问题", "等待期是 0 天对吧？", "须纠正错误前提 — 70.6%"],
        ],
        top=1.25,
        col_widths=[1.4, 3.8, 3.8],
    )

    # ── 5 Solution ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "解决方案概览")
    add_flow(
        s,
        [
            "用户上传 PDF → 内存解析入库（不保存源文件）",
            "         ↓",
            "pypdf 抽文本 → clause_chunking 章节切分 → Qdrant + BM25",
            "         ↓",
            "用户提问 → HybridKnowledge 混合检索 → 主 ReAct Agent",
            "         ├─ retrieve_knowledge（单问 / 多次检索）",
            "         └─ delegate_subagents（多独立子问题并行）",
            "         ↓",
            "ComplianceValidator 清洗 → SSE 流式返回 → 前端展示",
        ],
    )

    add_section_slide(prs, "第二章", "架构与技术栈")

    # ── 6 Core capabilities ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "核心能力一览", "与代码模块一一对应")
    add_table(
        s,
        ["能力", "实现", "要点"],
        [
            ["PDF 入库", "document_loader + orchestrator", "内存解析，manifest 登记"],
            ["混合检索", "hybrid_knowledge.py", "Dense + BM25 + RRF + Rerank"],
            ["主 Agent", "insurance_react_agent.py", "ReAct；retrieve + delegate"],
            ["SubAgent", "subagent_runner.py", "工具触发；Semaphore 并行"],
            ["会话记忆", "layered_session_memory.py", "80% 软视图 / 93% 硬压缩"],
            ["长期记忆", "markdown_long_term_memory + ltm_merge", "Markdown；Agent 工具读写"],
            ["合规", "validator.py", "Hook 清洗；前端展示免责"],
            ["流式", "streaming.py + server.py", "SSE thinking/answer/done"],
        ],
        top=1.22,
        col_widths=[1.5, 3.2, 4.3],
    )

    # ── 7 Tech stack ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "技术栈")
    add_two_col(
        s,
        [
            "Agent 框架：AgentScope",
            "- ReActAgent / Toolkit / LongTermMemoryBase",
            "对话：DeepSeek（默认）或 DashScope 通义",
            "嵌入：DashScope text-embedding-v4（1024 维）",
            "与对话提供商可分离配置",
        ],
        [
            "向量库：Qdrant 本地 path 持久化",
            "长期记忆：data/memory_store/{user_id}.md",
            "Web：FastAPI + Uvicorn",
            "前端：原生 HTML / CSS / JS",
            "评测：40 题 Rubric 自动评分脚本",
        ],
        left_title="AI 层",
        right_title="存储与接入",
    )

    # ── 8 Architecture ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "系统架构", "单主 Agent + 工具委托（无 Pipeline 路径分叉）")
    add_flow(
        s,
        [
            "Web 前端 ──SSE──► FastAPI (api/server.py)",
            "                    ▼",
            "         InsuranceAgentPipeline (orchestrator.py)",
            "           ├─ agent_router → TurnConfig（复杂度 / 检索预算）",
            "           ├─ SessionManager → LayeredSessionMemory（per session_id）",
            "           ├─ InsuranceReActAgent",
            "           │    ├─ retrieve_knowledge → HybridKnowledge",
            "           │    ├─ delegate_subagents → subagent_runner",
            "           │    └─ retrieve/record_from_memory（Markdown LTM）",
            "           ├─ ComplianceValidator.post_reply",
            "           └─ MemoryManager.update_after_response（ltm_merge）",
        ],
    )

    # ── 9 Structure ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "项目结构", "模块划分与代码规模")
    add_two_col(
        s,
        [
            "src/pipeline/ — 编排、路由、流式",
            "src/agent/ — Agent 工厂与 Prompt",
            "src/rag/ — 切分、检索、限长",
            "src/memory/ — 分层会话 + Markdown LTM",
            "src/model/ — LLM / Embedding 工厂",
            "src/compliance/ — 合规清洗",
            "src/api/ — FastAPI + SSE",
        ],
        [
            "frontend/ — Web UI",
            "scripts/ — 入库与评测",
            "test/ — 40 题 + 3 份条款 PDF",
            "data/ — vector_store / memory_store",
            "",
            "代码规模（约）：",
            "src ~4,270 + scripts ~2,450",
            "+ frontend ~1,880 ≈ 8,600 行",
        ],
        left_title="目录",
        right_title="规模",
    )

    add_section_slide(prs, "第三章", "核心模块实现")

    # ── 10 RAG ingest ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "RAG：PDF 入库与切分", "document_loader · clause_chunking")
    add_bullets(
        s,
        [
            "入库：pypdf.PdfReader 逐页 extract_text → 章节结构化切分",
            "层级识别（取优先级最高且 match≥2 的模式）：",
            "- 第X章/节（100）· 第X条（90）· 2.8 编号（80）· 一、序号（70）",
            "过滤目录行（连续 .... / 页码引导行）",
            "Parent：整节全文 → parent_chunks.json",
            "Child：256 字滑动窗口，overlap 64 → Qdrant + BM25",
            "命中 child 且 use_parent_return=True → 工具返回 parent 整节",
            "policy_manifest.json 登记已入库 filename；不保存 PDF 源文件",
        ],
        size=14,
    )

    # ── 11 RAG hybrid ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "RAG：混合检索", "hybrid_knowledge.py · reranker.py")
    add_bullets(
        s,
        [
            "① 双路召回：Qdrant 语义 Top-N + BM25Okapi 关键词 Top-N",
            "   中文分词：单字 + 双字 bigram（tokenizer.py）",
            "② RRF 融合：score += 1/(60+rank+1)",
            "③ 轻量 Rerank（非 cross-encoder）：",
            "   RRF 0.35 + 语义 0.35 + BM25 0.15 + 词面重叠 0.15",
            "④ 同一 parent 去重，保留最高分",
            "⑤ 格式化：[来源: doc_id | clause_label | 相关度] + display_text",
            "retrieve_knowledge 注册到 Toolkit；doc_id 对应 policy_library 文件名",
        ],
        size=14,
    )

    # ── 12 RAG limits ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "RAG：限长与表格瓶颈", "tool_response_limits.py")
    add_two_col(
        s,
        [
            "工具返回限长（默认）：",
            "RAG_TOOL_LIMIT = 3 块",
            "单块 ≤ 2000 字符",
            "总响应 ≤ 8000 字符",
            "超出附加 ...(内容已截断)",
            "防止 ReAct 上下文被 tool_result 撑爆",
        ],
        [
            "表格瓶颈（评测 Bad Case 根因）：",
            "pypdf 多列表格 → 列对齐丢失",
            "256 字 child 切分 → 表格拦腰截断",
            "检索命中半表 → Agent 缺条件/混淆计划",
            "Q07/Q15/Q21 等表格题偶发 1 分",
        ],
        left_title="限长保护",
        right_title="已知局限",
    )

    # ── 13 Main Agent ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "Agent：主 ReAct Agent", "insurance_agent.py · insurance_react_agent.py")
    add_bullets(
        s,
        [
            "基类 AgentScope ReActAgent；子类覆写迭代入口",
            "每轮 ReAct 开始前：LayeredSessionMemory.on_iteration_start() 检查硬压缩",
            "工具：retrieve_knowledge + 可选 delegate_subagents",
            "长期记忆 mode=agent_control：retrieve_from_memory / record_to_memory",
            "System Prompt 静态模板 + 每轮动态刷新（检索预算、引用格式）",
            "核心约束：条款问题必须检索；引用 [保单filename | 第X条]",
            "禁止正文写免责声明（前端 + compliance 元数据负责）",
            "post_reply Hook：ComplianceValidator 剥离误带页脚",
        ],
        size=14,
    )

    # ── 14 SubAgent ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "Agent：SubAgent 委托", "subagent_tool.py · subagent_runner.py")
    add_table(
        s,
        ["设计点", "实现", "默认值"],
        [
            ["谁决定调用", "主 Agent ReAct 自决", "非 Pipeline if-else"],
            ["SubAgent 工具", "仅 retrieve_knowledge", "无 delegate 递归"],
            ["最少子问题", "delegate_subagents 校验", "2"],
            ["最多子问题", "截断 cleaned 列表", "5"],
            ["并行度", "asyncio.Semaphore", "3"],
            ["汇总上限", "synthesize_subagent_brief", "3500 字"],
        ],
        top=1.25,
        col_widths=[1.8, 3.5, 3.7],
    )
    note = s.shapes.add_textbox(Inches(0.65), Inches(4.5), Inches(8.7), Inches(0.9))
    note.text_frame.paragraphs[0].text = (
        "流程：并行 SubAgent 检索 → LLM 汇总带引用短文 → 写入主 Agent tool_result → 主 Agent 组织最终回答"
    )
    note.text_frame.paragraphs[0].font.size = Pt(12)
    note.text_frame.paragraphs[0].font.color.rgb = ACCENT

    # ── 15 Pipeline ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "Pipeline 与 TurnConfig", "orchestrator.py · agent_router.py")
    add_two_col(
        s,
        [
            "InsuranceAgentPipeline 职责：",
            "initialize() — Embedding/Qdrant/Agent",
            "chat_stream() — 完整对话链路",
            "upload_policy_pdf() — 入库",
            "_bind_session() — 切换 LayeredSessionMemory",
            "",
            "Enriched Input 结构：",
            "policy_library + current_turn_only",
            "+ retrieval_budget + 用户原文",
        ],
        [
            "TurnConfig（每轮生成）：",
            "complexity: low / medium / high",
            "  启发式 + 可选 DECOMPOSE_MODEL_NAME JSON",
            "max_retrievals: 3 / 5 / 6",
            "reason — 配置理由（日志）",
            "",
            "LTM 合并基于 plain user_input",
            "避免 policy_library 污染长期记忆",
        ],
        left_title="Pipeline",
        right_title="TurnConfig",
    )

    # ── 16 SSE ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "SSE 流式输出", "streaming.py")
    add_table(
        s,
        ["事件 type", "内容", "关键字段"],
        [
            ["thinking", "推理 / tool_use / tool_result", "delta, stream_id, last"],
            ["answer", "assistant 正文", "增量 delta"],
            ["done", "流结束", "answer, thinking, compliance"],
            ["error", "异常", "message"],
        ],
        top=1.35,
        col_widths=[1.5, 3.5, 4.0],
    )
    add_bullets(
        s,
        [
            "增量逻辑：快照 diff 生成 delta；前端 64ms 节流渲染",
            "评测 TTFT：chat_stream 发起 → 首个 answer 事件（含检索+工具，非裸 LLM TTFT）",
            "性能：平均 TTFT ~11s（P50 ~7s，P95 ~40s）；复杂题可达 60–88s",
        ],
        top=3.35,
        size=13,
    )

    # ── 17 Memory ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "记忆模块", "短期会话 + Markdown 长期记忆")
    add_table(
        s,
        ["层级", "存储", "行为"],
        [
            ["短期会话", "进程内存 / session_id", "多 Tab 隔离；重启丢失"],
            ["软层 ≥80%", "LayeredSessionMemory", "动态视图：摘要+工具块+最近 N 条"],
            ["硬层 ≥93%", "同上", "LLM 四段摘要后替换工作区；archives 存快照"],
            ["长期记忆", "{user_id}.md", "Agent 工具读写用户上下文"],
            ["自动合并", "ltm_merge.py", "关键词触发 → LLM 精炼合并 Markdown"],
        ],
        top=1.28,
        col_widths=[1.5, 2.8, 4.7],
    )
    note = s.shapes.add_textbox(Inches(0.65), Inches(4.35), Inches(8.7), Inches(0.8))
    note.text_frame.paragraphs[0].text = (
        "消融结论：累积 vs 隔离记忆对条款 QA 仅 +1%；长 session 可能固化模糊题错误答案（Q31）"
    )
    note.text_frame.paragraphs[0].font.size = Pt(12)
    note.text_frame.paragraphs[0].font.color.rgb = ACCENT
    note.text_frame.paragraphs[0].font.italic = True

    # ── 18 Compliance & FE ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "合规与 Web 前端", "validator.py · frontend/app.js")
    add_two_col(
        s,
        [
            "ComplianceValidator.process()",
            "检测条款类关键词 → ComplianceMeta",
            "strip_display_footers() 移除误带免责",
            "GET /api/ui-config 全局免责文案",
            "done.compliance 控制来源提示",
        ],
        [
            "多 Tab 会话 + localStorage 持久化",
            "SSE delta 节流 + Markdown + DOMPurify",
            "server_boot_id 检测后端重启",
            "固定视口 + 消息区/输入框二级滚动",
            "PDF 拖拽上传 · 即时入库可问答",
        ],
        left_title="合规",
        right_title="前端",
    )

    # ── 19 Models ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "模型工厂", "按场景分工配置")
    add_table(
        s,
        ["场景", "配置项", "说明"],
        [
            ["主 / Sub Agent 对话", "MODEL_NAME", "ReAct 主循环"],
            ["复杂度评估", "DECOMPOSE_MODEL_NAME", "TurnConfig JSON"],
            ["上下文压缩", "COMPRESSION_MODEL_NAME", "软/硬层摘要"],
            ["记忆合并", "PROFILE_MODEL_NAME", "ltm_merge Markdown"],
            ["向量嵌入", "EMBEDDING_*", "独立工厂，与对话解耦"],
        ],
        top=1.4,
        col_widths=[2.2, 2.8, 4.0],
    )

    add_section_slide(prs, "第四章", "测试结果与分析")

    # ── 20 Eval setup ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "评测体系")
    add_table(
        s,
        ["项目", "说明"],
        [
            ["题库", "test/测试问题.txt — Q01–Q40，六类题型"],
            ["参考答案", "test/测试答案.txt — Rubric 规则评分 0/1/2"],
            ["条款 PDF", "如意鑫 · 守御一生（智臻版）· 臻宝贝 2026"],
            ["指标", "得分、总耗时、TTFT、得分点/扣分点"],
            ["脚本", "run_eval_test.py · run_eval_memory_ablation.py · score_eval.py"],
        ],
        top=1.35,
        col_widths=[2.0, 7.0],
    )

    # ── 21 Baseline ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "基线三轮评测", "eval_20260608_160054")
    add_two_col(
        s,
        [
            "137 次作答汇总：",
            "总得分 243/274（88.7%）",
            "完全正确 113 · 部分 17 · 错误 7",
            "平均耗时 18.32s",
            "平均 TTFT 10.89s",
            "",
            "分轮（session 隔离，非跨轮学习）：",
            "第 1 轮 74/94（78.7%）",
            "第 2 轮 82/90（91.1%）",
            "第 3 轮 87/90（96.7%）",
        ],
        [
            "分题型正确率：",
            "风险提示     97.4%",
            "基础事实     94.4%",
            "条款解释     92.3%",
            "多条件判断   91.7%",
            "模糊问题     77.8%",
            "诱导性问题   70.6%  ← 薄弱",
            "",
            "第 1 轮偏低含 regex 误扣引用",
        ],
        left_title="整体",
        right_title="分题型",
    )

    # ── 22 Ablation ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "记忆消融实验", "ablation_20260608_165329 · 47 题 × 3 轮")
    add_table(
        s,
        ["组别", "轮间策略", "得分", "正确率", "TTFT"],
        [
            ["A 隔离组", "clear_session + 清 LTM", "230/282", "81.6%", "11.84s"],
            ["B 累积组", "复用 session + LTM 累积", "233/282", "82.6%", "11.44s"],
            ["差值", "—", "+3", "+1.0%", "−0.40s"],
        ],
        top=1.35,
        col_widths=[1.3, 3.2, 1.5, 1.5, 1.5],
    )
    add_bullets(
        s,
        [
            "两组第 1 轮均为 83/94（88.3%）— 基线可比",
            "记忆层对条款 QA 增益处于噪声级；Q31 累积组显著更差",
            "Q36 诱导题两组 6/6 次全 0 分 — 与记忆策略无关的系统性弱项",
        ],
        top=3.5,
        size=13,
    )

    # ── 23 Bad cases ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "能力画像与典型 Bad Case")
    add_two_col(
        s,
        [
            "强项（RAG 命中后 85–97%）：",
            "基础事实 · 条款解释",
            "多条件判断 · 风险提示",
            "",
            "弱项：",
            "模糊题 Q31 — 未澄清三份保单",
            "诱导题 Q36/Q38 — 错误前提",
            "表格题 — 检索片段不完整",
        ],
        [
            "Bad Case 摘录：",
            "Q31：直接锁定守御一生 30 天",
            "  累积组反复只答一份 → 错误固化",
            "Q36：「无等待期」≈「0 天」触发 forbid",
            "Q38：对比表复述「至少 5%」误扣",
            "Q07/Q15：半张表 → 漏计划/医保条件",
        ],
        left_title="画像",
        right_title="案例",
    )

    add_section_slide(prs, "第五章", "总结与展望")

    # ── 24 Future ──
    s = blank_slide(prs)
    set_bg(s, LIGHT)
    add_header(s, "未来优化方向")
    add_table(
        s,
        ["优先级", "方向", "措施"],
        [
            ["P0", "诱导/模糊题", "Prompt 强制澄清；Q31/Q36 few-shot；Rubric 否定语境"],
            ["P0", "PDF 表格", "pdfplumber 保留行列；表格感知切分；整表 parent"],
            ["P0", "评分修复", "has_clause_ref 兼容 Markdown 引用"],
            ["P1", "TTFT 长尾", "简单题短路；Pipeline 分阶段打点"],
            ["P1", "评测规范", "固定题集 + 轮间 session/LTM 隔离"],
            ["P2", "Session 持久化", "当前进程内，重启丢失"],
            ["P3", "Rerank", "cross-encoder 或 LLM rerank"],
        ],
        top=1.25,
        col_widths=[1.0, 2.2, 5.8],
    )

    # ── 25 Summary ──
    s = blank_slide(prs)
    set_bg(s, PRIMARY)
    t = s.shapes.add_textbox(Inches(0.75), Inches(1.4), Inches(8.5), Inches(0.7))
    t.text_frame.paragraphs[0].text = "项目总结"
    t.text_frame.paragraphs[0].font.size = Pt(32)
    t.text_frame.paragraphs[0].font.bold = True
    t.text_frame.paragraphs[0].font.color.rgb = WHITE
    box = s.shapes.add_textbox(Inches(0.75), Inches(2.3), Inches(8.5), Inches(3.8))
    tf = box.text_frame
    tf.word_wrap = True
    summary_items = [
        "完成 PDF 入库 → 混合 RAG → ReAct + SubAgent → 分层记忆 → Web 流式全链路",
        "40 题标准题库综合正确率约 82%–89%（视实验与评分规则）",
        "SubAgent 工具化唤起 + 并行硬约束，可控支持多子问题",
        "建立 TTFT / 得分点 / 扣分点自动化评测流水线",
        "识别 PDF 表格 → 切分 → 检索为当前 RAG 主要瓶颈",
        "ReAct 约束分层：System Prompt + 工具描述 + 运行时校验",
    ]
    for i, item in enumerate(summary_items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "▸  " + item
        p.font.size = Pt(15)
        p.font.color.rgb = WHITE
        p.space_after = Pt(10)
    qa = s.shapes.add_textbox(Inches(0.75), Inches(6.35), Inches(8.5), Inches(0.5))
    qa.text_frame.paragraphs[0].text = "谢谢 · Q & A"
    qa.text_frame.paragraphs[0].font.size = Pt(18)
    qa.text_frame.paragraphs[0].font.color.rgb = MUTED
    qa.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    return prs


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "Loop_Insurance_Agent_Presentation.pptx"
    prs = build_presentation()
    prs.save(str(out))
    print(f"Generated: {out}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
