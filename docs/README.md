# 文档目录

本目录存放 Loop Insurance Agent 的设计与运维说明。根目录 [README.md](../README.md) 侧重快速上手；本文档说明各文件的用途与建议阅读顺序。

## 文档列表

| 文件 | 受众 | 内容 |
|------|------|------|
| [architecture.md](architecture.md) | 开发 / 架构评审 | 分层架构、对话编排、记忆与 RAG、API/SSE、合规、模块映射、Mermaid 图 |
| [../README.md](../README.md) | 所有人 | 安装、配置表、API 一览、项目结构、模型切换 |
| [../项目理解.txt](../项目理解.txt) | 新成员速览 | 中文模块职责与数据流摘要 |
| [../.env.example](../.env.example) | 部署 | 环境变量模板与注释 |

## 建议阅读顺序

1. 根目录 **README** → 跑通 Web 与 `.env`  
2. **项目理解.txt** → 建立 RAG / Agent / Memory 三块心智模型  
3. **architecture.md** → 需要改编排、记忆或 RAG 时再深入对应章节  

## 与代码的对应关系

- **编排入口**：`src/pipeline/orchestrator.py`  
- **轮次配置**：`src/pipeline/agent_router.py`（`TurnConfig`，非路径路由）  
- **主 Agent**：`src/agent/insurance_react_agent.py`  
- **SubAgent 工具**：`src/agent/subagent_tool.py` → `src/pipeline/subagent_runner.py`  
- **RAG**：`src/rag/hybrid_knowledge.py`  
- **合规**：`src/compliance/validator.py` + 前端 `frontend/app.js`  
- **配置真源**：`src/config.py`（环境变量默认值以代码为准，`.env.example` 为推荐模板）  

配置项变更时，请同步更新 **README 配置表**、**architecture.md §13** 与 **`.env.example`**。
