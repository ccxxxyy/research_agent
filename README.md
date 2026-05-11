# research-agent

> **基于 LangGraph + MCP + Agentic RAG 的多智能体深度研究系统**
> 面向 A 股二级市场研究：行情、披露公告、新闻舆情、研究报告知识库 —— 一个 supervisor + 六个 specialist 协作完成。

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-167%20passing-brightgreen.svg)](#测试)

---

## 一、它解决什么问题

「**给我做一份宁德时代 2023 年的业绩简评，对比一下行业平均，再看下最近一周市场对它的舆情情况，并参考我之前上传过的 ESG 报告里讲的碳中和承诺。**」

回答这种**混合数据源、混合粒度的研究问题**，传统做法是：
1. 一个 Agent 拿着一堆工具，靠 prompt 自己分配先调谁——容易乱套；
2. 或者一个写死流程的 pipeline——不灵活，加新工具就要重写编排逻辑。

本项目用 **LangGraph Supervisor 模式 + 6 个能力解耦的 specialist** 的方式，让 supervisor 只做"读用户需求 → 拆任务 → 选 specialist → 串联结果"，而 specialist 只做"我擅长这一种事，给我参数我就跑"。

业务侧实打实接了真实数据源：
- **行情/基本面** —— akshare 抓东方财富、雪球、新浪；
- **披露公告 PDF** —— 巨潮资讯（cninfo）；
- **市场新闻 & 舆情** —— 东方财富、财联社快讯、雪球热度榜、百度财经早晚报；
- **量化情感** —— SnowNLP + 金融关键词词典（确定性，不走 LLM 打分）；
- **私有 PDF 知识库** —— FAISS（向量）+ BM25（关键词）+ bge-reranker（cross-encoder 重排），Corrective-RAG 闭环。

---

## 二、架构

```
                                   ┌─────────────────────────────────────────┐
                                   │            FastAPI + SSE                │
                                   │  /api/supervisor/research        (sync) │
                                   │  /api/supervisor/research/stream (SSE)  │
                                   │  /api/knowledge/*  /api/memory/*  ...   │
                                   └────────────────────┬────────────────────┘
                                                        │ messages: list[BaseMessage]
                                                        ▼
                                   ┌─────────────────────────────────────────┐
                                   │      research_supervisor  (HEAVY)       │
                                   │   langgraph_supervisor.create_supervisor│
                                   └─┬───────┬───────┬───────┬───────┬───────┘
              transfer_to_X tool-calls │       │       │       │       │       │
                                        ▼       ▼       ▼       ▼       ▼       ▼
                                  data_expert  report   coder  news  knowledge sentiment
                                   (MEDIUM)   _expert  _expert _expert _expert  _expert
                                       │       │        │       │        │        │
                                       │      MCP-stdio subprocesses          in-proc   in-proc
                                       │       │        │       │        │        │
                                       ▼       ▼        ▼       ▼        ▼        ▼
                                  fin_data  pdf_report code   news    knowledge sentiment
                                  _server   _server  _server _server  _server   _server
                                  (akshare)(cninfo)(sandbox)(EM/CLS) (FAISS+BM25)(SnowNLP)
                                                                        │
                                                                        ▼
                                                                  bge-reranker
                                                                  cross-encoder
                                                                  + corrective
                                                                    quality signal
                                                                       │
                       short-term ── thread_id ──┐                     │
                       LangGraph checkpointer    │     long-term      │
                       PostgresSaver / SQLite /  │     MemoryStore     │
                       MemorySaver (auto-fb)     │     (user prefs +    │
                                                 │      research log)   │
                                                 ▼                      ▼
                                              postgres                 postgres
                                              (langgraph_*)            (store)
```

**关键设计点**：

| 设计 | 为什么这么做 |
|---|---|
| Supervisor + Specialists（不是 ReAct + 全工具） | 工具集解耦 → 路由更清晰 → 出 bug 时容易归因到具体 specialist |
| 三级模型路由（LIGHT/MEDIUM/HEAVY） | supervisor 走 HEAVY（要规划+综合），specialists 走 MEDIUM（要选工具+理解返回），grader/rewriter 走 LIGHT |
| MCP 协议化工具 | 工具实现可换 Python/Node/Rust；同一份 fastmcp server 可独立测试也可被 agent 远程调用 |
| Hybrid RAG + Rerank + Corrective signal | 单一向量检索召回不稳；BM25 补关键词命中；cross-encoder rerank 提精度；`quality` 标签让 agent 自己决定要不要重写查询 |
| 三档 Checkpointer fallback (PG → SQLite → Memory) | 开发不需要起 docker，生产又能持久化；TCP 探测预检防 Windows ProactorEventLoop 被 psycopg pool 卡死 |
| SSE `stream_mode='updates'` | 直接把 LangGraph 的节点级状态变更转成 handoff/update/final/error/done 五类事件给前端 |

---

## 三、技术栈

| 层 | 选型 |
|---|---|
| 编排 | `langgraph` + `langgraph-supervisor` |
| 工具协议 | `fastmcp` + `langchain-mcp-adapters` |
| RAG | `faiss-cpu`（向量）+ `rank-bm25`（关键词）+ `BAAI/bge-small-zh-v1.5`（embedder）+ `BAAI/bge-reranker-base`（cross-encoder rerank） |
| LLM | `langchain-openai` + `litellm`，三档路由（LIGHT/MEDIUM/HEAVY），按 `AgentName` → `ModelTier` 映射 |
| 持久化 | `langgraph-checkpoint-postgres` / `langgraph-checkpoint-sqlite`；`InMemoryStore` 作长期 user store |
| Web | `fastapi` + `uvicorn` + `sse-starlette`，Pydantic v2 严格校验 |
| 数据源 | `akshare`（A 股行情/基本面/公告/新闻）+ `httpx` + `pypdf`（cninfo PDF 解析）+ `snownlp`（中文情感） |
| 可观测 | `loguru` 结构化日志 + `langsmith` tracing |
| 工程化 | `uv` 包管理 + `ruff`（lint）+ `mypy --strict` + `pytest`（167 tests） |

---

## 四、Quick Start

### 4.1 准备依赖

```bash
# 安装 uv（如已装可跳过）
pip install uv

# 同步依赖（会自动创建 .venv，约 1-2 分钟）
uv sync --extra dev

# 配置环境变量
cp .env.example .env
# 至少填一个 LLM 提供商的 key —— OpenAI / DeepSeek / Dashscope 任选其一
# Postgres/Redis 可不填，应用会自动回退到 SQLite + 内存限流
```

### 4.2 跑测试（推荐第一步：确认绿）

```bash
.venv/Scripts/python.exe -m pytest -m "not network and not slow" -q
# 期望：167 passed
```

`network` 标记下的测试会真打 akshare/cninfo 等外网，`slow` 标记下的测试会下载 bge embedder 权重（~100MB）。这两组在 CI 里默认跳过。

### 4.3 启动 FastAPI

```bash
.venv/Scripts/python.exe -m research_agent.main
# 或者
uv run uvicorn research_agent.main:app --host 0.0.0.0 --port 8080
```

服务起来后：
- `http://localhost:8080/docs` —— OpenAPI 文档（自动生成）
- `GET /health` —— 健康检查，返回 postgres / redis / knowledge_db / research_supervisor 的实时状态

### 4.4 跑一个端到端 demo（含真实业务数据）

```bash
# 1) 先给 FAISS 知识库灌几份真实的 A 股年报（用 cninfo 公开 PDF）
uv run python scripts/seed_real_research_reports.py

# 2) 跑全栈研究 demo（supervisor 会拉通所有 6 个 specialist）
uv run python scripts/demo_full_research.py
```

### 4.5 docker 一键起

```bash
docker compose up -d
# 起：app(8080) + postgres(5432) + redis(6379)
docker compose logs -f app
```

---

## 五、主要 REST 接口

| Method | Path | 说明 |
|---|---|---|
| GET | `/health` | 多依赖健康探测（postgres / redis / faiss / supervisor 图） |
| POST | `/api/supervisor/chat` | Phase-3 演示：minimal supervisor（math/time/text 三个 toy specialist） |
| POST | `/api/supervisor/research` | **同步**调用 research supervisor，返回最终答案 + 经过的 specialist 列表 |
| POST | `/api/supervisor/research/stream` | **SSE 流式**调用 research supervisor（handoff / update / final / error / done） |
| POST | `/api/knowledge/ingest` | 上传 PDF 入 FAISS 私有知识库 |
| POST | `/api/knowledge/search` | 在指定 collection 做 Hybrid + Rerank 检索 |
| GET | `/api/memory/research/{user_id}` | 查询用户最近若干次研究记录（长期记忆） |
| POST | `/api/sentiment/text` | 对一段或一批中文文本做 SnowNLP + 金融词典量化情感 |

请求/响应 Pydantic schema 见 `src/research_agent/api/schemas.py`。

---

## 六、面试演示亮点（建议讲解顺序）

1. **多模型分层路由**（`llm/provider.py` + `llm/tier.py`）
   - 同一个 `ModelRouter` 暴露 `for_agent(AgentName)`，自动映射到 LIGHT/MEDIUM/HEAVY。
   - LLM 失败时自动 fallback 到下一档（`FALLBACK_CHAIN`）。
   - 故事：「supervisor 用 HEAVY，因为它要规划+综合 6 个 specialist；specialist 用 MEDIUM，因为它要从工具菜单挑工具+理解返回；grader/rewriter 用 LIGHT，因为这俩都是分类任务。**单 prompt 全 HEAVY 是浪费；全 LIGHT 又会让 supervisor 路由错乱**。」

2. **Supervisor 模式 + 动态团队**（`graph/research_supervisor.py`）
   - 任何 specialist 的工具集为空就自动从 prompt 里抹掉，避免 `transfer_to_<missing>` 幽灵路由。
   - 故事：「supervisor 永远只看到一份与运行时团队一致的 system prompt — **配置漂移导致幻觉路由的常见坑**」。

3. **Corrective RAG 全套**（`mcp_servers/knowledge_server.py` + `rag/reranker.py`）
   - FAISS 召回（normalize 后的 cosine） + BM25Okapi + RRF 融合 + bge-reranker-base cross-encoder 重排。
   - 每次 search 返回 `quality ∈ {high, medium, low}`；agent prompt 教它 quality 低就改写查询重试，最多 3 轮。
   - 故事：「**Corrective 循环写在 ReAct prompt 里**，因为 specialist 自己看到 quality 信号本身就够触发重试；要做成显式 LangGraph 节点也可以，但增加图复杂度，**当前选择是工程上的取舍**。」

4. **MCP 协议化工具 + 一个真实的取舍**（`mcp_servers/` + `tools/knowledge_tools.py`）
   - 5 个 server 走真 stdio 子进程，1 个（knowledge）走进程内并诚实写在 docstring 里——因为 Windows + Python 3.13 + anyio + heavy ML import 链有死锁。
   - 故事：「**调试这件事是给面试官展示真实排障能力的好素材**：从 `chromadb posthog 守护线程污染 stdout` → 写 `_stdio_firewall` → 迁 ChromaDB → FAISS → 发现 sentence-transformers 仍然干扰 → 工程上决定 in-process 同源代码，保留 MCP 接口作为契约文档。」

5. **三级 Checkpointer 自动 fallback**（`memory/checkpointer.py` + `memory/_pg_reachability.py`）
   - 启动时一个 2s 的 TCP 探测，避开 psycopg ConnectionPool 在 Windows ProactorEventLoop 下卡 25 分钟的坑。
   - 故事：「这是个真事故，故事性强 — `pool` 这种"反复重试"的设计在 happy path 下没问题，但 `/health` 探活落到它身上整个 worker 就挂了。」

6. **FastAPI SSE 流式**（`api/routes/supervisor.py`）
   - 用 `stream_mode='updates'` 把每个节点的状态变更映射成 5 类事件。
   - 故事：「不用 `astream_events` 是因为 supervisor 拓扑下事件太碎，UI 端会被淹没；**节点级 delta 视图刚好对齐"哪个 specialist 在说话"的直觉**。」

---

## 七、已知限制 / 不完整的地方（坦诚版）

| 项 | 现状 | 计划 |
|---|---|---|
| **Reflection 反思循环**（Writer → Reasoner critic → 迭代） | `agents/{writer,reasoner}.py` 文件占位但子图未接入主流程 | 在 supervisor 完成 final synthesis 后挂一个 reflection 子图，最多 2 轮自评/重写 |
| **knowledge_server 不走 MCP-stdio** | Windows asyncio + 重 ML 库 import 死锁，已切到 in-process 同源代码 | 后续可以尝试 fastmcp 的 SSE/HTTP transport，绕开 stdio JSON-RPC 帧 |
| **Redis 引入但未消费** | docker-compose 起了 Redis，但 `RateLimitMiddleware` 仍是进程内 dict | 接 Redis 做分布式限流 + LLM response 缓存 |
| **SSE 无 keep-alive 心跳** | 长任务可能被反向代理（30s/60s）切断 | 每 15s 发一个 `phase=heartbeat` 帧 |
| **没有 LangSmith 自动化评估集** | tracing 已接，但没建 dataset/experiment | 用 LangSmith Evaluation 做 supervisor 路由准确率 + RAG 召回率回归测试 |
| **`pgvector/pgvector:pg16` 引擎未消费** | docker-compose 装了，但 RAG 走 FAISS，pgvector 是预留 | 把长期记忆从 InMemoryStore 升级成 pgvector embeddings + ANN 召回 |

---

## 八、项目布局

```
src/research_agent/
├── agents/              # Specialist 构造器 + 角色 prompt
│   └── specialists.py   # 6 个 build_xxx_expert + KNOWLEDGE/REPORT/NEWS/... PROMPT
├── api/                 # FastAPI 层
│   ├── middleware.py    # Auth + 进程内 RateLimit
│   ├── routes/          # health / knowledge / memory / sentiment / supervisor
│   └── schemas.py       # Pydantic 请求/响应（含 SSE event 模型）
├── graph/
│   ├── minimal_supervisor.py    # Phase-3 教学版（math/time/text 三 toy 工具）
│   └── research_supervisor.py   # Phase-4 产品版（六 specialist + HEAVY supervisor）
├── llm/
│   ├── provider.py      # ModelRouter（基于 LiteLLM，含 fallback chain）
│   ├── tier.py          # ModelTier + AgentName + AGENT_TIER_MAP
│   ├── callbacks.py     # token usage / cost callback
│   └── usage_tracker.py
├── memory/
│   ├── checkpointer.py  # PG → SQLite → Memory 三级 fallback
│   ├── _pg_reachability.py  # TCP 探测预检
│   ├── store.py         # 长期 InMemoryStore（user prefs + recent_research）
│   └── manager.py
├── mcp_servers/         # 6 个 fastmcp server（5 个走 stdio，1 个走 in-process）
│   ├── code_server.py
│   ├── echo_server.py
│   ├── fin_data_server.py
│   ├── knowledge_server.py
│   ├── news_server.py
│   ├── news_sentiment_server.py
│   ├── pdf_report_server.py
│   └── client_factory.py        # 统一 loader（含 in-process knowledge_tools）
├── observability/
│   ├── logging.py       # loguru 配置
│   └── tracing.py       # LangSmith 集成
├── rag/
│   └── reranker.py      # bge-reranker-base cross-encoder（可选 disable）
├── tools/
│   ├── native.py        # toy @tool（calculate / time / word_count）
│   └── knowledge_tools.py  # in-process 暴露 knowledge_server 同名 4 工具
├── config.py            # pydantic-settings（LLM/Database/Observability 三层子配置）
└── main.py              # FastAPI app factory + lifespan + CLI 入口

scripts/                  # 10 demos + 6 smoke tests
tests/                    # 167 unit + 1 integration test
```

---

## 九、贡献 / 反馈

这是一个面向**面试演示和学习**的工程，欢迎围绕以下方向 PR：
- 补全 Reflection 子图
- 把 Redis 接进限流 + LLM cache
- 增加更多业务 specialist（如 macro_expert / fund_flow_expert）
- LangSmith 自动化评估集

---

## License

MIT
