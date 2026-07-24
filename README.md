# 金融多智能体研究系统

> **基于 LangGraph + MCP + Agentic RAG 的多智能体深度研究系统**
> 面向 A 股二级市场研究：行情、基金、披露公告、新闻舆情、研究报告知识库 —— 一个 supervisor + 七个 specialist 协作完成。

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-287%20passing-brightgreen.svg)](#评估体系)
[![Eval Dataset](https://img.shields.io/badge/eval-110%20examples-blueviolet.svg)](#评估体系)

---

## 一、它解决什么问题

「**给我做一份宁德时代 2023 年的业绩简评，对比一下行业平均，再看下最近一周市场对它的舆情情况，并参考我之前上传过的 ESG 报告里讲的碳中和承诺。**」

回答这种**混合数据源、混合粒度的研究问题**，传统做法是：
1. 一个 Agent 拿着一堆工具，靠 prompt 自己分配先调谁——容易乱套，或者一个写死流程的 pipeline——不灵活，加新工具就要重写编排逻辑。

本项目用 **LangGraph Supervisor 模式 + 7 个能力解耦的 specialist** 的方式，让 supervisor 只做"读用户需求 → 拆任务 → 选 specialist → 串联结果"，而 specialist 只做"我擅长这一种事，给我参数我就跑"。

业务侧实打实接了真实数据源：
- **行情/基本面** —— akshare 抓东方财富、雪球、新浪（含分时、龙虎榜、融资融券、股东持仓、概念/行业板块、资金流、港股通等 18 个工具）；
- **基金分析** —— akshare 抓天天基金/东方财富基金频道（ETF/LOF 实时行情、基金净值、持仓、评级、排名等 10 个工具）；
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
                                   ┌──────────────────────────────────────────────────┐
                                   │      research_supervisor  (HEAVY)                │
                                   │   langgraph_supervisor.create_supervisor         │
                                   └───┬───────┬───────┬───────┬──────┬──────┬──────┬─┘
              transfer_to_X tool-calls │       │       │       │      │      │      │
                                       ▼       ▼       ▼       ▼      ▼      ▼      ▼
                                      data    fund  report  coder  news  know-  senti-
                                 _expert _expert _expert _expert _expert ledge  ment
                                  (MEDIUM)                              _expert _expert
                                       │    │     │       │      │      │       │
                                       │   MCP-stdio subprocesses        in-proc  in-proc
                                       ▼    ▼     ▼       ▼      ▼      ▼       ▼
                                  fin_data fund pdf_rpt code  news  knowl-  senti-
                                  _server _server _server _server _server edge   ment
                                  (akshare)(akshare)(cninfo)(sandbox)(EM/CLS) _server _server
                                  18 tools 10 tools                       (FAISS+BM25)(SnowNLP)
                                                                        │
                                                                        ▼
                                                                  bge-reranker
                                                                  cross-encoder
                                                                  + corrective
                                                                   quality signal
                                                                        │
                       short-term ── thread_id ──┐                      │
                       LangGraph checkpointer    │     long-term        │
                       PostgresSaver / SQLite /  │     MemoryStore      │
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
| SSE `stream_mode='updates'` + `subgraphs=True` | 节点级状态变更转成 8 类事件（`handoff` / `update` / `final` / `tool_call` / `review_requested` / `heartbeat` / `error` / `done`）；`subgraphs=True` 透传 specialist 内部工具调用，便于 UI 显示"data_expert 正在调 get_kline" |
| HITL 真实可恢复 | `human_review` 节点调 LangGraph `interrupt()` 把状态完整存进 checkpointer；进程重启都能从同一个 `thread_id` resume，由 `POST .../approve` 或 `.../resume` 续跑 |
| Observability 三件套 | `request_id` 中间件贯穿 HTTP 头 / loguru 日志 / LangSmith trace 标签；`/api/usage` 暴露按 tier 分桶的 token 用量 + 美元成本；`/metrics` 输出 Prometheus 文本（不依赖 `prometheus_client`，零额外开销） |

---

## 三、技术栈

| 层 | 选型 |
|---|---|
| 编排 | `langgraph` + `langgraph-supervisor` |
| 工具协议 | `fastmcp` + `langchain-mcp-adapters` |
| RAG | `faiss-cpu`（向量）+ `rank-bm25`（关键词）+ `BAAI/bge-small-zh-v1.5`（embedder）+ `BAAI/bge-reranker-base`（cross-encoder rerank） |
| LLM | `langchain-openai` 的 `ChatOpenAI` 直接打 OpenAI / DeepSeek / Dashscope 兼容端点；三档路由（LIGHT/MEDIUM/HEAVY），按 `AgentName` → `ModelTier` 映射，自动按 `FALLBACK_CHAIN` 降级 |
| 持久化 | 短期 thread 状态：`langgraph-checkpoint-postgres` → `langgraph-checkpoint-sqlite` → `MemorySaver` 三级 fallback；长期用户记忆：`PostgresStore` → `AsyncSqliteStore` → `InMemoryStore` 三级 fallback（均按 TCP 探测预检自动降级） |
| Web | `fastapi` + `uvicorn` + `sse-starlette`，Pydantic v2 严格校验 |
| 数据源 | `akshare`（A 股行情/基本面/分时/龙虎榜/融资融券/概念板块/行业板块/资金流/港股通 + 基金净值/ETF/LOF/持仓/评级/排名）+ `httpx` + `pypdf`（cninfo PDF 解析）+ `snownlp`（中文情感） |
| 可观测 | `loguru` 结构化日志 + `langsmith` tracing |
| 工程化 | `uv` 包管理 + `ruff`（lint）+ `mypy --strict` + `pytest`（281 tests） |
| 安全 | Prompt 注入检测（`security/prompt_guard.py`）+ 代码沙箱 subprocess 隔离 |
| 协议 | MCP（Agent→Tool，已有）+ A2A（Agent→Agent，`/.well-known/agent.json`） |

---

## 四、Quick Start

### 4.1 准备依赖

```bash
# 安装 uv
pip install uv

# 同步依赖
uv sync --extra dev

# 配置环境变量
cp .env.example .env
# 至少填一个 LLM 提供商的 key —— OpenAI / DeepSeek / Dashscope 任选其一
# Postgres/Redis 若不填，自动回退到 SQLite + 内存限流
```

**常用开关**：

```bash
# 启用人审拦截（draft 出来后等待点 Approve / Revise 后继续）
HITL_ENABLED=true

# 启用 LangSmith 在线 trace（先在 https://smith.langchain.com 拿到 key）
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=ls__xxx
LANGSMITH_PROJECT=research-agent

# 启用 reflection 子图（默认 OFF 省配额；打开后能看到 critic 节点）
REFLECTION_ENABLED=true

# SSE 心跳间隔，0 关闭（ 5s 更直观）
SSE_RESEARCH_HEARTBEAT_SECONDS=5
```

### 4.2 测试

```bash
.venv/Scripts/python.exe -m pytest -m "not network and not slow" -q
# 期望：281 passed, 14 deselected（不带 network/slow）
```

`network` 标记下的测试会真打 akshare/cninfo 等外网，`slow` 标记下的测试会下载 bge embedder 权重（~100MB）。这两组在 CI 里默认跳过。

### 4.3 启动 FastAPI

```bash
.venv/Scripts/python.exe -m research_agent.main
# 或者
uv run uvicorn research_agent.main:app --host 0.0.0.0 --port 8080
```

服务起来后：
- `http://localhost:8080/` —— 内置 Chat UI（静态 HTML，与 API 同端口，无独立前端 dev server）
- `http://localhost:8080/docs` —— OpenAPI 文档（自动生成）
- `GET /health` —— 健康检查，返回 postgres / redis / knowledge_db / research_supervisor 的实时状态

端口一览（`docker compose up` 时）：

| 端口 | 服务 | 说明 |
|---|---|---|
| 8080 | FastAPI + 静态前端 | 唯一对外 HTTP 入口；API + Chat UI 都在这里 |
| 5432 | Postgres | 可选；checkpointer + 长期记忆持久化 |
| 6379 | Redis | 可选；分布式限流（不配则内存限流） |

MCP 工具服务器（`fin_data_server`、`fund_server`、`code_server` 等）不是 HTTP 服务，而是 FastAPI 启动时拉起的 stdio 子进程，通过标准输入/输出与主进程通信，不占用额外端口。

### 4.4 含真实业务数据demo

> **本地知识库**：FAISS 索引文件存在 `./data/knowledge_db/<collection>/` 下，首次部署是空的。`knowledge_expert` 检索空索引会返回零结果。跑前先 seed（或经 `POST /api/knowledge/ingest` 自己灌 PDF），否则 supervisor 路由到 `knowledge_expert` 会拿到空 hit。
>
> Embedder（`BAAI/bge-small-zh-v1.5` ~100 MB）和 Reranker（`BAAI/bge-reranker-base` ~280 MB）首次 ingest 会从 HuggingFace Hub 下载到 `~/.cache/huggingface/`

```bash
# 1) 先给 FAISS 知识库灌几份真实的 A 股年报（用 cninfo 公开 PDF，幂等，可重复运行）
uv run python scripts/seed_real_research_reports.py
# 完成后 ./data/knowledge_db/prod_reports/ 下应该看到 index.faiss + index.pkl

# 2) 跑全栈研究 demo（supervisor 会拉通所有 7 个 specialist）
uv run python scripts/demo_full_research.py
```

### 4.5 docker

```bash
docker compose up -d
# 起：app(8080) + postgres(5432) + redis(6379)
docker compose logs -f app
```

### 4.6 实测基线

下面这组数字来自一次真实端到端运行，可作为机器跑通后的对照基线：

| 指标 | 实测 |
|---|---|
| 单元测试 | **281 passed, 14 deselected**（`-m "not network and not slow"`），耗时 ~26 s |
| 启动到 ready | ~13 s（首次拉 MCP-stdio + 编译 supervisor，含 4 个子进程冷启） |
| `/health` | 单机无 PG / 无 Redis：`status=degraded`，自动落到 `AsyncSqliteSaver` + `InMemoryStore`，**功能完整** |
| Specialist 在线数 | **7 / 7**（fin_data:18 / fund:10 / pdf_report:4 / code:1 / knowledge:4 / news:5 / sentiment:2 = **44 个工具**） |
| 知识库 `prod_reports` 大小 | **822 chunks**（寒武纪 / 中际旭创 / 兆易创新 年报；seed 脚本自动建） |
| 端到端复合查询<br>（"寒武纪近 5 日股价 + 知识库 2024 业务进展"） | **192 s**，`specialists_reached=['data_expert','knowledge_expert']`，`msg_count=12`，输出含 akshare 实时行情 + 年报原文逐字引用 |
| 单次复合查询 LLM 成本 | **~$0.090 USD**（40 calls / 122,310 tokens；heavy 11 + medium 23 + light 6） |
| 模型路由 | 实测路由到 `deepseek-v4-pro`（HEAVY+MEDIUM 共 34 calls / $0.089）+ `qwen3.6-plus`（LIGHT 6 calls / $0.0007） |

> **如何复现**：起服务 → `python scripts/seed_real_research_reports.py`（首次 ~5 min）→ Chat UI 或 `POST /api/supervisor/research` 提同样的复合问题。

---

## 五、主要 REST 接口

### 5.1 健康检查 / 可观测性

| Method | Path | 说明 |
|---|---|---|
| GET | `/health` | 多依赖健康探测（postgres / redis / memory_store / supervisor 图 / specialist roster） |
| GET | `/api/usage` | 累计 LLM token 用量（按 tier / 按模型分桶）+ 美元成本估算 |
| GET | `/metrics` | Prometheus 文本暴露格式（QPS、按 specialist 调用次数、请求耗时直方图） |

### 5.2 Supervisor（核心研究流）

| Method | Path | 说明 |
|---|---|---|
| POST | `/api/supervisor/chat` | minimal supervisor（math/time/text 三个 toy specialist） |
| POST | `/api/supervisor/research` | **同步**调用 research supervisor，返回最终答案 + 经过的 specialist 列表 |
| POST | `/api/supervisor/research/stream` | **SSE 流式**调用（事件：`handoff` / `update` / `final` / `tool_call` / `review_requested` / `heartbeat` / `error` / `done`） |
| POST | `/api/supervisor/research/{thread_id}/approve` | **HITL**：审核通过被 `interrupt()` 暂停的 draft，graph 从断点续跑 |
| POST | `/api/supervisor/research/{thread_id}/resume` | **HITL**：携带 `feedback` 修订意见 resume；feedback 通过 `Command(resume=...)` 注入到 `interrupt()` 返回值 |

### 5.2.1 A2A（Agent-to-Agent 协议）

| Method | Path | 说明 |
|---|---|---|
| GET | `/.well-known/agent.json` | Agent Card 发现：声明本系统能做什么（金融研究 / 知识检索 / 舆情） |
| POST | `/a2a/tasks/send` | 外部 Agent 提交研究任务（异步） |
| GET | `/a2a/tasks/{task_id}` | 查询任务状态（submitted → working → completed/failed） |
| POST | `/a2a/tasks/{task_id}/cancel` | 取消进行中的任务 |

MCP 解决 **Agent → Tool**（supervisor 调 fin_data/code 等工具）；A2A 解决 **Agent → Agent**（别的系统的 Agent 通过标准协议调本系统的研究能力）。

### 5.3 知识库 / 记忆 / 情感

| Method | Path | 说明 |
|---|---|---|
| POST | `/api/knowledge/ingest` | 上传 PDF 入 FAISS 私有知识库 |
| POST | `/api/knowledge/search` | 在指定 collection 做 Hybrid + Rerank 检索 |
| GET | `/api/knowledge/collections` | 列出已建索引 |
| DELETE | `/api/knowledge/collections/{name}` | 删除指定 collection |
| GET | `/api/memory/context` | 取该 user 的长期偏好 + 最近研究摘要（拼到 supervisor preamble） |
| GET | `/api/memory/history` | 该 user 的历次研究记录（thread_id + 查询 + 简介） |
| POST | `/api/memory/preferences` | 写入用户偏好（KV，namespace 按 user_id 隔离） |
| DELETE | `/api/memory/preferences/{key}` | 删除指定偏好 |
| POST | `/api/sentiment/analyze` | 对一段或一批中文文本做 SnowNLP + 金融词典量化情感 |
| GET | `/api/sentiment/report/{symbol}` | 综合舆情报告（拉取相关新闻 + 量化打分） |

请求/响应 Pydantic schema 在 `src/research_agent/api/schemas.py`。HTTP 响应头一律带 `X-Request-ID`，可与 loguru 日志和 LangSmith trace 一一对应。

### 5.4 前端 Chat UI

`src/research_agent/static/index.html` 自带一个零依赖的单文件 Chat UI（HTML + 原生 JS + Fetch+SSE），访问 `http://localhost:8080/` 。侧栏会从 `app.state.available_specialists` 实时显示 specialist 在线状态；触发 HITL 时弹审核面板，Approve / Revise 调用对应 REST 路由。支持多会话并行管理、一键停止正在进行的查询、数据来源可点击跳转至具体页面。

### 5.5 MCP 工具清单

每个 specialist 通过 MCP-stdio 协议连接到专属的工具服务器。以下为当前全部 44 个工具：

#### data_expert — `fin_data_server`（18 个工具，数据源：akshare）

| 工具 | 功能 |
|---|---|
| `fin_get_realtime_quotes` | A 股实时行情快照 |
| `fin_get_kline` | 日/周/月 K 线历史数据 |
| `fin_get_financial_summary` | 财务摘要（每股指标、盈利能力等） |
| `fin_get_individual_info` | 个股基本信息（所属行业、上市日期等） |
| `fin_get_stock_comments` | 个股千股千评（技术面综合评价） |
| `fin_get_board_changes` | 板块异动实时推送 |
| `fin_get_index_components` | 指数成份股列表 |
| `fin_get_hist_min` | 分钟级分时数据 |
| `fin_get_intraday` | 日内分时明细 |
| `fin_get_lhb_detail` | 龙虎榜详情 |
| `fin_get_margin_detail` | 融资融券明细 |
| `fin_get_top_holders` | 十大股东/流通股东 |
| `fin_get_etf_spot` | ETF 实时行情快照 |
| `fin_get_macro_china` | 宏观经济指标（GDP/CPI/PMI 等） |
| `fin_get_concept_board` | 概念板块行情 |
| `fin_get_industry_board` | 行业板块行情 |
| `fin_get_individual_fund_flow` | 个股资金流向 |
| `fin_get_hsgt_flow` | 沪深港通资金流向 |

#### fund_expert — `fund_server`（10 个工具，数据源：akshare）

| 工具 | 功能 |
|---|---|
| `fund_search_fund` | 基金模糊搜索（按名称/拼音） |
| `fund_get_fund_info` | 基金基本信息（类型、规模、经理等） |
| `fund_get_fund_nav` | 基金历史净值（单位/累计净值） |
| `fund_get_fund_etf_spot` | ETF 基金实时行情 |
| `fund_get_fund_lof_spot` | LOF 基金实时行情 |
| `fund_get_fund_etf_hist` | ETF 历史 K 线数据 |
| `fund_get_fund_holdings` | 基金持仓明细（重仓股） |
| `fund_get_fund_rating` | 基金评级（招商/上海证券/济安金信） |
| `fund_get_fund_rank` | 基金业绩排名（按区间收益率） |
| `fund_get_fund_daily` | 开放式基金每日净值列表 |

#### report_expert — `pdf_report_server`（4 个工具）

| 工具 | 功能 |
|---|---|
| `report_search_disclosure` | 巨潮资讯公告搜索 |
| `report_fetch_pdf` | PDF 下载并提取文本 |
| `report_summarize_text` | 长文本摘要（LLM） |
| `report_generate_briefing` | 生成结构化简报 |

#### news_expert — `news_server`（5 个工具）

| 工具 | 功能 |
|---|---|
| `news_get_em_news` | 东方财富个股新闻 |
| `news_get_cls_telegraph` | 财联社电报快讯 |
| `news_get_xueqiu_hot` | 雪球热帖榜 |
| `news_get_baidu_finance` | 百度财经早晚报 |
| `news_search_news` | 综合新闻搜索 |

#### sentiment_expert — `news_sentiment_server`（2 个工具）

| 工具 | 功能 |
|---|---|
| `sentiment_analyze_batch` | 批量文本情感量化打分 |
| `sentiment_get_report` | 个股综合舆情报告 |

#### knowledge_expert — in-process（4 个工具）

| 工具 | 功能 |
|---|---|
| `knowledge_search` | Hybrid RAG 检索（FAISS + BM25 + Rerank） |
| `knowledge_ingest` | PDF 入库 |
| `knowledge_list_collections` | 列出知识库 |
| `knowledge_delete_collection` | 删除知识库 |

#### coder_expert — `code_server`（1 个工具）

| 工具 | 功能 |
|---|---|
| `code_execute_python` | 沙箱 Python 代码执行 |

---

## 六、细节

1. 多模型分层路由 + 熔断器（`llm/provider.py` + `llm/tier.py`）
   - 同一个 `ModelRouter` 暴露 `for_agent(AgentName)`，自动映射到 LIGHT/MEDIUM/HEAVY。
   - LLM 失败时自动 fallback 到下一档（`FALLBACK_CHAIN`）。
   - 熔断器：同一 tier 的模型连续失败 3 次 → 状态变为 OPEN，后续 30 秒内不再尝试该模型，直接走 fallback（避免每次请求都白等超时）。30 秒后进入 HALF_OPEN，允许试探一次；成功则恢复 CLOSED。
   - 故事：「DeepSeek 宕机时，第 4 个用户不应再傻等 30 秒超时 —— 熔断器让 fallback 瞬间生效。」

2. **Supervisor 模式 + 动态团队**（`graph/research_supervisor.py`）
   - 七个 specialist（data / fund / report / coder / news / knowledge / sentiment），任何 specialist 的工具集为空就自动从 prompt 里抹掉，避免 `transfer_to_<missing>` 幽灵路由。
   - 故事：「supervisor 永远只看到一份与运行时团队一致的 system prompt — **配置漂移导致幻觉路由的常见坑**」。

3. **Corrective RAG 全套**（`rag/` 模块拆成 5 个独立职责文件 + `mcp_servers/knowledge_server.py` 复用它们）
   - 召回：FAISS 向量（cosine on normalized vectors）+ `rag/retriever.py` 中的 `BM25Index` → `hybrid_rrf_fuse` 加权 RRF 融合。
   - 重排：`rag/reranker.py` 的 `bge-reranker-base` cross-encoder（按需 disable）。
   - 质量评估：`rag/grader.py` 的 `RetrievalGrader`（query-doc overlap + 平均分数 → high/medium/low 三档）。
   - 改写：`rag/query_rewriter.py` 的 LLM-based `QueryRewriter`，quality=low 时触发，agent prompt 自决要不要重试，最多 3 轮。
   - 故事：「这套抽象之前散落在 `knowledge_server.py` 里，重构成 5 个文件后**每个组件都能独立测**（`tests/unit/test_rag_{retriever,grader,query_rewriter}.py` 共 26 个 case），**而且任何一个组件可独立替换**（比如把 grader 换成 LLM 打分，或 reranker 换成 bge-m3）。」

4. **MCP 协议化工具 + 一个真实的取舍**（`mcp_servers/` + `tools/knowledge_tools.py`）
   - 6 个 server 走真 stdio 子进程（fin_data / fund / pdf_report / code / news / news_sentiment），1 个（knowledge）走进程内并诚实写在 docstring 里——因为 Windows + Python 3.13 + anyio + heavy ML import 链有死锁。
   - `code_server.py` 沙箱：`execute_python` 用 **subprocess** 在独立子进程跑用户代码；内层 **API 白名单**（自定义 `__builtins__` 字典，只允许 `print/range/len/...`，禁止 `open/__import__/eval`）+ 预导入模块（math/statistics/json/collections/itertools）。`execute_python_inproc` 是降级方案（仅白名单、无进程隔离，与改造前相同）。
   - 故事：「从 `chromadb posthog 守护线程污染 stdout` → 写 `_stdio_firewall` → 迁 ChromaDB → FAISS → 发现 sentence-transformers 仍然干扰 → 工程上决定 in-process 同源代码，保留 MCP 接口作为契约文档。」

5. **三级 Checkpointer 自动 fallback**（`memory/checkpointer.py` + `memory/_pg_reachability.py`）
   - 启动时一个 2s 的 TCP 探测，避开 psycopg ConnectionPool 在 Windows ProactorEventLoop 下卡 25 分钟的坑。
   - 故事：「`pool` 这种"反复重试"的设计在 happy path 下没问题，但 `/health` 探活落到它身上整个 worker 就挂了。」

6. **FastAPI SSE 流式**（`api/routes/supervisor.py`）
   - 用 `stream_mode='updates' + subgraphs=True` 把节点级 delta 映射成 8 类事件：`handoff` / `update` / `final` / `tool_call` / `review_requested` / `heartbeat` / `error` / `done`。
   - 故事：「不用 `astream_events` 是因为 supervisor 拓扑下事件太碎，UI 端会被淹没；**节点级 delta 视图刚好对齐"哪个 specialist 在说话"的直觉**。`subgraphs=True` 是 LangGraph 1.0 的能力，让 specialist 内部的工具调用也能透到顶层 SSE，前端能看到'data_expert 正在调 get_kline'这种实时帧。」

7. **Reflection 反思循环**（`graph/reflection.py`，可选启用，见 [ADR-0003](docs/adr/0003-reflection-loop.md)）
   - critic-first 子图：先用 LIGHT 模型给 supervisor 的最终答案打分（faithfulness / citation / completeness / structure / clarity 五维），不达标才用 HEAVY 模型重写，最多 2 轮。
   - 包成父 StateGraph 节点（`supervisor → reflection → END`），checkpointer 挂在父图上 —— **崩在 reflection 中途从 critic 节点恢复，不会从头跑一遍 6 个 specialist**。
   - `finalize` 节点返回**历史最高分草稿**而不是最新草稿，防止"过度修正"造成的回归。
   - 故事：「supervisor 的合成 prompt 已经写得很长了，再塞自检规则会稀释；**质量校验是分类任务，不该和创作任务挤在一个 LLM 调用里** —— 这就是为什么要单独切一个不同 tier 的 critic 出来。」

8. **Human-in-the-Loop 真实可恢复**（`graph/research_supervisor.py` 的 `human_review` 节点 + `api/routes/supervisor.py` 的 approve/resume）
   - `HITL_ENABLED=true` 时，supervisor 出 draft 后进入 `human_review` 节点，调用 LangGraph `interrupt()` 把状态完整 dump 到 checkpointer。
   - SSE 实时探测 `state.next != ()` 并推 `review_requested` 事件，前端弹审核面板。
   - 审核者点 **Approve** 或填写 **Revise feedback** 触发 `POST .../approve` / `.../resume`，通过 `Command(resume={"action":..., "feedback":...})` 把决策注入到 `interrupt()` 的返回值，graph 从断点续跑。
   - **错误码分级**：404（thread 不存在）/ 409（thread 已完成）/ 500（checkpointer I/O 故障），前端可分别提示，不会把"thread 不存在"误显示成"已完成"。
   - 故事：「金融场景禁止 AI 自动出投资建议——这是合规底线。所以 HITL 不是 mock，**进程重启都能从同一个 thread_id resume**，是 checkpointer 必须挂 Postgres / SQLite 而不是 in-memory的原因。」

9. **Observability 三件套**（`api/middleware.py` + `observability/{logging,metrics,tracing}.py` + `api/routes/usage.py`）
   - `RequestIdMiddleware` 给每个请求注入 UUID，并写入 HTTP 响应头 `X-Request-ID`、loguru 日志上下文、LangSmith trace 标签 —— **nginx 一个 ID 就能贯穿到具体 prompt**。
   - `MetricsMiddleware` 累计 QPS、按 specialist 的调用次数、请求耗时直方图，由 `GET /metrics` 输出标准 Prometheus 文本。零额外依赖（手写 `_Counters`，不引 `prometheus_client`）。
   - `UsageCallbackHandler` 走 `run_inline=True` 在 async 路径内联执行（避免 LangChain 默认的 `run_in_executor` 跨线程开销），把每次 `on_llm_end` 的 token 用量按 `(tier, model_name)` 累加 + 用 `MODEL_PRICING` 估成本，由 `GET /api/usage` 暴露。
   - 故事：「LLM 应用'盲目调'的代价太高了——一次实验跑爆几百块的事故每周都在群里看见。**所以 token 和 trace 必须从 day 1 就有，而非等出事再补**。」

10. **Prompt 注入防御（双向）**（`security/prompt_guard.py` + `api/routes/supervisor.py`）
    - **Prompt 注入**：用户在问题里伪装成系统指令（如 "ignore previous instructions"），试图覆盖 supervisor 的系统提示词或泄漏内部配置。
    - **输入规则**（正则，微秒级）：指令覆盖、角色劫持、系统提示词提取、越狱模板（DAN/developer mode）、间接注入标记、编码绕过等 → 命中则 `ThreatLevel.BLOCKED`，返回 HTTP 400。
    - **输出规则**（正则，LLM 返回后、HTTP 响应前）：系统提示词泄漏、API Key/密码泄漏、内部路径泄漏 → 命中则替换为 `[输出已过滤：检测到敏感信息泄漏风险]`。
    - **覆盖端点**：`/chat`、`/research` 的输入+输出均检测；`/research/stream` 仅入口输入检测（SSE 流逐帧过滤待实现）。
    - 故事：「OWASP Agentic AI Top 10 第一条就是 Prompt Injection —— 金融场景不能让恶意用户把 AI 变成'无限制模式'。」

11. **专家输出置信度校验**（`agents/confidence.py`，可选接入 supervisor）
    - 规则层：幻觉模式（编造引用、过度推测）、数值合理性（PE/ROE/股价范围）、与源文本数字一致性。
    - 提供 `build_llm_validation_prompt()` 供 LIGHT 模型做深度语义校验。
    - 返回 `ConfidenceVerdict(score, level, recommendation)`：accept / downweight / reject。

---

## 七、架构决策记录（ADR）

载入项目的非显然决策都写成了 ADR，存放于 [`docs/adr/`](docs/adr/)。

| # | 决定 | 何时该读 |
|---|---|---|
| [0001](docs/adr/0001-faiss-over-chroma.md) | 向量库从 ChromaDB 迁到 FAISS | "为什么不用 Chroma" |
| [0002](docs/adr/0002-knowledge-server-inprocess.md) | knowledge_expert 走 in-process 不走 MCP-stdio | "为什么 6 个 server 里偏偏这一个不走子进程" |
| [0003](docs/adr/0003-reflection-loop.md) | Reflection 作为子图挂 supervisor 后面 | "为什么不把反思放 supervisor prompt 里" |

---

## 八、评估体系

本项目具备完整的量化评估流水线，覆盖路由准确率、回复质量、关键词命中、记忆持久化和工具选择精确度五个维度。

### 评估指标

| 评估器 | 类型 | 衡量什么 |
|---|---|---|
| `routing_accuracy` | 确定性（Jaccard） | 预期 vs 实际 specialist 路由的集合相似度 |
| `reply_quality` | LLM-as-judge（LIGHT） | 相关性、完整性、事实性 1-5 分归一化 |
| `keyword_coverage` | 确定性 | 回复是否包含预期关键词 |
| `tool_selection_precision` | 确定性 | 是否路由了不必要的 specialist（惩罚过度路由） |
| `memory_persistence` | 确定性 | 长期记忆是否在该写入时已写入 |

### 数据集

`evals/datasets/supervisor_routing.json` — **110 条标注样本**，覆盖：

| 类别 | 数量 | 说明 |
|---|---|---|
| `single_route` | 35 | 每个 specialist 5 条（7 个 specialist） |
| `multi_route` | 30 | 2-5 个 specialist 的各种组合（含 fund_expert 跨专家协作 8 条） |
| `edge_case` | 18 | 寒暄、英文、模糊查询、股票/基金代码、空输入 |
| `robustness` | 13 | 口语化、拼写随意、中英混杂 |
| `adversarial` | 8 | prompt injection、角色劫持、恶意代码注入 |
| `memory_test` | 6 | 匿名 vs 登录用户的记忆写入验证 |

### 运行评估

```bash
# 方式 1：LangSmith 在线评估（需要 LANGSMITH_API_KEY）
python -m evals.eval_supervisor

# 方式 2：本地离线评估，输出 JSON 报告（不依赖 LangSmith）
python -m evals.run_local
python -m evals.run_local --limit 10  # 快速验证前 10 条

# 对比两次评估结果（检测 regression）
python -m evals.compare eval_results/baseline.json eval_results/current.json

# 评估器单元测试（32 个，纯离线）
pytest evals/test_evaluators.py -q
```

报告输出到 `eval_results/` 目录，格式示例：

```
======================================================================
  EVALUATION REPORT — 2026-05-28T03:00:00
  Dataset: supervisor_routing.json (110 examples)
======================================================================

Metric                       Mean      Min      Max      Std     N
-----------------------------------------------------------------
  keyword_coverage            0.870   0.000    1.000    0.210   100
  memory_persistence          0.950   0.000    1.000    0.180   100
  routing_accuracy            0.940   0.000    1.000    0.120   100
  tool_selection_precision    0.920   0.000    1.000    0.150   100
======================================================================
```

---

### 8.1 CI/CD

三层 GitHub Actions 流水线：

| 流水线 | 触发条件 | 内容 |
|---|---|---|
| **CI** (`.github/workflows/ci.yml`) | 每次 PR / push main | ruff lint + format check → pytest 单元测试 → 覆盖率 ≥60% |
| **Eval & Network** (`.github/workflows/nightly.yml`) | 手动触发（workflow_dispatch） | `pytest -m network` MCP 烟测；勾选 `run_eval=true` 时额外跑 LangSmith 路由评估 |
| **Docker** (`.github/workflows/docker.yml`) | tag push (`v*`) | Docker 构建 + 缓存（GHCR push 已预留配置） |

Pre-commit 钩子（`.pre-commit-config.yaml`）：ruff check + ruff format + trailing-whitespace + end-of-file-fixer + check-yaml。

```bash
# 安装 pre-commit 钩子
pre-commit install

# 手动跑一次全量检查
pre-commit run --all-files
```

---

### 8.2 可观测性 Dashboard

`docker compose up` 现在同时启动 **Prometheus + Grafana** 监控栈：

| 端口 | 服务 | 说明 |
|---|---|---|
| 9090 | Prometheus | 每 15s 抓取 `/metrics`，存储时序数据 |
| 3000 | Grafana | 预置 dashboard，匿名可读（admin/admin 可编辑） |

预置 Dashboard（`monitoring/grafana/dashboards/research-agent.json`）包含三行面板：

- **HTTP Overview**：请求总数、QPS、平均延迟（按 path）、错误率
- **LLM Usage**：LLM 调用总数、按模型的调用速率、token 消耗（prompt/completion 堆叠）、累计费用（CNY）
- **System**：可用 specialist 数量（Gauge）、请求量 Top 10 端点（表格）

```bash
# 启动完整栈（app + postgres + redis + prometheus + grafana）
docker compose up -d

# 访问
# http://localhost:8080  — 应用 + Chat UI
# http://localhost:3000  — Grafana Dashboard
# http://localhost:9090  — Prometheus
```

---

## 九、已知限制 / 未来扩展

| 项 | 现状 | 计划 |
|---|---|---|
| **knowledge_server 不走 MCP-stdio** | 🟡 工程取舍：Windows + Python 3.13 + asyncio + heavy ML import 链有系统性死锁，已切到 in-process 同源代码——业务逻辑不变，MCP 协议契约保留在 `@mcp.tool` 装饰器上作为文档。详见 [ADR-0002](docs/adr/0002-knowledge-server-inprocess.md) | 若未来需跨进程/跨语言，可换 fastmcp 的 SSE/HTTP transport 绕开 stdio JSON-RPC 帧 |
| **fund_server 仅覆盖公募基金** | 🟡 当前覆盖开放式基金、ETF、LOF 的净值/行情/持仓/评级/排名，尚未接入私募基金和 QDII 专项数据 | 按需扩展 akshare 私募/QDII 接口 |
| **LangSmith 评估集已上 CI** | ✅ `evals/` 下有 100 条标注样本 + 5 个评估器 + 离线评估 runner + regression 对比工具；Nightly CI 自动跑路由评估 | 持续扩充样本 + 增加 RAG 召回率评估 |
| **pgvector 引擎装了但未用作向量搜索后端** | 🟡 设计取舍：docker-compose 用 `pgvector/pgvector:pg16` 是为给 Postgres checkpointer + KV-style 长期记忆提供后端；RAG 走本地 FAISS 因为 demo 不依赖外部服务 | 当用户研究量 >100 条时，把"语义相似历史研究召回"切到 pgvector + ANN |
| **静态知识语义缓存** | ✅ 已落地 | `cache/semantic_cache.py`：glossary/methodology/template/faq/macro/historical_event；L0 精确键 + L1 FAISS 语义；维度过滤 version/locale/prompt_version；research 入口命中则短路 LLM |
| **工具原始数据缓存** | ✅ 已落地 | `cache/tool_cache.py` 按 TTL 分层缓存 MCP 工具成功返回值（realtime 20s / short 2min / medium 5min / daily 1h / long 6h）；错误不入缓存；默认进程内内存，可选 Redis |
| **Reflection 评估 delta** | ✅ 子图已落地（`REFLECTION_ENABLED=true`），但尚未量化 ON/OFF 答案质量差异 | 需 LLM 预算跑 30 examples × 2 组对照 |

---

## 十、项目布局

```
src/research_agent/
├── agents/              # Specialist 构造器 + 角色 prompt
│   ├── __init__.py      # build_xxx_expert + SPECIALIST_BUILDERS 全 re-export
│   ├── confidence.py    # 专家输出置信度校验（规则 + 可选 LLM 深度校验）
│   └── specialists.py   # 7 个 build_xxx_expert + KNOWLEDGE/REPORT/NEWS/FUND/... PROMPT
├── api/                 # FastAPI 层
│   ├── middleware.py    # Auth + RateLimit + RequestTimeout + RequestId
│   ├── dependencies.py  # FastAPI Depends 工厂
│   ├── routes/
│   │   ├── a2a.py       # /.well-known/agent.json + /a2a/tasks/*
│   │   ├── health.py    # /health
│   │   ├── knowledge.py # /api/knowledge/{ingest,search,collections}
│   │   ├── memory.py    # /api/memory/{context,history,preferences}
│   │   ├── sentiment.py # /api/sentiment/{analyze, report/{symbol}}
│   │   ├── supervisor.py# /api/supervisor/{chat,research,research/stream,research/{tid}/approve,…}
│   │   └── usage.py     # /api/usage + /metrics
│   └── schemas.py       # Pydantic 请求/响应（含 SSE 8 类事件枚举）
├── security/
│   ├── prompt_guard.py  # Prompt 注入检测（中英文 15+ 规则 + 金融输出安全 + 免责声明）
│   └── token_quota.py   # Per-user Token 配额管理（Redis / 内存双后端）
├── graph/
│   ├── minimal_supervisor.py    # Phase-3 教学版（math/time/text 三 toy 工具）
│   ├── research_supervisor.py   # Phase-4 产品版（七 specialist + HEAVY supervisor + 可选 reflection + 可选 HITL human_review 节点）
│   └── reflection.py            # Phase-5 critic-first 反思子图（Writer / Reasoner 自评-重写）
├── llm/
│   ├── provider.py      # ModelRouter（三档 tier + with_fallbacks 降级链；用 langchain-openai 直连 OpenAI/DeepSeek/Dashscope 兼容端点）
│   ├── tier.py          # ModelTier + AgentName + AGENT_TIER_MAP
│   └── usage_tracker.py # UsageTracker + UsageCallbackHandler(run_inline=True)
├── memory/
│   ├── checkpointer.py  # PG → SQLite → Memory 三级 fallback
│   ├── _pg_reachability.py  # TCP 探测预检
│   ├── store.py         # 长期 InMemoryStore / SqliteStore / PostgresStore
│   └── manager.py
├── mcp_servers/         # 6 个走 stdio + 1 个 in-process（详见 ADR-0002）
│   ├── code_server.py             # Python 沙箱执行
│   ├── echo_server.py             # 调试用
│   ├── fin_data_server.py         # akshare A 股行情/基本面（18 工具：K 线、分时、龙虎榜、融资融券、股东、ETF、宏观、概念/行业板块、资金流、港股通等）
│   ├── fund_server.py             # akshare 基金分析（10 工具：基金搜索、净值、ETF/LOF 行情、持仓、评级、排名等）
│   ├── knowledge_server.py        # FAISS+BM25 RAG（生产入口在 tools/knowledge_tools.py，in-process）
│   ├── news_server.py             # 东方财富 / 财联社新闻
│   ├── news_sentiment_server.py   # SnowNLP + 金融词典量化情感
│   ├── pdf_report_server.py       # 生成 PDF 简报
│   └── client_factory.py          # 统一 loader（含 in-process knowledge_tools）
├── observability/
│   ├── logging.py       # loguru 配置（注入 request_id 上下文）
│   ├── metrics.py       # Prometheus 文本格式 + MetricsMiddleware（零依赖）
│   └── tracing.py       # LangSmith setup_tracing()，lifespan 启动时自动调用
├── rag/                 # 每个文件一职责
│   ├── retriever.py     # BM25Index + hybrid_rrf_fuse（加权 RRF）
│   ├── grader.py        # RetrievalGrader（high / medium / low 启发式）
│   ├── query_rewriter.py# LLM-based 查询改写（Corrective-RAG 改写一环）
│   ├── reranker.py      # bge-reranker-base cross-encoder（可选 disable）
│   ├── embedder.py / chunker.py / loader.py
│   └── __init__.py      # 全公开 API re-export
├── static/
│   └── index.html       # 零依赖单文件 Chat UI（HTML + 原生 JS + SSE + HITL 审核面板）
├── tools/
│   ├── native.py        # toy @tool（calculate / time / word_count）
│   └── knowledge_tools.py  # in-process 暴露 knowledge_server 同名 4 工具
├── config.py            # pydantic-settings（LLM/Database/Observability 三层子配置）
└── main.py              # FastAPI app factory + lifespan + CLI 入口

evals/                    # 量化评估套件
├── datasets/
│   └── supervisor_routing.json  # 100 条标注样本（6 类 × 多场景）
├── evaluators.py         # 5 个评估器（routing_accuracy / reply_quality / keyword_coverage / memory_persistence / tool_selection_precision）
├── eval_supervisor.py    # LangSmith 在线评估入口
├── run_local.py          # 离线评估 runner（输出 JSON 报告）
├── compare.py            # 评估报告 regression 对比工具
└── test_evaluators.py    # 评估器单元测试（32 个）
eval_results/             # 评估报告输出目录（git tracked）
.github/workflows/        # CI/CD 三层流水线
├── ci.yml                # PR: lint + test + coverage
├── nightly.yml           # 每日: network smoke + eval
└── docker.yml            # Tag: Docker build
monitoring/               # Prometheus + Grafana 可观测性栈
├── prometheus.yml        # Prometheus 采集配置
└── grafana/              # Grafana 预置 dashboard + 数据源
scripts/                  # 10 demos + 6 smoke tests + 1 seed + benchmark_e2e.py
benchmark_results/        # 性能基准测试报告输出目录
docs/
├── architecture.md       # 系统架构设计文档
├── failure-modes.md      # 故障模式分析
└── adr/                  # 架构决策记录（4 篇）
tests/                    # unit（346 passing）+ integration
```

---

## 十一、安全 Guardrails

多层安全防御体系（详见 [ADR-0004](docs/adr/0004-guardrails-security-layers.md)）：

| 层 | 机制 | 位置 |
|---|---|---|
| **输入安全** | PromptGuard 正则引擎：15+ 种中英文注入模式检测（指令覆盖、角色劫持、系统提示词提取、越狱、间接注入、编码绕过） | `security/prompt_guard.py` |
| **输出安全** | 凭据泄漏检测 + 金融不当投资建议检测（"建议买入"、"保证收益"等） | `security/prompt_guard.py` |
| **金融免责声明** | 每次研究回答自动附加 AI 免责声明（同步 + SSE 两种响应模式） | `api/routes/supervisor.py` |
| **Per-user Token 配额** | 滑动窗口 24h 配额（默认 50 万 token/天），支持 Redis 分布式 / 内存兜底 | `security/token_quota.py` |
| **IP 限流** | 滑动窗口 60s RPM 限制，Redis Lua 原子脚本 / 内存兜底 | `api/middleware.py` |
| **认证** | Bearer Token 校验（`API_SECRET_KEY` 环境变量，空值时禁用） | `api/middleware.py` |
| **请求超时** | 可配置 ASGI 层超时，SSE 流豁免 | `api/middleware.py` |
| **Multi-tenant 知识库隔离** | `{user_id}__{collection}` 命名空间前缀；搜索/列表/删除全路径自动 scope，anonymous 用户透明共享 | `tools/knowledge_tools.py` |

## 十二、性能 Benchmark

```bash
# 快速冒烟测试（仅无 LLM 端点）
python scripts/benchmark_e2e.py --quick

# 完整测试（含 LLM 调用，需启动服务）
python scripts/benchmark_e2e.py --concurrency 1,5,10 --iterations 30

# 报告输出到 benchmark_results/（JSON 格式，含 P50/P95/P99 延迟 + 吞吐量）
```

## 十三、架构文档

| 文档 | 内容 |
|---|---|
| [系统架构设计](docs/architecture.md) | 全景图、核心设计决策矩阵、数据流详解、可靠性设计、安全层、可扩展性 |
| [故障模式分析](docs/failure-modes.md) | 12+ 种故障模式矩阵、三级降级策略、可观测性信号、灾难恢复 |
| [ADR-0001: FAISS > Chroma](docs/adr/0001-faiss-over-chroma.md) | 向量存储选型 |
| [ADR-0002: Knowledge in-process](docs/adr/0002-knowledge-server-inprocess.md) | MCP stdio 死锁规避 |
| [ADR-0003: Reflection Loop](docs/adr/0003-reflection-loop.md) | 反思循环设计 |
| [ADR-0004: Guardrails](docs/adr/0004-guardrails-security-layers.md) | 多层安全防御体系 |

## 十四、Roadmap（待做）

- **美股 P1（已完成）**：`us_data_server`（yfinance）+ `us_data_expert` 已挂载（股票/指数/ETF）— 见 [ADR-0006](docs/adr/0006-us-market-parallel-isolation.md)
- **美股 P2（已完成）**：`us_filing_server`（EDGAR）+ `us_filing_expert`（10-K/10-Q/8-K/DEF 14A）
- **美股 P3+**：美股新闻舆情；跨市场 MIXED 编排
- 增加更多业务 specialist（如 `bond_expert` 债券 / `option_expert` 期权）
- RAG 专项评估（retriever recall@k、reranker NDCG）
- knowledge_server 主路径接入 `vector_backend` 抽象层（当前仍直接走 FAISS）
