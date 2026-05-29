# 系统架构设计文档

## 一、系统概览

Research Agent 是一个面向 A 股市场的多智能体金融研究系统。基于 LangGraph Supervisor 模式编排 6 个专业化 Agent，通过 MCP 协议对接外部数据源，结合 Corrective RAG、三级 LLM 路由、HITL 审批工作流，为用户提供有引用来源的结构化研究报告。系统在安全层面实施了 Prompt 注入检测、输出过滤、限流、Token 配额等多层防御；在可靠性层面实现了熔断器、多级降级、优雅启动等生产级模式。

## 二、架构全景图

```
                                    用户
                                      │
                                      ▼
                         ┌──── FastAPI HTTP ────┐
                         │  CORS │ Auth │ Rate  │
                         │  Limit │ Timeout │   │
                         │  RequestId │ Metrics │
                         └──────────┬───────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               │                    │                     │
               ▼                    ▼                     ▼
        /api/supervisor       /api/knowledge         /api/memory
        /research             /api/sentiment         /api/usage
        /research/stream                             /metrics
               │
               ▼
    ┌─── PromptGuard ───┐     输入安全过滤
    │  中英文注入检测     │     （规则引擎，微秒级）
    └────────┬──────────┘
             │
             ▼
    ┌── Research Supervisor ──────────────────────────────────┐
    │   (HEAVY LLM — deepseek-v4-pro)                        │
    │                                                         │
    │   分析用户请求 → 识别子问题 → 规划移交序列               │
    │                                                         │
    │   ┌──────────┬──────────┬──────────┬──────────┬────────┐│
    │   ▼          ▼          ▼          ▼          ▼        ▼│
    │ data_     report_    coder_   knowledge_ news_   sentiment_
    │ expert    expert     expert   expert     expert   expert │
    │ (MEDIUM)  (MEDIUM)  (MEDIUM)  (MEDIUM)  (MEDIUM) (MEDIUM)│
    │   │          │          │          │        │        │   │
    │   ▼          ▼          ▼          ▼        ▼        ▼   │
    │ akshare  巨潮PDF    Python    FAISS+BM25  东财/   SnowNLP │
    │  MCP      MCP      sandbox    Corrective  财联社   确定性  │
    │ (stdio)  (stdio)    MCP       RAG        MCP     模型    │
    │                    (stdio)   (in-proc)  (stdio)          │
    └──────────────────────────────────────────────────────────┘
             │
             ▼ (可选)
    ┌── Reflection 子图 ──┐
    │  Critic → Writer    │    评审优先拓扑：
    │  评分 < 0.85 → 重写  │    最多 2 轮迭代
    └────────┬────────────┘
             │
             ▼ (可选)
    ┌── HITL 审批 ────────┐
    │  interrupt() 暂停    │    前端收到 review_requested SSE
    │  人工 approve/resume │    通过 API 继续或修改
    └────────┬────────────┘
             │
             ▼
    ┌── PromptGuard ───┐     输出安全过滤
    │  泄漏检测 + 金融   │     + 金融免责声明
    │  合规规则          │
    └────────┬──────────┘
             │
             ▼
         JSON / SSE 响应 → 用户
```

## 三、核心设计决策

| 决策 | 备选方案 | 选择 | 理由 |
|------|---------|------|------|
| **Agent 编排模式** | 单 Agent + 全部工具 / ReAct 链 / 多 Agent Supervisor | LangGraph Supervisor 模式 | 工具集不相交消除路由歧义；专家职责单一，prompt 可精调；失败隔离在单个专家内 |
| **RAG 向量存储** | Chroma / pgvector / FAISS | FAISS (本地) | Windows + Python 3.13 + MCP stdio 下 Chroma 的遥测线程破坏 JSON-RPC 通道（见 ADR-0001）；FAISS 零依赖、单文件持久化，适合 demo 规模 |
| **检索策略** | 单路纯向量 / 纯 BM25 | 混合 RRF (向量+BM25) + CrossEncoder 重排序 + Corrective RAG | 向量擅长语义、BM25 擅长精确术语；RRF 融合互补；重排序提升 top-k 精度；纠正式循环在质量低时自动重写查询 |
| **MCP 协议** | 直接函数调用 / HTTP API / gRPC | MCP stdio (+ in-process fallback) | 标准化工具通信协议；子进程隔离防止 import 污染主进程；knowledge_server 因 ML 权重加载过重改用 in-process（见 ADR-0002） |
| **LLM 路由** | 单一模型 / 手动切换 | 三级自动路由 (LIGHT/MEDIUM/HEAVY) + 熔断器 + fallback | Supervisor 需要强推理用 HEAVY，专家只做工具调用用 MEDIUM，轻量任务用 LIGHT；熔断器防止不可用提供商拖累延迟 |
| **HITL 审批** | 无审批 / 全量审批 | 可选 interrupt() 暂停 | 金融场景需要人工确认关键结论；通过配置开关控制，不影响无审批流程的性能 |
| **检查点存储** | 纯内存 / 纯 PostgreSQL | PostgreSQL → SQLite → Memory 三级 fallback | 生产用 PG 保证持久化；开发环境无 PG 时 SQLite 兜底；最差情况内存保证可用 |
| **限流方案** | Nginx 层限流 / 纯内存 | Redis 有序集合 + Lua 原子脚本 (降级到内存) | 支持多实例分布式计数；Redis 不可用时透明降级到进程内 dict |

## 四、数据流详解

以 `"分析宁德时代 2023 年业绩 + ESG 披露中提到的碳中和承诺"` 为例：

```
1. API 层接收请求
   → PromptGuard.check_input() 通过
   → 加载用户长期记忆上下文（偏好 + 近期研究历史）
   → 注入为 SystemMessage 前导

2. Supervisor 分析（HEAVY LLM）
   识别 4 个子问题：
   ├── 财务摘要 → data_expert
   ├── 年报披露 → report_expert
   ├── 衍生指标计算 → coder_expert
   └── ESG 知识库检索 → knowledge_expert

3. 串行移交（每次一个专家）
   ① transfer_to_data_expert
      → fin_get_financial_abstract("300750") → 返回营收/利润/现金流
   ② transfer_to_report_expert
      → pdf_search_announcements("300750", "2023年报")
      → pdf_parse_pdf_pages(url, pages=[12,13,14]) → 提取经营情况章节
   ③ transfer_to_coder_expert
      → code_execute_python("计算同比增长率") → 返回精确数值
   ④ transfer_to_knowledge_expert
      → knowledge_search("碳中和承诺", collection="esg")
      → quality="high" → 返回匹配段落（无需重写查询）

4. Supervisor 综合
   → 根据 4 个专家的输出撰写结构化报告
   → 包含 "### 核心发现" + "### 数据来源"

5. (可选) Reflection 子图
   → Critic 评分 0.82 < 0.85 → 触发 Writer 重写
   → 第二轮 Critic 评分 0.91 → 通过

6. (可选) HITL 审批
   → interrupt() 暂停 → SSE 发送 review_requested
   → 用户 approve → 继续

7. 输出处理
   → PromptGuard.check_output() 通过
   → 附加金融免责声明
   → 保存到长期记忆
   → 返回 JSON / SSE 响应
```

## 五、可靠性设计

### 5.1 降级链

| 组件 | 主要 | 降级 1 | 降级 2 | 最终兜底 |
|------|------|--------|--------|---------|
| LLM | deepseek-v4-pro | 熔断后 fallback 到 MEDIUM 模型 | — | 抛错（不静默返回错误答案） |
| 检查点存储 | PostgreSQL | SQLite (本地文件) | InMemoryStore | 启动时一次性决定 |
| 长期记忆 | PostgreSQL Store | SQLite Store | InMemoryStore | 同上 |
| 限流后端 | Redis | 进程内 dict | — | 始终可用 |
| Token 配额 | Redis | 进程内 dict | — | 始终可用 |
| MCP 工具发现 | 全部 6 个 specialist | 部分成功则仅注册可用专家 | — | 全部失败 → 503 |

### 5.2 熔断器

每个 LLM 层级（LIGHT/MEDIUM/HEAVY）配备独立的 `CircuitBreaker`：
- **CLOSED** → 正常通过，连续失败 3 次后 → **OPEN**
- **OPEN** → 直接跳过主模型，走 `with_fallbacks` 降级链；30s 后 → **HALF_OPEN**
- **HALF_OPEN** → 允许一次试探请求，成功则 → CLOSED，失败则 → OPEN

### 5.3 优雅启动

`lifespan` 中 MCP 工具发现使用 `asyncio.gather(return_exceptions=True)`：任何子进程失败不阻塞其他 specialist 的加载。Supervisor prompt 动态裁剪——只列举实际可用的专家，避免生成指向不存在专家的 `transfer_to_<missing>` 调用。

## 六、安全层

| 层 | 机制 | 位置 | 触发行为 |
|----|------|------|---------|
| **输入注入检测** | 正则规则引擎（中英文 15+ 模式） | `PromptGuard.check_input()` | BLOCKED→400, SUSPICIOUS→日志 |
| **输出泄漏检测** | 系统提示逐字泄漏、凭据泄漏、路径泄漏 | `PromptGuard.check_output()` | BLOCKED→内容替换 |
| **金融合规** | 不当投资建议检测 + 金融免责声明自动附加 | 输出规则 + 路由后处理 | SUSPICIOUS→日志；所有研究输出附带免责声明 |
| **认证** | Bearer token（`API_SECRET_KEY`） | `AuthMiddleware` | 401 |
| **限流** | 滑动窗口 RPM（Redis/内存） | `RateLimitMiddleware` | 429 + Retry-After |
| **Token 配额** | Per-user 24h Token 预算 | `TokenQuotaManager` | 429 |
| **请求超时** | ASGI 层挂钟超时 | `RequestTimeoutMiddleware` | 504 |

## 七、可扩展性

### 添加新 Specialist

1. 在 `agents/specialists.py` 中添加 `build_xxx_expert()` 构造器
2. 在 `mcp_servers/` 中添加 MCP 服务器（或 in-process 工具）
3. 在 `client_factory.py` 中添加 `load_xxx_tools()` 加载器
4. 在 `main.py` 的 `_try_build_research_supervisor()` 中注册
5. 在 `research_supervisor.py` 中添加对应的 `SUPERVISOR_PROMPT_XXX`

Supervisor prompt 会自动裁剪——新专家的工具发现失败不影响其他专家。

### 水平扩展

- 无状态 API 层：多实例部署，共享 PostgreSQL 检查点 + Redis 限流
- MCP 子进程随主进程启动，无需独立部署
- 长期记忆通过 PostgreSQL Store 跨实例共享
- 限流通过 Redis 跨实例协同
