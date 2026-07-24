# ADR 0006: A 股 / 美股平行隔离与市场判定契约（P0）

## 背景

系统现有工具链（akshare、巨潮、东财、雪球、SnowNLP）全部面向 **A 股**。
产品需要拓展美股，同时要求：

1. PoC 接受 **yfinance**；
2. 一期范围仅 **美股股票 + 指数 + ETF**（不含共同基金 / 期权）；
3. 默认市场跟 **用户 memory 偏好**；用户提问时会说公司/基金名字，**按名字判断市场**；
4. **A 股与美股必须分开**，不能把 `market=` 参数硬塞进现有 `fin_*` / `news_*`。

若在现有 `data_expert` 上增加 `market=US`，会导致：prompt 膨胀、工具幻觉、 交易日历/披露体系缠死、故障难归因。

## 决策

### 1. 市场作为一等公民

引入 `research_agent.market` 包：

| 类型 | 含义 |
|---|---|
| `Market` | `CN_A` / `US` / `MIXED` / `UNKNOWN` |
| `AssetClass` | `equity` / `index` / `etf`（一期） |
| `SymbolRef` | 规范化标的（market + ticker + asset_class + 显示名） |
| `MarketResolution` | 一次问句的判定结果（含 source / confidence / reasons） |

### 2. 判定优先级

1. API 显式 `market` 覆盖（非 `auto`）
2. 问句硬信号：6 位 A 股代码、知名中英文名、市场关键词、白名单内美股 ticker
3. 用户偏好 key：`preferred_market` ∈ {`CN_A`,`US`}
4. 产品默认：`CN_A`（当前已上线工具全集）

### 3. 平行专家团队（后续阶段落地，本 ADR 锁定边界）

```
CN_A: data / fund / report / news / sentiment / knowledge / coder
US:   us_data / us_etf(可并入 us_data) / us_filing / us_news / us_sentiment / knowledge / coder
```

共享层仅限：supervisor 编排、HITL、reflection、RAG **引擎**、tool TTL / semantic cache **框架**、coder。

### 4. P0 行为（判定契约）

- Supervisor 系统提示词写入市场路由约束；
- `/api/supervisor/research` 注入 `[MarketResolution]` 前导，响应带 `market` / `market_source`；
- SSE 响应头：`X-Market` / `X-Market-Source`；
- `POST /api/memory/preferences` 对 `preferred_market` 做枚举校验。

### 5. P1 行为（已落地）

- `us_data_server`（yfinance）暴露 `us_*` 工具；
- `us_data_expert` 挂入 research supervisor roster；
- 判定为 US 时路由至美股行情专家，**禁止**用 A 股 `fin_*` / `news_*` 查美股。

### 5b. P4-ETF 行为（已落地）

- `us_get_etf_holdings`：Yahoo top holdings（重仓股 + 权重）；
- `us_get_etf_sector_weights`：行业权重 + 大类资产占比（`funds_data`）；
- 仍并入 `us_data_expert`，不另开 `us_etf_expert`；**禁止**用 A 股 `fund_*` 查美股 ETF 持仓。

### 6. P2 行为（已落地）

- `us_filing_server`（SEC EDGAR）暴露 `us_filing_*` 工具；
- `us_filing_expert` 挂入 research supervisor roster；
- 美股 10-K / 10-Q / 8-K / DEF 14A 走 EDGAR，**禁止**用巨潮 `pdf_*` 查美股披露。

### 7. P3 行为（已落地）

- `us_news_server`（yfinance Yahoo 新闻 + 可选 8-K 标题）暴露 `us_news_*`；
- `us_sentiment_server`（英文金融关键词词典）暴露 `us_sentiment_*`；
- **禁止**用 A 股 `news_*` / `sentiment_*`（SnowNLP）处理美股新闻舆情。

### 8. P4-语义缓存 US 域（已落地）

- 种子条目增加 ``market``：`CN_A` / `US` / `SHARED`；
- `lookup(..., market=)`：`US` → 仅 US+SHARED，`CN_A` → 仅 CN_A+SHARED；
- 跳过规则扩展：白名单美股 ticker + 英文时效/研究意图模式；
- research 路由先 `resolve_market` 再查缓存，避免美股问句命中 A 股 FAQ。

### 9. P4-Eval（已落地）

- 新增 `evals/datasets/us_market_routing.json`（美股单路由 / 多路由 / 隔离 / 对照）；
- 评估器：`market_routing_accuracy`、`market_isolation`；
- `supervisor_target` 注入 `[MarketResolution]` 前导，与生产对齐；
- 默认本地/LangSmith 评估合并 CN + US 样本集。

## 考虑过的替代方案

| 方案 | 否决理由 |
|---|---|
| 扩展现有 `fin_*` 加 `market` 参数 | 路由与 prompt 不可维护；A/US 故障耦合 |
| 默认美股 | 与现有工具全集不匹配；无信号时应默认已上线市场 |
| 任意大写字母当 ticker | 误报率高；P0 仅白名单 + 知名名表 |

## 后果

**正面**

- 后续 P1+ 可按市场平行加 MCP，不改判定契约；
- 评估集可按 `market` 维度标注误路由。

**负面 / 中性**

- 跨市场 MIXED 深度编排与 UI 市场徽章仍属后续；
- 英文舆情为关键词词典 PoC（可换 VADER / 专用模型）。

## 后续

- **P1（已完成）**：`us_data_server`（yfinance）+ `us_data_expert` + 接入 supervisor roster
- **P2（已完成）**：EDGAR `us_filing_server` + `us_filing_expert`
- **P3（已完成）**：`us_news` / `us_sentiment`
- **P4**（四块并行，完成一块勾一块）：
  - ETF 深化 — **已完成**
  - 语义缓存 US 域 — **已完成**
  - Eval（美股路由 / 隔离评估）— **已完成**
  - UI 市场徽章 — **待做**
