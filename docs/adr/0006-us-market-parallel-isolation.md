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

### 4. P0 行为

- Supervisor 系统提示词写入市场路由约束；
- `/api/supervisor/research` 注入 `[MarketResolution]` 前导，响应带 `market` / `market_source`；
- SSE 响应头：`X-Market` / `X-Market-Source`；
- `POST /api/memory/preferences` 对 `preferred_market` 做枚举校验；
- **尚未**实现 `us_data_server`（P1）；判定为 US 时须告知能力未上线，禁止用 A 股工具查美股。

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

- P0 阶段 US 问句只能「诚实降级」，用户体验不完整；
- 知名名表需持续扩充（或 P1 接 yfinance search）。

## 后续

- **P1**：`us_data_server`（yfinance）+ `us_data_expert` + 接入 supervisor roster
- **P2**：EDGAR `us_filing_server`
- **P3**：`us_news` / `us_sentiment`
- **P4**：ETF 深化、语义缓存 US 域、eval、UI 市场徽章
