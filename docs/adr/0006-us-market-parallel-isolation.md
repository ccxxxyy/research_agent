# ADR 0006: A 股 / 美股平行隔离与市场判定契约（P0）

## 背景

系统现有工具链（akshare、巨潮、东财、雪球、SnowNLP）全部面向 **A 股**。
产品需要拓展美股，同时要求：

1. 一期可接受 Yahoo 生态（yfinance + Chart/Search HTTP），并允许东财美股作报价兜底（非「真·多源」配置项）；
2. 一期范围仅 **美股股票 + 指数 + ETF**（不含共同基金 / 期权）；
3. 默认市场：问句信号 → 会话粘性 → 用户 memory 偏好 → 产品默认 `CN_A`；用户常说公司/基金名字，**按名字判断市场**；
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
3. 同会话粘性 `thread_market`（问句无信号时沿用上一轮；见请求字段 / `resolve_market(sticky_market=…)`）
4. 用户偏好 key：`preferred_market` ∈ {`CN_A`,`US`}
5. 产品默认：`CN_A`（当前已上线工具全集）

### 3. 平行专家团队（P1～P5 已全部落地）

```
CN_A: data / fund / report / news / sentiment / knowledge / coder
US:   us_data（含 ETF 持仓工具）/ us_filing / us_news / us_sentiment / knowledge / coder
```

共享层仅限：supervisor 编排、HITL、reflection、RAG **引擎**、tool TTL / semantic cache **框架**、coder。

### 4. P0 行为（判定契约）

- Supervisor 系统提示词写入市场路由约束；
- `/api/supervisor/research` 注入 `[MarketResolution]` 前导，响应带 `market` / `market_source`；
- SSE 响应头：`X-Market` / `X-Market-Source`；
- `POST /api/memory/preferences` 对 `preferred_market` 做枚举校验。

### 5. P1 行为（已落地）

- `us_data_server` 暴露 `us_*` 工具；
- `us_data_expert` 挂入 research supervisor roster；
- 判定为 US 时路由至美股行情专家，**禁止**用 A 股 `fin_*` / `news_*` 查美股。
- **报价主备（非多供应商配置项）**：Yahoo Chart HTTP → **东财美股 ulist** → yfinance；
  东财无 VIX/罗素现货时用 VIXY/IWM 代理并改展示名（`proxy` / `warning`）。
  详见 [数据来源说明](../data-sources.md)。

### 5b. P4-ETF 行为（已落地）

- `us_get_etf_holdings`：Yahoo top holdings（重仓股 + 权重）；
- `us_get_etf_sector_weights`：行业权重 + 大类资产占比（`funds_data`）；
- 仍并入 `us_data_expert`，不另开 `us_etf_expert`；**禁止**用 A 股 `fund_*` 查美股 ETF 持仓。

### 6. P2 行为（已落地）

- `us_filing_server`（SEC EDGAR）暴露 `us_filing_*` 工具；
- `us_filing_expert` 挂入 research supervisor roster；
- 美股披露走 EDGAR（普通股 10-K / 10-Q / 8-K / DEF 14A；ETF 另含 NPORT-P / N-CSR / N-CSRS / 485BPOS），**禁止**用巨潮 `pdf_*` 查美股披露。

### 7. P3 行为（已落地）

- `us_news_server`：Yahoo Search HTTP 优先，失败再 yfinance；可选 8-K 标题；暴露 `us_news_*`；
- `us_sentiment_server`：同上拉新闻 + **VADER + 金融词表**打分（`en_vader_finlex_v2`，标题/摘要/必要时正文前段）；暴露 `us_sentiment_*`；
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

### 10. P4-UI 市场徽章（已落地）

- 研究页 topbar 展示解析市场（`CN_A` / `US` / `MIXED`），来自 SSE `X-Market` / `X-Market-Source` 与 final metadata；
- 输入区提供市场覆盖选择（`auto` / `CN_A` / `US` / `MIXED`），强制 `CN_A`/`US` 时写入 `preferred_market`；
- 侧栏专家列表包含 `us_*` 专家。

### 11. P5-MIXED 深度编排（已落地）

- `build_mixed_orchestration_plan`：按双边标的 / 意图生成分侧子任务（experts + instruction）；
- `format_market_preamble` 在 `market=MIXED` 时注入 `[MixedOrchestration]`；
- supervisor 强制按清单分侧移交、禁止跨市场工具、最终分侧再综合；
- `parse_market_override` 支持请求级强制 `MIXED`；
- 评估集：`evals/datasets/mixed_market_routing.json`。

## 考虑过的替代方案

| 方案 | 否决理由 |
|---|---|
| 扩展现有 `fin_*` 加 `market` 参数 | 路由与 prompt 不可维护；A/US 故障耦合 |
| 默认美股 | 与现有工具全集不匹配；无信号时应默认已上线市场 |
| 任意大写字母当 ticker | 误报率高；P0 仅白名单 + 知名名表 |

## 后果

**正面**

- P1～P5 已按市场平行加 MCP，判定契约保持稳定；
- 评估集可按 `market` 维度标注误路由；
- 国内 Yahoo 不可达时，东财报价兜底保证看板与问答可用（需遵守 proxy 诚实表述）。

**负面 / 中性**

- 跨市场 MIXED 深度编排已落地（P5：``[MixedOrchestration]`` 分侧子任务）；
- 英文舆情已为 VADER + 金融词表（`en_vader_finlex_v2`，含摘要/正文前段）；FinBERT 等仍为可选；
- 日线历史 / ETF holdings 仍依赖 yfinance，Yahoo 全挂时可能空。

## 后续（一期范围外 / 可选）

一期 P0～P5 **已全部完成**。可选增强见 [数据来源说明 §8](../data-sources.md)：

- Finnhub / Polygon 等多供应商行情（真·多源）
- 美股共同基金 / 期权（明确不在一期）
- 英文舆情再升级 FinBERT / 专用模型（当前 VADER+finlex 已够用）
- 通用联网搜索（非核心；不能替代行情 API）
