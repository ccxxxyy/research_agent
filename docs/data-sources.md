# 数据来源说明（A 股 / 美股 / 检索）

本文说明本仓库**实际落地**的数据源、主备逻辑，以及与 Finnhub / Polygon / Alpha Vantage 等「真·多源」方案的区别。  
实现代码以 `src/research_agent/mcp_servers/` 为准。

---

## 1. 有没有「通用联网搜索」？

**结论：研究主链路里没有已挂载的通用 Web Search 工具。**

| 位置 | 现状 |
|------|------|
| `src/research_agent/agents/retriever.py` | Prompt **文案**里提到 `web_search` / `vector_search`，属于早期 Retriever Agent 设计草稿 |
| `research_supervisor` 专家表 | **未**挂载独立的 `web_search` / Tavily / SerpAPI / DuckDuckGo 等工具 |
| `knowledge_expert` | 只搜**用户上传知识库**（FAISS + BM25），不是公网搜索 |
| `us_search_ticker` / Yahoo Search HTTP | 是 **Yahoo 金融域**内的标的/新闻检索，**不是**通用网页搜索 |

若以后要做「通用联网搜索」，需要单独接入 Tavily / Brave / SerpAPI 等，并挂到某个 specialist；当前未实现。

---

## 2. 市场隔离总览

```
CN_A（A 股）                                  US（美股）
─────────────────                        ─────────────────
akshare → 东财/新浪/雪球等            Yahoo 体系（yfinance + Chart/Search HTTP）
巨潮 cninfo（披露 PDF）               SEC EDGAR（10-K/10-Q/8-K 等）
东财/财联社/雪球（新闻）               Yahoo 新闻（yfinance / Search HTTP）
SnowNLP（中文舆情）                   英文关键词词典（本地打分）
知识库 FAISS（跨市场共享引擎）         同上（共享 RAG 引擎，内容按集合隔离）
```

原则（见 [ADR-0006](adr/0006-us-market-parallel-isolation.md)）：**禁止**用 A 股工具查美股，反之亦然。

---

## 3. A 股数据源（实现）

| 能力 | MCP / 模块 | 数据来源 | 备注 |
|------|------------|----------|------|
| 行情/板块/龙虎榜等 | `fin_data_server` | **akshare**（底层多为东财、新浪等） | 主路径 |
| 基金净值/ETF | `fund_server` | **akshare**（天天基金/东财基金） | 主路径 |
| 公告 PDF | `pdf_report_server` | **巨潮 cninfo** | 主路径 |
| 新闻 | `news_server` | 东财 / 财联社 / 雪球等 | 主路径 |
| 舆情 | `news_sentiment_server` | 新闻文本 + **SnowNLP** | 本地模型 |
| 看板热搜/科技股等 | `main.py` 看板 API | 东财 push2 / 新浪等 HTTP（部分 `curl_cffi`） | UI 聚合，与 MCP 同源生态 |

A 股侧目前也是「**一个接入层（akshare）+ 多家底层站点**」，不是可配置的多供应商切换。

---

## 4. 美股数据源：最初 vs 现在

### 4.1 最初设计（P1～P3）

下表是**立项时的 PoC 形态**（ADR / 早期 README）；**当前实现以 §4.2 为准**。

| 能力 | 模块 | 当时接入 | 底层 |
|------|------|----------|------|
| 报价 / 历史 / ETF | `us_data_server` | 仅 yfinance | Yahoo Finance |
| 披露 | `us_filing_server` | SEC submissions / company tickers | **EDGAR（官方）** |
| 新闻 | `us_news_server` | 仅 yfinance.Ticker.news | Yahoo |
| 舆情 | `us_sentiment_server` | 仅 yfinance 拉新闻 + 本地词典 | Yahoo + 本地 |

文档与 ADR 中写的「PoC 接受 yfinance」即指此阶段。之后已在 **同一 Yahoo 栈内**加了 Chart / Search HTTP 快路径（见 §4.2），**不是**换供应商。

### 4.2 现在的主备逻辑（实现细节）

**没有**在 `.env` / settings 里配置「多供应商开关」。主备写在代码分支里，且 **Chart / Search 与 yfinance 仍同属 Yahoo**。

#### 报价 / 指数（`us_data_server`）

```
get_quote / get_index_quotes / _quote_from_ticker
    │
    ├─① 优先：Yahoo Chart HTTP
    │     query1.finance.yahoo.com/v8/finance/chart/{symbol}
    │     （curl_cffi 或 requests(trust_env=False)，约数秒）
    │     昨收：日线倒数第二根 Close（勿用易偏离的 chartPreviousClose）
    │
    └─② 失败再：yfinance（fast_info → history）
```

看板美股指数（`main.py` → `_quote_one`）**复用**同一 `_quote_from_ticker`，避免问答与看板涨跌幅口径不一致。

- `get_market_status`：**本地美东时钟**，不请求外网（避免被挂起的 yfinance 占满线程池导致误超时）。
- 日线历史、公司概况、ETF holdings / sector weights：仍以 **yfinance** 为主。

#### 舆情新闻（`us_sentiment_server`）

```
get_ticker_sentiment_report
    │
    ├─① 优先：Yahoo Search HTTP
    │     .../v1/finance/search?q={ticker}&newsCount=...
    │
    └─② 失败再：yfinance.Ticker.news
```

#### 新闻专家（`us_news_server`）

```
get_ticker_news / get_market_news / get_etf_news
    │
    ├─① 优先：Yahoo Search HTTP（与舆情对齐）
    └─② 失败再：yfinance.Ticker.news
```

- 可选 8-K 标题仍走 EDGAR submissions（`get_recent_8k_headlines`）。

#### 披露（不变）

- **仅 SEC EDGAR**，与 Yahoo 无关。

### 4.3 名称对照（避免误解）

| 名称 | 是什么 | 主 / 备 | 是否通用搜索 |
|------|--------|---------|--------------|
| **yfinance** | Python 库，封装 Yahoo | 历史/概况/ETF 持仓等仍为主；报价与新闻上多为**后备** | 否 |
| **Yahoo Chart HTTP** | 直连 Yahoo 图表 API | 报价/指数的**优先快路径** | 否（行情接口） |
| **Yahoo Search HTTP** | 直连 Yahoo 搜索接口的 news | 新闻 + 舆情的**优先快路径** | 否（金融域新闻，不是 Google/Bing） |
| **SEC EDGAR** | 美国证监会披露 | 披露主路径 | 否 |

三者（yfinance / Chart / Search）= **同一数据商（Yahoo）的不同访问方式**，不是三家独立行情商。

---

## 5. 为什么说 Finnhub / Polygon / Alpha Vantage 才算「真·多源」？

### 5.1 区别在哪

| 维度 | 当前（Yahoo 单栈） | Finnhub / Polygon / Alpha Vantage 等 |
|------|-------------------|-------------------------------------|
| 供应商数量 | 行情侧实质 **1 家**（Yahoo） | 可配置 **多家**，故障可切换 |
| 接入形态 | 非官方库 + 非官方 HTTP（易限流/改版/假 delisted） | 多为**正式 REST API + Key**，有配额与文档契约 |
| 配置方式 | 代码写死优先顺序 | 通常 `.env` 配 `PROVIDER=...` / 多 Key |
| 主备含义 | 同源不同通路（Chart vs yfinance） | **异源**：A 挂了用 B 的报价 |
| 合规与 SLA | 免费 PoC，无 SLA | 可买付费档，延迟/完整性更可控 |
| 覆盖面 | 股票/指数/ETF 够用；期权/逐笔等弱 | 各家强弱不同（期权、外汇、基本面字段等） |

「真·多源」指的是：**独立供应商冗余**，不是「同一个 Yahoo 再开一条 HTTP」。

### 5.2 是否一定更准确？

**不一定。**「更准确」要分字段：

| 场景 | 说明 |
|------|------|
| 常规收盘价 / 日线 OHLC | 各家通常对齐交易所官方收盘；Yahoo 与付费源在**日线收盘**上多数一致 |
| 盘中实时、盘前盘后 | 免费 Yahoo 常为**延迟**；付费源可提供更低延迟或官方授权 feed，这时差距明显 |
| 公司行为（拆股、分红调整） | 各家复权口径可能不同，需看是否 adjusted |
| 新闻 / 舆情 | 源不同 → 标题集合不同，不是「谁更准」而是「覆盖是否够」 |
| 披露正文 | 仍以 **EDGAR** 为准；行情商替代不了 SEC |

因此：引入 Finnhub 等，主要价值是 **稳定性、配额、延迟、字段完整性与可运维的主备**，而不是默认「数字一定比 Yahoo 更对」。  
日线研究 PoC 用 Yahoo 通常够用；要做生产级实时或强 SLA，再上多源更合适。

### 5.3 若未来接入多源，建议形态（尚未实现）

```
get_quote(symbol)
  → provider_chain: [polygon, finnhub, yahoo_chart, yfinance]
  → 第一个成功且字段完整者胜出
  → 响应带 source / as_of / latency
```

配置示例（示意，**当前仓库无此配置**）：

```env
US_QUOTE_PROVIDERS=polygon,finnhub,yahoo_chart
POLYGON_API_KEY=...
FINNHUB_API_KEY=...
```

---

## 6. 和「搜索类工具」的边界

| 工具类型 | 例子 | 用途 |
|----------|------|------|
| **行情数据 API** | Yahoo Chart、yfinance、Polygon、Finnhub | 价格、成交量、基本面字段 |
| **金融域搜索** | Yahoo Search news、`us_search_ticker` | 找 ticker / 拉相关新闻标题 |
| **通用联网搜索** | Tavily、SerpAPI、Brave（**未接入**） | 任意网页/资讯，不保证是交易所级行情 |
| **知识库检索** | `knowledge_search` | 仅用户上传文档 |

通用搜索**不能**替代行情 API：搜到的网页价格可能过时、错误或无时间戳；研究回答应优先走行情/披露工具。

---

## 7. 代码索引（便于跳转）

| 主题 | 路径 |
|------|------|
| 美股报价主备 | `mcp_servers/us_data_server.py`（`_quote_via_yahoo_chart`、`_quote_from_ticker`） |
| 美股舆情新闻主备 | `mcp_servers/us_sentiment_server.py`（`_fetch_news_via_yahoo_search`） |
| 美股新闻主备（已与舆情对齐） | `mcp_servers/us_news_server.py`（`_fetch_news_via_yahoo_search` → yfinance） |
| 美股披露 | `mcp_servers/us_filing_server.py` |
| A 股行情 | `mcp_servers/fin_data_server.py` |
| 市场隔离 ADR | `docs/adr/0006-us-market-parallel-isolation.md` |
| 未落地的 web_search 文案 | `agents/retriever.py` |

---

## 8. 尚未做 / 后续可选

| 项 | 现状 |
|----|------|
| 通用联网搜索（Tavily 等） | 未挂载；仅有 `retriever.py` 文案 |
| 美股共同基金 / 期权 | 明确不在一期范围 |
| ETF 专属 EDGAR 表单（N-PORT / N-CSR / 485BPOS） | `us_filing_*` 默认不含 |
| Finnhub / Polygon 等多供应商 | 未接入 |
| 知识库按市场自动分集合 | 无；靠用户手填 collection 名 |
| 左侧知识库栏按「当前集合」过滤显示 | 无；列出该用户全部集合的 PDF |

## 9. 变更记录

| 说明 |
|------|
| 初版：记录 Yahoo 单栈、Chart/Search 快路径、与真·多源及通用搜索的区别 |
| `us_news_*` 与舆情对齐：Search HTTP 优先；§4.1 标明为历史 PoC；代码索引去掉「仍偏 yfinance」 |
