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
akshare → 东财/新浪/雪球等            报价：Yahoo Chart → 东财 ulist → yfinance
巨潮 cninfo（披露 PDF）               SEC EDGAR（10-K… + ETF: NPORT-P/N-CSR/485BPOS）
东财/财联社/雪球（新闻）               新闻：Yahoo → 可选 Finnhub → 过滤/聚类/标签
SnowNLP（中文舆情）                   VADER + 金融词表增强（本地打分）
知识库 FAISS（跨市场共享引擎）         同上（共享 RAG 引擎，内容按集合隔离）
```

原则（见 [ADR-0006](adr/0006-us-market-parallel-isolation.md)）：**禁止**用 A 股工具查美股，反之亦然。

---

## 3. A 股数据源（实现）

| 能力 | MCP / 模块 | 数据来源 | 备注 |
|------|------------|----------|------|
| 行情/板块/龙虎榜等 | `fin_data_server` | **akshare**（底层多为东财、新浪等） | 主路径；`get_market_status`（交易日北京时间）：**未开盘 00:00–09:14** / **开盘集合竞价·盘前 09:15–09:25** / **静默 09:25–09:30** / **连续竞价 09:30–11:30、13:00–14:57** / **午休 11:30–13:00** / **收盘集合竞价 14:57–15:00** / **已收盘** / 非交易日 |
| 基金净值/ETF/QDII/私募备案 | `fund_server` | **akshare**（天天基金/东财基金）+ **中基协 AMAC** | 公募主路径；私募为协会备案公示（`search_private_*`），无实时净值 |
| 国内期货/期权 | `derivatives_server` | **akshare**（新浪期货/期权） | 主路径 |
| 公告 PDF | `pdf_report_server` | **巨潮 cninfo** | 主路径；成功返回带 `source=cninfo` |
| 新闻 | `news_server` | 东财 / 财联社 / 雪球等 | 主路径 |
| 舆情 | `news_sentiment_server` | 新闻文本 + **SnowNLP** | 本地模型 |
| 看板热搜/科技股等 | `main.py` 看板 API | 东财 push2 / 新浪等 HTTP（部分 `curl_cffi`） | UI 聚合，与 MCP 同源生态 |
| 看板·国内期货/ETF/QDII | `market/dashboard_extras` → `/api/dashboard` | 新浪期货 / 基金 ETF / 东财开放式 QDII | 首页 A 股区；**QDII 为场外开放式**，按**日增长率**（最近净值日）排 Top，非场内 ETF |
| 看板·美股期货/共同基金 | 同上（并入 `us` 包） | Yahoo `=F` / yfinance NAV | 首页美股区；期权以快捷提问入口 |
| 看板·我的自选 | `watchlist_store` + `watchlist_resolve` → `/api/watchlist` | CN：新浪/净值；US：`_quote_from_ticker` 同源 | 按 `user_id` 持久化；研究 preamble 注入 |

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
| 舆情 | `us_sentiment_server` | 仅 yfinance 拉新闻 + 本地词典打分 | Yahoo + 本地词典（历史） |

文档与 ADR 中写的「PoC 接受 yfinance」即指此阶段。之后已在 **同一 Yahoo 栈内**加了 Chart / Search HTTP 快路径（见 §4.2），**不是**换供应商。

### 4.2 现在的主备逻辑（实现细节）

**没有**在 `.env` / settings 里配置「多供应商开关」。主备写在代码分支里。

#### 报价 / 指数（`us_data_server`）

```
get_quote / get_index_quotes / _quote_from_ticker
    │
    ├─① 优先：Yahoo Chart HTTP
    │     query1.finance.yahoo.com/v8/finance/chart/{symbol}
    │     （curl_cffi 或 requests(trust_env=False)，约数秒）
    │     昨收：日线倒数第二根 Close（勿用易偏离的 chartPreviousClose）
    │
    ├─② 失败再：东财美股 ulist（国内可达；Yahoo 403/限流时看板主力）push2delay.eastmoney.com/api/qt/ulist.np/get
    │     指数：100.SPX / 100.DJIA / 100.NDX(综指) / 100.NDX100(纳指100)
    │     罗素/VIX 无现货码时用 IWM / VIXY 代理并改展示名
    │     个股/ETF：105/106/107.{ticker}
    │
    └─③ 再失败：yfinance（fast_info → history；国内常与①一同失败）
```

看板美股（`main.py` → `_quote_one` / 筛选榜）**复用** `_quote_from_ticker`；Yahoo `yf.screen` 失败时涨跌/活跃榜回退东财 `clist`（`m:105,m:106`）。

东财回退下的三类报价（通用，不只 VIX）：

| 类型 | 含义 | 例子 | 回答要求 |
|------|------|------|----------|
| **直连** | 东财有同标的代码 | `^GSPC→100.SPX`、`AAPL`、`SPY` | 可按指数/个股表述；`source=eastmoney_us` |
| **代理** | 东财无指数现货，用 ETF 近似 | `^VIX→VIXY`、`^RUT→IWM` | 必须写代理名；`proxy=true` + `warning` |
| **不可用** | 东财/Yahoo 皆无 | 部分冷门期权等 | 承认缺失，禁止编造 |

诚实性校验：`research_agent.market.us_source_honesty.find_us_quote_misstatements`（单测覆盖全部代理表项）。

- `get_market_status`：**本地美东时钟**，不请求外网（避免被挂起的 yfinance 占满线程池导致误超时）。
- 日线历史（提问触发）：yfinance → Yahoo Chart HTTP → 东财美股 K 线。
- 公司概况 / ETF sector weights：仍以 **yfinance** 为主。
- ETF holdings（提问触发）：yfinance → Yahoo quoteSummary；东财无稳定美股 ETF 持仓公开兜底。

#### 舆情新闻（`us_sentiment_server`）与新闻专家（`us_news_server`）

概要见下图；**技术细节与 Finnhub 操作指南见 [§4.4](#44-美股新闻管道技术实现与-finnhub-操作指南)**。

```
拉新闻（共用 us_news_pipeline.collect_us_news）
    │
    ├─① Yahoo Search HTTP → 失败再 yfinance.Ticker.news
    ├─② 可选 Finnhub company-news（FINNHUB_API_KEY；无 key 则跳过）
    │
    └─清洗：垃圾源过滤 → 标题相似度聚类 → 关键词事件标签
         代表条带 cluster_size / event_type / event_label_zh

舆情打分（本地，不走 LLM）：
    文本 = 标题 + 摘要；摘要过短且有 URL 时再抓页面 meta/正文前段（非全文）
    sentiment_score = clip(0.6 * VADER.compound + 0.4 * finlex_adjustment, -1, 1)
    model_version = en_vader_finlex_v2
```

- 可选 8-K 标题仍走 EDGAR submissions（`get_recent_8k_headlines`），与 Yahoo/Finnhub 列表并行，不经过聚类管道。

#### 披露

- **仅 SEC EDGAR**，与 Yahoo / Finnhub 无关。
- 主体概况：`get_entity_overview`；名称搜码：`search_entity_by_name`。
- 私募相关：显式 `forms=D,ADV`（Form D 发行备案 / ADV 顾问）；**无**实时 NAV。
### 4.3 名称对照（避免误解）

| 名称 | 是什么 | 主 / 备 | 是否通用搜索 |
|------|--------|---------|--------------|
| **yfinance** | Python 库，封装 Yahoo | 历史/概况/ETF 持仓等仍为主；报价与新闻上多为**后备** | 否 |
| **Yahoo Chart HTTP** | 直连 Yahoo 图表 API | 报价/指数的**优先快路径** | 否（行情接口） |
| **Yahoo Search HTTP** | 直连 Yahoo 搜索接口的 news | 新闻主源之一 | 否（金融域新闻，不是 Google/Bing） |
| **Finnhub company-news** | Finnhub REST，需 API Key | 新闻**第二源**（可选） | 否 |
| **SEC EDGAR** | 美国证监会披露 | 披露主路径 | 否 |

yfinance / Chart / Search = **同一数据商（Yahoo）的不同访问方式**。Finnhub 是**另一家供应商**，目前仅用于新闻，不用于报价主备。

### 4.4 美股新闻管道：技术实现与 Finnhub 操作指南

实现入口：`src/research_agent/mcp_servers/us_news_pipeline.py`。  
调用方：`us_news_server`（`get_ticker_news` / `get_market_news` / `get_etf_news`）与 `us_sentiment_server`（`get_ticker_sentiment_report`）先拉 Yahoo 条目，再交给 `collect_us_news(yahoo_items=...)`；管道内按需再拉 Finnhub 并完成清洗。

#### 4.4.1 端到端流程

```
Yahoo 条目（Search → 失败则 yfinance）
        +
Finnhub 条目（仅当 FINNHUB_API_KEY 非空）
        │
        ▼
   合并（每条带 provider / source）
        │
        ▼
   垃圾源过滤（is_junk_item）——丢掉促销/仙股类
        │
        ▼
   标题/URL 相似度聚类（cluster_news_items）
        │  同簇选「代表条」（权威 publisher 优先）
        │  写出 cluster_size、cluster_urls、providers_in_cluster
        ▼
   关键词事件标签（tag_event）
        │  event_type + event_label_zh
        ▼
   截断到 limit，返回给新闻工具 / 舆情打分
```

| 返回字段（代表条上） | 含义 |
|----------------------|------|
| `title` / `summary` / `publisher` / `url` / `published_at` | 展示与跳转 |
| `provider` / `source` | 该代表条来自哪条拉取通路（如 `yahoo_search`、`finnhub`） |
| `providers_used`（响应顶层） | 本次实际用到的源列表，如 `["yahoo","finnhub"]` |
| `cluster_size` | 簇内原始条数；`>1` 表示多家转载同一事件 |
| `cluster_urls` | 同簇其他链接（最多 3 个），不是删光只留一条 |
| `event_type` | `earnings` / `m_and_a` / `sec_8k` / `analyst` / `legal` / `other` |
| `event_label_zh` | 财报 / 并购 / SEC披露 / 分析师 / 诉讼监管 / 其他 |
| `note`（可选） | 未配置 Finnhub 时的提示文案 |

#### 4.4.2 垃圾源过滤（方案）

函数：`is_junk_item`。命中任一条即丢弃，不进入聚类。

| 规则类型 | 实现要点（见代码常量） |
|----------|------------------------|
| Publisher 黑名单子串 | 如含 `penny stock`、`stockstotrade` 等 |
| URL host 片段 | 如 `pennystock`、`stockpromoter`、`getrich` 等 |
| 标题启发式 | `!!!` ≥ 3；含 `penny stock` / `guaranteed` / `get rich` / `secret stock`；较长全大写英文字母标题 |

目的：减少营销稿与仙股软文污染列表与舆情样本。黑名单可按运营反馈在 `us_news_pipeline.py` 中扩展。

#### 4.4.3 聚类（方案）

- **同簇条件**：URL 完全相同，**或** 标题规范化后 `difflib.SequenceMatcher` 比例 ≥ **0.72**（`_CLUSTER_THRESHOLD`）。
- **标题规范化**：小写、去 URL、去掉标点、压缩空白（保留中英文与数字）。
- **代表条选择**：按 `_TRUSTED_PUBLISHERS` 排序（Reuters / Bloomberg / WSJ / AP / FT / CNBC / MarketWatch 等优先），再看 `published_at`。
- **与「简单去重」的区别**：简单去重只删到一行；聚类保留 `cluster_size` 与 `cluster_urls`，便于判断「是否被多家转载」。

依赖：仅 Python 标准库，无额外 ML 包。

#### 4.4.4 事件标签（方案）

函数：`tag_event`。对 `title + summary` 做英文关键词匹配（**不是** LLM 抽取「谁收购了什么」）。

| `event_type` | 中文标签 | 典型关键词（节选） |
|--------------|----------|-------------------|
| `earnings` | 财报 | earnings, guidance, eps, quarterly results, q1–q4 |
| `m_and_a` | 并购 | acquire, acquisition, merger, buyout, takeover |
| `sec_8k` | SEC披露 | 8-k, sec filing, edgar；或 provider/form 标明 SEC |
| `analyst` | 分析师 | upgrade, downgrade, price target, analyst |
| `legal` | 诉讼监管 | lawsuit, litigation, probe, antitrust, investigation |
| `other` | 其他 | 未命中以上规则 |

误标可能：关键词命中但语境无关；漏标：同义改写未进词表。后续若要「事件抽取」需另上模型，不在本管道范围。

#### 4.4.5 双源合并与降级

| 场景 | 行为 |
|------|------|
| 已配置 `FINNHUB_API_KEY` | Yahoo 与 Finnhub **都拉**（各多拉若干条再聚类），合并后过滤/聚类 |
| 未配置 Key | 只走 Yahoo；响应可带 `note`：未配置第二源 |
| Yahoo 失败、Finnhub 成功 | 仍可能返回新闻（第二源的稳定价值） |
| 两源都失败 | 空列表；工具不因缺 Key 而抛配置异常 |

Finnhub 接口：`GET https://finnhub.io/api/v1/company-news?symbol={T}&from={YYYY-MM-DD}&to={YYYY-MM-DD}&token={KEY}`  
默认窗口约近 **7** 日；字段映射：`headline`→title，`summary`→summary，`source`→publisher，`url`→url，`datetime`→UTC ISO。

配置读取：`Settings.finnhub_api_key` ← 环境变量 `FINNHUB_API_KEY`（见 `config.py` / `.env.example`）。

#### 4.4.6 Finnhub 第二新闻源：操作指南

**1. 申请 Key**

1. 打开 [https://finnhub.io/](https://finnhub.io/) 注册账号。  
2. Dashboard 中复制 **API Key**（免费档通常有每日调用上限，以官网为准）。  
3. 确认套餐含 **company-news**（免费档一般可用；超额会 HTTP 非 200，管道会打日志并跳过该源）。

**2. 写入本机配置**

```bash
# 在项目根目录
cp .env.example .env   # 若尚无 .env

# 编辑 .env，增加或填写：
FINNHUB_API_KEY=你的_finnhub_key
```

保存后**重启** FastAPI / MCP 进程（`get_settings` 有缓存，不重启不会读到新 Key）。

**3. 验证是否生效**

```bash
# 重启服务后，用研究问答或工具：
# 「英伟达最近有哪些新闻？」
# 成功时工具结果常见字段：
#   providers_used: ["yahoo", "finnhub"]  （或其一）
#   news[].cluster_size / event_type / event_label_zh
```

也可用离线单测确认管道逻辑（不依赖真实 Key）：

```bash
uv run pytest tests/unit/test_us_news_pipeline.py -q
```

**4. 配额与排错**

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 只有 `yahoo`，且有 `note` 说未配置 | `.env` 未写或未重启 | 检查 Key、重启进程 |
| 配置了 Key 仍无 finnhub | Key 错、配额用尽、网络拦截 | 看日志 `finnhub company-news HTTP …`；浏览器/ curl 直连 Finnhub 自测 |
| 列表变短 | 过滤 + 聚类压缩 | 正常；看 `raw_count` vs `count` |
| 标签不准 | 关键词规则局限 | 改 `_EVENT_RULES`，或后续上模型抽取 |

**5. 安全注意**

- **不要**把 Key 提交进 Git；只放在本地 `.env` 或部署密钥库。  
- CI / 公开仓库默认不设 Key，测试用 mock，行为与「Yahoo-only」一致。

#### 4.4.7 单测索引

| 文件 | 覆盖 |
|------|------|
| `tests/unit/test_us_news_pipeline.py` | 垃圾过滤、事件标签、聚类代表条、无 Key 降级、Finnhub mock 合并 |
| `tests/unit/test_us_news_offline.py` | `get_ticker_news` 等工具层 mock |
| `tests/unit/test_us_sentiment_offline.py` | 舆情拉新闻 + 打分 |

---

## 5. 为什么说 Finnhub / Polygon / Alpha Vantage 才算「真·多源」？

### 5.1 区别在哪

| 维度 | 报价侧（当前） | 新闻侧（当前） | 完整真·多源（行情） |
|------|----------------|----------------|---------------------|
| 供应商 | 实质 Yahoo + 东财报价兜底 | **Yahoo + 可选 Finnhub** | Polygon / Finnhub 等可配链 |
| 接入形态 | Chart / yfinance / 东财 HTTP | Yahoo Search + Finnhub REST Key | 正式 REST + Key + 配额 |
| 主备含义 | 报价异源已有东财；Yahoo 内多通路 | 新闻异源已可选 Finnhub | 报价/历史/期权等整链可切换 |

「真·多源」在**行情**上仍指：可配置的多家独立供应商冗余。  
**新闻**已实现「Yahoo + Finnhub」异源可选；**报价全链 Finnhub/Polygon 配置化**仍属后续（见 §5.3）。

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

### 5.3 若未来接入**行情**多源，建议形态（尚未实现）

新闻侧 Finnhub 已按 §4.4 落地；下表是**报价/历史**配置化主备的目标形态（当前仓库**无** `US_QUOTE_PROVIDERS`）：

```
get_quote(symbol)
  → provider_chain: [polygon, finnhub, yahoo_chart, yfinance]
  → 第一个成功且字段完整者胜出
  → 响应带 source / as_of / latency
```

配置示例（示意）：

```env
US_QUOTE_PROVIDERS=polygon,finnhub,yahoo_chart
POLYGON_API_KEY=...
# FINNHUB_API_KEY 现已用于新闻第二源；未来也可复用于行情链
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
| 美股报价主备 | `mcp_servers/us_data_server.py`（Chart → 东财 → yfinance；含共同基金/期货/期权） |
| 国内期货/期权 | `mcp_servers/derivatives_server.py` |
| 看板扩展（期货/ETF/QDII/共同基金） | `market/dashboard_extras.py` + `main.py` `/api/dashboard` |
| 看板自选 | `memory/watchlist_store.py` + `market/watchlist_resolve.py` + `api/routes/watchlist.py` |
| 代理/来源诚实性 | `market/us_source_honesty.py` |
| 美股新闻管道（双源/过滤/聚类/标签） | `mcp_servers/us_news_pipeline.py` |
| 美股新闻工具 | `mcp_servers/us_news_server.py`（Yahoo → Finnhub → pipeline） |
| 美股舆情（共用 pipeline + VADER） | `mcp_servers/us_sentiment_server.py` |
| 美股披露 | `mcp_servers/us_filing_server.py` |
| A 股行情 | `mcp_servers/fin_data_server.py` |
| 市场隔离 ADR | `docs/adr/0006-us-market-parallel-isolation.md` |
| 会话粘性市场 | `market/detect.py`（`sticky_market`）+ API `thread_market` |
| 免责去重 / 正负号清洗 | `text/disclaimer.py`、`text/finance_signs.py` |
| 未落地的 web_search 文案 | `agents/retriever.py` |

---

## 8. 尚未做 / 后续可选

| 项 | 现状 |
|----|------|
| 通用联网搜索（Tavily 等） | 未挂载；仅有 `retriever.py` 文案；**不应用作行情主源** |
| Finnhub / Polygon 等**行情**多供应商配置链 | 未接入（新闻侧 Finnhub 已可选，见 §4.4） |
| 英文舆情 FinBERT / 专用 Transformer | 可选；当前为 VADER + 金融词表 + 标题/摘要/正文前段（`en_vader_finlex_v2`） |
| 知识库按市场自动分集合 | 无；靠用户手填 collection 名 |
| 左侧知识库栏按「当前集合」过滤显示 | 无；列出该用户全部集合的 PDF |
| 10-Q/10-K **整篇精读** | 刻意不做：`us_filing_parse_filing_text` 有界窗口（默认 8k 字，最多约 3 窗 / 6 次工具）；宜多轮点名科目追问 |
| 回答截断后**自动续写** | 未做；可用环境变量 `MAX_OUTPUT_TOKENS` 提高单次输出上限 |
| 日线历史 / ETF holdings | **提问触发**（非看板）。日线：yfinance → Yahoo Chart HTTP → 东财美股 K 线；holdings：yfinance → Yahoo quoteSummary（东财无稳定美股 ETF 持仓公开接口，Yahoo 全挂时仍可能空） |
| 私募**实时净值** / 付费 PE 库 | 不做；国内仅 AMAC 备案，美股仅 EDGAR 概况 + Form D/ADV |

## 9. 产品侧已落地（非数据源，但影响体验）

| 项 | 说明 |
|----|------|
| 会话市场粘性 | 跟进句无市场信号时沿用上一轮 `US`/`CN_A`/`MIXED`，避免默认成 A 股 |
| 免责声明去重 | 剥掉模型自写免责后只附加系统一条 |
| 数据来源可点 | 正文 `数据来源：` 自动链东财/Yahoo；缺省时按 `us_*` 工具或美股市场兜底 |
| 涨跌着色 | 美股绿涨红跌；清洗 `-+0.64%` 误号 |
| 对话滚动位置 | 返回看板再进同一会话时恢复离开时的滚动位置 |
| 美股主线 / 日内异动 / 情绪 / 投机面板 | 由行业·主题 ETF、涨跌榜等**本地聚合**；规则近似 |
| 看板·基金/期货/共同基金 | A 股：ETF 双榜、国内期货、**QDII 日涨幅（场外日增长率）**；美股：商品/股指期货、共同基金 NAV；期权快捷提问→`us_get_option_*` |
| 看板·我的自选 | A/美股分区；搜码加入；报价与问答同源（US）；同步记忆并注入研究 preamble |
| A 股市场状态文案 | `not_yet_open`（00:00–09:14）/ `call_auction` 盘前（09:15–09:25）/ `pre_open_silence`（09:25–09:30）/ `trading` / `lunch_break` / `closing_auction`（14:57–15:00）/ `closed` |
| 最终回答清洗 | 剥除「上述分析已完整呈现」等虚指开场；禁止模型假装气泡上方还有分析 |
| 美股共同基金/期货/期权工具 | `us_data_server`；国内衍生品 `derivatives_server` |
| 冷门中文名（别名表外） | **不**靠解析器自动 MIXED；Supervisor + `fin_search_stock_by_name` / `us_search_ticker` 搜码协作（别名表仅加速） |
| A 股 / 巨潮 runtime `source` | `fin_*` 与 `pdf_*` 成功路径带顶层 `source`；SSE `tool_done` + UI `SOURCE_CODE_RULES` 优先展示 |
| 国内私募（AMAC） | `fund_search_private_*` / `get_private_fund_info`：协会备案公示，**无实时净值** |
| 美股私募披露/概况 | `us_filing_get_entity_overview` + `search_filings(forms=D,ADV)` + `search_entity_by_name`；无 NAV |

## 10. 变更记录

| 说明 |
|------|
| 初版：记录 Yahoo 单栈、Chart/Search 快路径、与真·多源及通用搜索的区别 |
| `us_news_*` 与舆情对齐：Search HTTP 优先；§4.1 标明为历史 PoC；代码索引去掉「仍偏 yfinance」 |
| `us_filing_*` 默认纳入 ETF 表单 `NPORT-P` / `N-CSR` / `N-CSRS` / `485BPOS`（`N-PORT` 别名可匹配） |
| 报价链写入东财 ulist；代理诚实性；文档与 ADR-0006 / README 漂移对齐（粘性市场、免责去重、来源链接等） |
| 美股舆情打分：词典 PoC → VADER + 金融词表（`en_vader_finlex_v1`） |
| 舆情打分文本：标题+摘要；摘要短则补抓页面前段（`en_vader_finlex_v2`） |
| §4.4：美股新闻管道技术说明（过滤/聚类/标签）+ Finnhub 第二源操作指南 |
| 美股共同基金/期货/期权 + 国内 fund QDII/经理 + derivatives_server |
| 看板同步：A 股期货/ETF/QDII + 美股期货/共同基金/期权快捷；专家列表含 derivatives_expert |
| 看板自选 `/api/watchlist` + SQLite 持久化 + 记忆注入；QDII 面板改为场外日增长率；A 股时段对齐交易所（集合竞价/静默/收盘竞价）；剥除「上述分析」虚指 |
| architecture 专家↔数据源图补回双市场全工具；A 股 `get_market_status` 取消旧 `pre_market` 清晨窗 |
| 冷门名 Supervisor/搜码协作强化；A 股/巨潮 runtime `source`；国内 AMAC 私募备案 + 美股 EDGAR 概况/Form D·ADV |
