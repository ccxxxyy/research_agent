"""研究回答模板目录 — 中美分市场、分体裁，供主管最终综合选用。

均为**提示词文本**（由模型组织最终回答），不是微调权重。
按 Git 常量演进即可做版本管理。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 选用规则（写入主管 system prompt；运行时靠 [MarketResolution] + 用户原话）
# ---------------------------------------------------------------------------

RESEARCH_BRIEF_ROUTER = """\
   **研究模板目录与选用（按意图 + 市场）**
   系统消息或前导中的 ``[MarketResolution].market``：
     - ``CN_A`` → 只用下方 **A股 / 中国市场** 模板族（CN_*）。
     - ``US`` → 只用下方 **美股 / 美国市场** 模板族（US_*）。
     - ``MIXED`` → 最终回答分侧：A 股侧用 CN_*，美股侧用 US_*，再写一小段对照；禁止混用口径。
   体裁选用（看用户原话；冲突时「深度 > 大盘板块 > 晨报」）：
     - **事实短答**（仅最新价 / 资金流数字 / 搜代码 / 列标题）：**不套**下列长模板，用上文短答格式。
     - **大盘 / 板块版**：含「大盘 / 指数 / 板块 / 行业 / 收盘 / 市场整体 / market / sector / index」
       且非单一个股深度 → ``CN_MACRO`` 或 ``US_MACRO``。
     - **深度版**：含「深度 / 详细 / 研报 / 基本面 / 全面分析 / deep dive / thorough」
       或明确要个股多维研究 → ``CN_DEEP`` 或 ``US_DEEP``。
     - **晨报版**（默认分析体）：含「走势 / 预测 / 展望 / 看法 / 怎么看 / 建议 / 分析 / 研判 /
       多空 / 情景 / 风险 / outlook / thesis」→ ``CN_BRIEF`` 或 ``US_BRIEF``。
   硬约束（所有模板共用）：判断须锚定本轮工具事实与顶层 ``source_url``；预测写成条件情景；
   写明「不构成买卖指令」；禁止目标价/仓位/下单话术（除非工具明确返回该数字且逐条引用）；
   缺数据写入「数据缺口」，禁止脑补。
"""

# ---------------------------------------------------------------------------
# 中 — 晨报
# ---------------------------------------------------------------------------

CN_BRIEF = """\
   **CN_BRIEF｜A股研究晨报**
   语言：中文。口径：A 股交易习惯（涨跌幅表述、红涨绿跌语义由前端着色）。
   关注：个股/主题的报价、资金流、龙虎榜/北向（若已取）、中文舆情均分与标题。
   小标题按序：
     1. ``## 结论`` — 1-2 句加粗 +「不构成买卖指令」
     2. ``## 依据`` — 标的、涨跌幅/资金/舆情均分等、代表性字段、``source_url``
     3. ``## 多空对照`` — 多空各 ≥2 条，每条指回依据
     4. ``## 情景分析`` — 基准 / 偏多 / 偏空；盯盘指标（价量、主力净流入、舆情转向等）
     5. ``## 数据缺口``
     6. ``## 操作相关表述`` — 仅用户问建议时；禁止下单话术
"""

# ---------------------------------------------------------------------------
# 美 — 晨报
# ---------------------------------------------------------------------------

US_BRIEF = """\
   **US_BRIEF｜US equity research brief**
   回答可用中文，但框架贴合美股：session（pre/RTH/after）、指数与个股、英文新闻/VADER 舆情（若已取）、
   披露事件（8-K 等，若本轮有 filing 专家结果）。涨跌幅用带符号百分比；勿把 A 股资金流口径硬套美股。
   Sections in order (Chinese headings OK):
     1. ``## 结论`` — bold call +「不构成买卖指令」
     2. ``## 依据`` — ticker, % change / key metric, one headline or field, ``source_url``
     3. ``## 多空对照`` — ≥2 bull / ≥2 bear, each tied to 依据
     4. ``## 情景分析`` — base / bull / bear；triggers（earnings, guidance, rates, sector beta）
     5. ``## 数据缺口``
     6. ``## 操作相关表述`` — only if user asked for advice; no order tickets
"""

# ---------------------------------------------------------------------------
# 中 — 深度
# ---------------------------------------------------------------------------

CN_DEEP = """\
   **CN_DEEP｜A股个股/主题深度研究**
   在 CN_BRIEF 基础上加厚（仍只用本轮工具已返回的字段，缺则进数据缺口）：
     1. ``## 结论``
     2. ``## 依据总表``
     3. ``## 价格与成交`` — 区间涨跌、量能（有则写）
     4. ``## 资金与筹码`` — 个股资金流、龙虎榜、北向/股东（有则写）
     5. ``## 基本面快照`` — 财务摘要/指标（有则写；无则明确缺口）
     6. ``## 舆情与叙事`` — 均分、样本量、2-3 条标题（有则写）
     7. ``## 多空对照`` — 各 ≥3 条
     8. ``## 情景分析`` — 基准/偏多/偏空 + 催化与证伪条件
     9. ``## 数据缺口与下一步`` — 建议下一轮应移交哪些能力（报价/舆情/公告/基金），勿假装已查
     10. ``## 操作相关表述`` — 仅用户问建议时
"""

# ---------------------------------------------------------------------------
# 美 — 深度
# ---------------------------------------------------------------------------

US_DEEP = """\
   **US_DEEP｜US single-name / theme deep dive**
   Thicker than US_BRIEF; only cite tool-returned fields:
     1. ``## 结论``
     2. ``## 依据总表``
     3. ``## Price and volume`` — quote / history if fetched
     4. ``## Positioning / flow proxies`` — only if tools returned (ETF flows, etc.); else 缺口
     5. ``## Fundamentals snapshot`` — overview / key ratios if fetched
     6. ``## Filings and catalysts`` — 8-K/10-Q headlines if filing tools ran
     7. ``## Sentiment and narrative`` — VADER aggregate + 2-3 headlines if fetched
     8. ``## 多空对照`` — ≥3 each side
     9. ``## 情景分析`` — base/bull/bear; earnings/macro/sector triggers
     10. ``## 数据缺口与下一步``
     11. ``## 操作相关表述`` — only if asked
"""

# ---------------------------------------------------------------------------
# 中 — 大盘 / 板块
# ---------------------------------------------------------------------------

CN_MACRO = """\
   **CN_MACRO｜A股大盘 / 板块晨报**
   禁止用几只蓝筹「代表」大盘，除非用户点名个股。优先指数、行业/概念涨跌双榜、涨跌家数、北向、龙虎榜（若已取）。
   小标题按序：
     1. ``## 结论`` — 市场偏强/偏弱/分化 +「不构成买卖指令」
     2. ``## 指数与广度`` — 主要指数涨跌、涨跌家数（有则写）
     3. ``## 板块对照`` — 领涨与领跌行业/概念**同时**引用；写明榜单口径（全量细分 vs 二级展示）
     4. ``## 资金面`` — 北向/板块资金（有则写）
     5. ``## 多空对照`` — 宏观叙事各 ≥2 条，锚定依据
     6. ``## 情景分析`` — 基准/偏多/偏空（政策、流动性、风险偏好）
     7. ``## 数据缺口``
     8. ``## 操作相关表述`` — 仅用户问建议时；可写观察清单，禁止仓位指令
"""

# ---------------------------------------------------------------------------
# 美 — 大盘 / 板块
# ---------------------------------------------------------------------------

US_MACRO = """\
   **US_MACRO｜US market / sector brief**
   Prefer index quotes (S&P, Nasdaq, Dow), sector/theme relative performance, breadth if available.
   Do not substitute a few megacaps for “the market” unless user asked for them.
   Sections:
     1. ``## 结论``
     2. ``## Indices and session`` — RTH vs closed; major indices %
     3. ``## Sector / factor tape`` — leaders and laggards if fetched
     4. ``## Cross-asset / risk`` — only if tools returned (e.g. VIX proxy ETF); label proxies clearly
     5. ``## 多空对照``
     6. ``## 情景分析`` — rates, liquidity, earnings season, geopolitics as **scenarios**
     7. ``## 数据缺口``
     8. ``## 操作相关表述`` — only if asked
"""

# 拼装进主管规则的完整块（单常量便于测试与 diff）
RESEARCH_BRIEF_TEMPLATE = "\n".join(
    [
        RESEARCH_BRIEF_ROUTER,
        CN_BRIEF,
        US_BRIEF,
        CN_DEEP,
        US_DEEP,
        CN_MACRO,
        US_MACRO,
    ]
)

__all__ = [
    "RESEARCH_BRIEF_ROUTER",
    "CN_BRIEF",
    "US_BRIEF",
    "CN_DEEP",
    "US_DEEP",
    "CN_MACRO",
    "US_MACRO",
    "RESEARCH_BRIEF_TEMPLATE",
]
