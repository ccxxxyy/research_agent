"""美股报价回答的诚实性检查（来源署名 + 代理标的）。

不依赖某一只 ticker：凡工具返回 ``proxy=true`` / ``source=eastmoney_us`` 的条目，都可用同一套规则验收模型是否误导。
"""

from __future__ import annotations

import re
from typing import Any


def find_us_quote_misstatements(reply: str, tool_items: list[dict[str, Any]]) -> list[str]:
    """检查回复相对工具载荷是否误导；返回问题码列表（空=通过）。

    规则（通用，非 VIXY 特例）：
    1. 工具 ``source`` 全为东财时，文末/来源不得单独写成 Yahoo。
    2. ``proxy=true`` 时不得把代理价说成「指数现货/收盘指数」而不提代理代码。
    3. 跌幅不得出现 ``-+`` 双符号。
    """
    if not reply or not tool_items:
        return []

    issues: list[str] = []
    text = reply.strip()
    text_l = text.lower()

    sources = {str(it.get("source") or "").strip() for it in tool_items if it.get("source")}
    sources.discard("")
    if sources and sources <= {"eastmoney_us"}:
        claims_yahoo = bool(
            re.search(r"数据来源\s*[:：]\s*yahoo|yahoo\s*finance|来源[:：]\s*yahoo", text_l)
        )
        mentions_em = ("eastmoney" in text_l) or ("东财" in text)
        if claims_yahoo and not mentions_em:
            issues.append("source_mislabel_yahoo")

    for it in tool_items:
        if not it.get("proxy"):
            continue
        quoted = str(it.get("quoted_instrument") or "").strip().upper()
        proxy_of = str(it.get("proxy_of") or it.get("symbol") or "").strip().upper()
        name = str(it.get("name") or "")
        # 回复提到了被代理的指数语义，却未出现代理代码 / 「代理」字样
        index_hints = {
            "^VIX": (r"VIX\s*恐慌|VIX\s*指数|恐慌指数", "VIXY"),
            "^RUT": (r"罗素\s*2000(?!\s*ETF)|Russell\s*2000(?!\s*ETF)", "IWM"),
        }
        hint = index_hints.get(proxy_of)
        if hint:
            pat, default_quoted = hint
            qcode = quoted or default_quoted
            if (
                re.search(pat, text, flags=re.IGNORECASE)
                and qcode.upper() not in text.upper()
                and "代理" not in text
            ):
                issues.append(f"proxy_presented_as_index:{proxy_of}->{qcode}")
        elif (
            quoted
            and re.search(rf"{re.escape(proxy_of.lstrip('^'))}\s*指数", text, re.I)
            and quoted.upper() not in text.upper()
            and "代理" not in text
        ):
            issues.append(f"proxy_presented_as_index:{proxy_of}->{quoted}")

        # 价格被写成「指数收盘」且等于代理价时额外告警（弱规则）
        price = it.get("price")
        if (
            price is not None
            and name
            and "ETF" in name
            and re.search(rf"指数[^。\n]{{0,24}}{re.escape(str(price))}", text)
            and quoted.upper() not in text.upper()
            and "代理" not in text
        ):
            issues.append(f"proxy_price_as_index:{proxy_of}")

    if re.search(r"-\+\d", text):
        issues.append("signed_percent_typo")

    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for x in issues:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


__all__ = ["find_us_quote_misstatements"]
