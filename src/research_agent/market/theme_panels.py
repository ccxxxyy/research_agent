"""A 股看板主题面板：主线题材 / 情绪标杆 / 妖股·庄股规则近似。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def as_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        f = float(val)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def build_mainline_themes(concepts: list[dict], zt_pool: list[dict]) -> list[dict]:
    """主线题材：概念涨幅 + 匹配涨停家数合成得分。"""
    ind_zt: dict[str, list[dict]] = defaultdict(list)
    for s in zt_pool or []:
        ind = str(s.get("industry") or "").strip()
        if ind:
            ind_zt[ind].append(s)

    themes: list[dict] = []
    for c in concepts or []:
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        change = as_float(c.get("change_pct")) or 0.0
        matched: list[dict] = []
        for ind, stocks in ind_zt.items():
            if len(ind) < 2:
                continue
            if ind in name or name in ind:
                matched.extend(stocks)
        seen_codes: set[str] = set()
        uniq: list[dict] = []
        for s in matched:
            code = str(s.get("code") or "")
            if code and code not in seen_codes:
                seen_codes.add(code)
                uniq.append(s)
        zt_count = len(uniq)
        max_streak = 0
        for s in uniq:
            st = as_float(s.get("streak"))
            if st is not None:
                max_streak = max(max_streak, int(st))
        up_count = as_float(c.get("up_count"))
        score = change + zt_count * 1.8 + max_streak * 0.4
        if up_count is not None:
            score += min(up_count, 80) * 0.02
        leaders = [str(s.get("name") or "") for s in uniq[:2] if s.get("name")]
        themes.append(
            {
                "code": str(c.get("code") or ""),
                "name": name,
                "change_pct": change,
                "zt_count": zt_count,
                "max_streak": max_streak,
                "up_count": int(up_count) if up_count is not None else None,
                "score": round(score, 2),
                "leaders": leaders,
            }
        )
    themes.sort(key=lambda x: (-(x.get("score") or 0), -(x.get("zt_count") or 0)))
    return themes[:10]


def build_sentiment_benchmark(zt_pool: list[dict]) -> list[dict]:
    """情绪标杆：连板高度优先，其次封单（封板资金）。"""
    items: list[dict] = []
    for s in zt_pool or []:
        streak = as_float(s.get("streak"))
        seal = as_float(s.get("seal_amount"))
        st = int(streak) if streak is not None else 0
        items.append(
            {
                "code": str(s.get("code") or ""),
                "name": str(s.get("name") or ""),
                "industry": str(s.get("industry") or ""),
                "streak": st,
                "seal_amount": seal,
                "open_count": s.get("open_count"),
                "first_time": s.get("first_time"),
                "tag": f"{st}连板" if st > 1 else "首板",
            }
        )
    items.sort(key=lambda x: (-(x.get("streak") or 0), -(x.get("seal_amount") or 0)))
    return items[:10]


def build_speculative_pool(zt_pool: list[dict], lhb: list[dict], changes: list[dict]) -> list[dict]:
    """妖股 / 庄股规则近似（非官方标签，仅供留意比对）。"""
    out: list[dict] = []
    seen: set[str] = set()

    def _add(item: dict) -> None:
        code = str(item.get("code") or "")
        if not code or code in seen:
            return
        seen.add(code)
        out.append(item)

    for s in zt_pool or []:
        st = int(as_float(s.get("streak")) or 0)
        open_n = int(as_float(s.get("open_count")) or 0)
        if st >= 3:
            _add(
                {
                    "code": str(s.get("code") or ""),
                    "name": str(s.get("name") or ""),
                    "industry": str(s.get("industry") or ""),
                    "label": "妖股候选",
                    "reason": f"{st}连板高标",
                    "streak": st,
                    "change_pct": s.get("change_pct"),
                }
            )
        elif st >= 2 and open_n >= 1:
            _add(
                {
                    "code": str(s.get("code") or ""),
                    "name": str(s.get("name") or ""),
                    "industry": str(s.get("industry") or ""),
                    "label": "妖股候选",
                    "reason": f"{st}连板·炸板{open_n}次（分歧）",
                    "streak": st,
                    "change_pct": s.get("change_pct"),
                }
            )

    for s in lhb or []:
        comment = str(s.get("comment") or "")
        net = as_float(s.get("net_buy")) or 0.0
        if "机构" in comment and net > 0:
            _add(
                {
                    "code": str(s.get("code") or ""),
                    "name": str(s.get("name") or ""),
                    "industry": str(s.get("industry") or ""),
                    "label": "庄股/机构候选",
                    "reason": comment[:40] or "龙虎榜机构净买",
                    "change_pct": s.get("change_pct"),
                    "net_buy": net,
                }
            )
        elif net >= 50_000_000:  # ≥5000 万
            _add(
                {
                    "code": str(s.get("code") or ""),
                    "name": str(s.get("name") or ""),
                    "industry": str(s.get("industry") or ""),
                    "label": "游资强买",
                    "reason": f"龙虎榜净买约 {net / 10000:.0f} 万",
                    "change_pct": s.get("change_pct"),
                    "net_buy": net,
                }
            )

    for s in changes or []:
        if str(s.get("type") or "") == "急速拉升" and str(s.get("code") or "") not in seen:
            _add(
                {
                    "code": str(s.get("code") or ""),
                    "name": str(s.get("name") or ""),
                    "industry": str(s.get("industry") or ""),
                    "label": "情绪发散",
                    "reason": "盘中急速拉升",
                    "change_pct": None,
                }
            )
        if len(out) >= 12:
            break
    return out[:10]
