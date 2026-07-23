"""市场与标的的一等公民类型（P0 契约）。

美股拓展的边界约定：
* A 股（``CN_A``）与美股（``US``）平行隔离，禁止混用同一套 MCP 工具。
* PoC 数据源：yfinance；一期范围：美股股票 + 指数 + ETF（不含共同基金/期权）。
* 默认市场：用户 memory 偏好；问句中的名字 / ticker / 6 位代码优先于偏好。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Market(StrEnum):
    """交易市场。"""

    CN_A = "CN_A"
    """中国 A 股（沪深京）。"""

    US = "US"
    """美国市场（NYSE / NASDAQ 等）。"""

    MIXED = "MIXED"
    """同一问句明确要求跨市场对比。"""

    UNKNOWN = "UNKNOWN"
    """信号不足；应回退到用户偏好，再不行由产品默认处理。"""


class AssetClass(StrEnum):
    """一期支持的资产类别。"""

    EQUITY = "equity"
    """普通股。"""

    INDEX = "index"
    """指数。"""

    ETF = "etf"
    """交易所交易基金。"""

    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SymbolRef:
    """规范化标的引用。

    ``raw`` 保留用户原文；
    ``ticker`` 为可交给下游工具的代码（A 股 6 位数字，美股大写 ticker）。
    P0 阶段 ticker 可能为空（仅识别到市场 / 中文名，尚未解析到代码）。
    """

    market: Market
    raw: str
    ticker: str = ""
    asset_class: AssetClass = AssetClass.UNKNOWN
    exchange: str = ""
    display_name: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market.value,
            "raw": self.raw,
            "ticker": self.ticker,
            "asset_class": self.asset_class.value,
            "exchange": self.exchange,
            "display_name": self.display_name,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class MarketResolution:
    """一次用户问句的市场判定结果。"""

    market: Market
    source: str
    """判定来源：``query_signal`` / ``user_preference`` / ``default`` / ``request_override``。"""

    confidence: float
    symbols: tuple[SymbolRef, ...] = ()
    reasons: tuple[str, ...] = ()
    preferred_market: Market | None = None
    """用户偏好（若有），便于日志与前端展示。"""

    notes: str = ""
    """给人看的补充说明（例如「美股工具尚未上线」）。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market.value,
            "source": self.source,
            "confidence": self.confidence,
            "symbols": [s.to_dict() for s in self.symbols],
            "reasons": list(self.reasons),
            "preferred_market": (self.preferred_market.value if self.preferred_market else None),
            "notes": self.notes,
        }


# 用户偏好在 MemoryNamespace.USER_PREFERENCES 下的固定 key
PREFERRED_MARKET_KEY = "preferred_market"

# 产品默认：无偏好且问句无信号时，仍以 A 股为主（当前工具全集）
PRODUCT_DEFAULT_MARKET = Market.CN_A

__all__ = [
    "PREFERRED_MARKET_KEY",
    "PRODUCT_DEFAULT_MARKET",
    "AssetClass",
    "Market",
    "MarketResolution",
    "SymbolRef",
]
