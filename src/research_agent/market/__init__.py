"""市场维度：枚举、标的引用、问句/偏好解析（P0）。"""

from research_agent.market.detect import (
    detect_market_from_query,
    extract_symbols_from_query,
    format_market_preamble,
    get_user_preferred_market,
    parse_preferred_market,
    resolve_market,
    set_user_preferred_market,
)
from research_agent.market.types import (
    PREFERRED_MARKET_KEY,
    PRODUCT_DEFAULT_MARKET,
    AssetClass,
    Market,
    MarketResolution,
    SymbolRef,
)

__all__ = [
    "PREFERRED_MARKET_KEY",
    "PRODUCT_DEFAULT_MARKET",
    "AssetClass",
    "Market",
    "MarketResolution",
    "SymbolRef",
    "detect_market_from_query",
    "extract_symbols_from_query",
    "format_market_preamble",
    "get_user_preferred_market",
    "parse_preferred_market",
    "resolve_market",
    "set_user_preferred_market",
]
