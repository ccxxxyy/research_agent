"""评估数据集加载与合并。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATASETS_DIR = Path(__file__).parent / "datasets"
CN_ROUTING_PATH = _DATASETS_DIR / "supervisor_routing.json"
US_ROUTING_PATH = _DATASETS_DIR / "us_market_routing.json"


def load_json_dataset(path: Path) -> list[dict[str, Any]]:
    """加载单个 JSON 数组数据集。"""
    return json.loads(path.read_text(encoding="utf-8"))


def load_merged_routing_dataset(
    *,
    include_cn: bool = True,
    include_us: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """合并 A 股路由集与美股路由集。

    默认两者都加载；``limit`` 截断合并后的列表（先 CN 后 US）。
    """
    examples: list[dict[str, Any]] = []
    if include_cn and CN_ROUTING_PATH.is_file():
        examples.extend(load_json_dataset(CN_ROUTING_PATH))
    if include_us and US_ROUTING_PATH.is_file():
        examples.extend(load_json_dataset(US_ROUTING_PATH))
    if limit is not None:
        examples = examples[:limit]
    return examples
