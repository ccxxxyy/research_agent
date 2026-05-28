"""对比两次评估报告的差异，用于 prompt/模型变更前后的 regression 检测。

用法::

    python -m evals.compare eval_results/report_baseline.json eval_results/report_current.json

    # 设置回归阈值（默认 0.05，即某指标下降超过 5% 就报红）
    python -m evals.compare baseline.json current.json --threshold 0.03

退出码:
  0 — 无回归
  1 — 检测到回归（至少一个指标下降超过阈值）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(baseline: dict, current: dict, threshold: float) -> bool:
    """Compare two reports. Returns True if regression detected."""
    agg_b = baseline["aggregate"]
    agg_c = current["aggregate"]
    meta_b = baseline["metadata"]
    meta_c = current["metadata"]

    print("\n" + "=" * 78)
    print("  EVALUATION COMPARISON")
    print(f"  Baseline : {meta_b['timestamp'][:19]} ({meta_b['num_examples']} examples)")
    print(f"  Current  : {meta_c['timestamp'][:19]} ({meta_c['num_examples']} examples)")
    print(f"  Threshold: {threshold:.1%} (regression if delta < -{threshold:.1%})")
    print("=" * 78)

    all_keys = sorted(set(agg_b) | set(agg_c))
    regressions: list[str] = []

    print(f"\n{'Metric':<28} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Status':>10}")
    print("-" * 70)

    for key in all_keys:
        mean_b = agg_b.get(key, {}).get("mean", 0.0)
        mean_c = agg_c.get(key, {}).get("mean", 0.0)
        delta = mean_c - mean_b

        if delta < -threshold:
            status = "REGRESS"
            regressions.append(key)
        elif delta > threshold:
            status = "IMPROVE"
        else:
            status = "OK"

        sign = "+" if delta >= 0 else ""
        print(f"  {key:<26} {mean_b:>9.3f} {mean_c:>9.3f} {sign}{delta:>8.3f}   {status}")

    cat_b = baseline.get("by_category", {})
    cat_c = current.get("by_category", {})
    all_cats = sorted(set(cat_b) | set(cat_c))

    if all_cats:
        print(f"\n{'Category':<20} {'Metric':<26} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
        print("-" * 78)
        for cat in all_cats:
            cat_keys = sorted(set(cat_b.get(cat, {})) | set(cat_c.get(cat, {})))
            for key in cat_keys:
                mean_b = cat_b.get(cat, {}).get(key, {}).get("mean", 0.0)
                mean_c = cat_c.get(cat, {}).get(key, {}).get("mean", 0.0)
                delta = mean_c - mean_b
                sign = "+" if delta >= 0 else ""
                print(f"  {cat:<18} {key:<24} {mean_b:>9.3f} {mean_c:>9.3f} {sign}{delta:>8.3f}")

    print("=" * 78)

    if regressions:
        print(f"\n  REGRESSION DETECTED in: {', '.join(regressions)}")
        print(f"  {len(regressions)} metric(s) declined by more than {threshold:.1%}\n")
        return True
    else:
        print("\n  No regressions detected.\n")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation reports.")
    parser.add_argument("baseline", type=Path, help="Path to baseline report JSON")
    parser.add_argument("current", type=Path, help="Path to current report JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Regression threshold (default: 0.05 = 5%%)",
    )
    args = parser.parse_args()

    baseline = _load_report(args.baseline)
    current = _load_report(args.current)
    has_regression = compare(baseline, current, args.threshold)

    sys.exit(1 if has_regression else 0)


if __name__ == "__main__":
    main()
