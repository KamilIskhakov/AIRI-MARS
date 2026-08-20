#!/usr/bin/env python3
"""Measure blind re-judge agreement with existing substitution labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_label(row: dict[str, Any]) -> str:
    value = row.get("label")
    if value in {1, "1"}:
        return "preserved"
    if value in {0, "0"}:
        return "changed"
    score = row.get("expected_score")
    return "preserved" if score == 1 else "changed" if score == 0 else "unknown"


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = Counter((expected_label(row), str(row.get("judge_label", "missing"))) for row in rows)
    usable = [row for row in rows if row.get("judge_label") in {"preserved", "changed"}]
    correct = sum(expected_label(row) == row["judge_label"] for row in usable)
    expected_counts = Counter(expected_label(row) for row in usable)
    judged_counts = Counter(str(row["judge_label"]) for row in usable)
    n = len(usable)
    observed = correct / n if n else 0.0
    chance = sum(expected_counts[label] * judged_counts[label] for label in ("preserved", "changed")) / max(n * n, 1)
    kappa = (observed - chance) / (1 - chance) if chance < 1 else 0.0
    return {
        "rows": len(rows),
        "usable_binary": n,
        "agreement": round(observed, 4),
        "cohen_kappa": round(kappa, 4),
        "uncertain_or_missing": len(rows) - n,
        "confusion": {f"expected={a}|judged={b}": count for (a, b), count in sorted(confusion.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review-out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.judged)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[f"source:{row.get('corpus_source', 'unknown')}"] .append(row)
        buckets[f"type:{row.get('entity_type', row.get('fine_type', 'unknown'))}"] .append(row)
        buckets[f"label:{expected_label(row)}"].append(row)
    report = {
        "overall": metrics(rows),
        "by_bucket": {name: metrics(values) for name, values in sorted(buckets.items())},
    }
    review = [
        row for row in rows
        if row.get("judge_label") not in {expected_label(row), None}
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.review_out.open("w", encoding="utf-8") as out:
        for row in review:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
