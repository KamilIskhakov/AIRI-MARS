#!/usr/bin/env python3
"""Summarize entity-group coverage across inventory, annotation, train and eval."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(paths: list[Path], unique_entities: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if unique_entities:
                    entity_id = str(row.get("entity_id", ""))
                    if not entity_id or entity_id in seen:
                        continue
                    seen.add(entity_id)
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], unit: str) -> dict[str, Any]:
    coarse = Counter(str(row.get("coarse_group") or "missing") for row in rows)
    fine = Counter(str(row.get("fine_type", row.get("entity_type")) or "missing") for row in rows)
    nested: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group = str(row.get("coarse_group") or "missing")
        entity_type = str(row.get("fine_type", row.get("entity_type")) or "missing")
        nested[group][entity_type] += 1
    return {
        "unit": unit,
        "total": len(rows),
        "coarse": dict(coarse),
        "fine": dict(fine),
        "fine_by_coarse": {group: dict(counts) for group, counts in nested.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-snapshot-dir", required=True, type=Path)
    parser.add_argument("--pilot-pairs", required=True, type=Path)
    parser.add_argument("--train-pairs", required=True, type=Path)
    parser.add_argument("--comparison-pairs", required=True, type=Path)
    parser.add_argument("--comparison-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory-target", type=int, default=50000)
    args = parser.parse_args()

    inventory = load_jsonl(sorted(args.tag_snapshot_dir.glob("*.jsonl")), unique_entities=True)
    pilot = load_jsonl([args.pilot_pairs])
    train = load_jsonl([args.train_pairs])
    comparison = load_jsonl([args.comparison_pairs])
    comparison_summary = json.loads(args.comparison_summary.read_text(encoding="utf-8"))

    result = {
        "inventory_target": args.inventory_target,
        "inventory_tagged": len(inventory),
        "inventory_tagged_ratio": round(len(inventory) / max(args.inventory_target, 1), 6),
        "stages": {
            "inventory": summarize(inventory, "unique_entities"),
            "pilot_annotation": summarize(pilot, "pairs"),
            "reliable_train": summarize(train, "pairs"),
            "fair_comparison": summarize(comparison, "pairs"),
        },
        "performance_by_type": {
            entity_type: {
                "count": metrics["count"],
                "old_macro_f1": comparison_summary["old_by_type"][entity_type]["macro_f1"],
                "new_macro_f1": metrics["macro_f1"],
            }
            for entity_type, metrics in comparison_summary["new_by_type"].items()
            if entity_type in comparison_summary["old_by_type"]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
