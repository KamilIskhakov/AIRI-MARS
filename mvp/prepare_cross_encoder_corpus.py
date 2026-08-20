#!/usr/bin/env python3
"""Merge annotation windows and retain reliable contextual training pairs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def load_windows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("window_*_offset_*/train_consensus_clean.jsonl")):
        # This directory is an old accidental partial duplicate of window 000.
        if path.parent.name == "window_000_offset_000500":
            continue
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def input_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        norm(row.get("left")),
        norm(row.get("right")),
        norm(row.get("entity")),
        norm(row.get("candidate")),
    )


def judge_count(row: dict[str, Any]) -> int:
    rationale = str(row.get("rationale") or "")
    return len([part for part in rationale.split(";") if "=" in part])


def expectation_matches(row: dict[str, Any]) -> bool:
    kind = row.get("pair_kind")
    label = int(row["label"])
    if kind == "proper_agent_alias":
        return label == 1
    if kind == "proper_agent_hard_negative":
        return label == 0
    return True


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()

    raw = load_windows(args.input_root)
    by_input: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        by_input[input_key(row)].append(row)

    kept: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    stats: Counter[str] = Counter(raw_rows=len(raw), unique_model_inputs=len(by_input))

    for items in by_input.values():
        labels = {int(row["label"]) for row in items}
        if len(labels) != 1:
            stats["drop_conflicting_labels"] += len(items)
            review.extend({**row, "review_reason": "identical_input_conflicting_labels"} for row in items)
            continue

        row = items[0]
        stats["drop_exact_duplicates"] += len(items) - 1
        if not expectation_matches(row) and judge_count(row) < 2:
            stats["drop_single_judge_generation_conflict"] += 1
            review.append({**row, "review_reason": "single_judge_generation_conflict"})
            continue

        kept.append(row)
        stats[f"label_{row.get('label')}"] += 1
        stats[f"kind_{row.get('pair_kind', 'unknown')}"] += 1
        stats[f"type_{row.get('entity_type', 'unknown')}"] += 1
        stats[f"domain_{row.get('domain', 'unknown')}"] += 1

    stats["output_rows"] = len(kept)
    stats["review_rows"] = len(review)
    stats["unique_entities"] = len({str(row.get("entity_id")) for row in kept})
    stats["unique_mentions"] = len({str(row.get("mention_id")) for row in kept})
    stats["unique_documents"] = len({str(row.get("source_id")) for row in kept if row.get("source_id")})

    write_jsonl(args.output, kept)
    write_jsonl(args.review_output, review)
    args.summary_output.write_text(
        json.dumps(dict(stats), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
