#!/usr/bin/env python3
"""Build leakage-safe positive-vs-negative ranking examples per entity mention."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--group-key", default="mention_id")
    parser.add_argument("--max-negatives-per-positive", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_group = 0
    for row in rows:
        value = row.get(args.group_key)
        if value in (None, ""):
            missing_group += 1
            continue
        groups[str(value)].append(row)

    rng = random.Random(args.seed)
    output: list[dict[str, Any]] = []
    groups_with_both = 0
    by_source: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    for group_id, items in sorted(groups.items()):
        positives = [row for row in items if int(row["label"]) == 1]
        negatives = [row for row in items if int(row["label"]) == 0]
        if not positives or not negatives:
            continue
        groups_with_both += 1
        for positive in positives:
            selected = list(negatives)
            rng.shuffle(selected)
            selected = selected[: args.max_negatives_per_positive]
            for negative in selected:
                entity_type = str(positive.get("entity_type") or negative.get("entity_type") or "UNKNOWN")
                source = str(positive.get("corpus_source") or negative.get("corpus_source") or "unknown")
                output.append(
                    {
                        "ranking_id": stable_id(group_id, positive.get("pair_id"), negative.get("pair_id")),
                        "group_id": group_id,
                        "group_key": args.group_key,
                        "positive_pair_id": positive.get("pair_id"),
                        "negative_pair_id": negative.get("pair_id"),
                        "positive": positive,
                        "negative": negative,
                        "entity_type": entity_type,
                        "corpus_source": source,
                    }
                )
                by_source[source] += 1
                by_type[entity_type] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for item in output:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "rows": len(rows),
        "group_key": args.group_key,
        "groups": len(groups),
        "groups_with_both_labels": groups_with_both,
        "ranking_pairs": len(output),
        "missing_group": missing_group,
        "max_negatives_per_positive": args.max_negatives_per_positive,
        "by_corpus_source": dict(sorted(by_source.items())),
        "by_entity_type": dict(sorted(by_type.items())),
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
