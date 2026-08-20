#!/usr/bin/env python3
"""Create a stratified blind re-judging sample from train-ready annotations."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=240)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(args.input)
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = int(row["label"])
        buckets[(str(row.get("corpus_source")), str(row.get("entity_type")), label)].append(row)

    selected: list[dict[str, Any]] = []
    active = {key for key, values in buckets.items() if values}
    while active and len(selected) < args.sample_size:
        for key in sorted(active):
            if len(selected) >= args.sample_size:
                break
            values = buckets[key]
            if values:
                selected.append(values.pop(rng.randrange(len(values))))
        active = {key for key in active if buckets[key]}
    rng.shuffle(selected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(selected, start=1):
            item: dict[str, Any] = dict(row)
            item["pair_id"] = f"audit{idx:06d}"
            item["original_pair_id"] = row.get("pair_id")
            item["expected_score"] = float(int(row["label"]))
            item["original_context"] = row["left"]
            item["candidate_context"] = row["right"]
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({
        "source_rows": len(rows),
        "sample_rows": len(selected),
        "labels": dict(Counter(int(row["label"]) for row in selected)),
        "sources": dict(Counter(str(row.get("corpus_source")) for row in selected)),
        "types": dict(Counter(str(row.get("entity_type")) for row in selected)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
