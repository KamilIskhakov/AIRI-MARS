#!/usr/bin/env python3
"""Build a tiny real-data fixture that exercises every multi-task branch."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        norm(row.get("left") or row.get("original_context")),
        norm(row.get("right") or row.get("candidate_context")),
        norm(row.get("entity")),
        norm(row.get("candidate")),
    )


def valid_label(row: dict[str, Any]) -> int | None:
    value = row.get("label")
    return int(value) if value in {0, 1, "0", "1"} else None


def select_split(path: Path, directional_keys: set[tuple[str, str, str, str]], count: int) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    directional = [row for row in rows if row_key(row) in directional_keys and valid_label(row) is not None]
    positives = [row for row in rows if valid_label(row) == 1]
    negatives = [row for row in rows if valid_label(row) == 0]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(row: dict[str, Any]) -> None:
        key = row_key(row)
        if key not in seen and len(selected) < count:
            selected.append(row)
            seen.add(key)

    if directional:
        add(directional[0])
    for bucket in (positives, negatives):
        for row in bucket:
            add(row)
            if sum(valid_label(item) == valid_label(row) for item in selected) >= max(2, count // 2):
                break
    for row in rows:
        add(row)
        if len(selected) == count:
            break

    labels = Counter(valid_label(row) for row in selected)
    directional_count = sum(row_key(row) in directional_keys for row in selected)
    if len(selected) != count or not labels[0] or not labels[1]:
        raise RuntimeError(f"Cannot build balanced {path.name} fixture: rows={len(selected)} labels={dict(labels)}")
    if not directional_count:
        raise RuntimeError(f"No exact directional row found in {path}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--directional-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows-per-split", type=int, default=8)
    parser.add_argument("--ranking-pairs", type=int, default=1)
    args = parser.parse_args()

    directional_rows = list(read_jsonl(args.directional_jsonl))
    directional_keys = {
        row_key(row)
        for row in directional_rows
        if float(row.get("directional_confidence", 0.0) or 0.0) >= 0.8
        and row.get("directional_a_to_b") in {"entailment", "neutral", "contradiction"}
        and row.get("directional_b_to_a") in {"entailment", "neutral", "contradiction"}
    }
    if not directional_keys:
        raise RuntimeError("Directional file has no valid high-confidence rows")

    summary: dict[str, Any] = {"rows_per_split": args.rows_per_split, "splits": {}}
    for split in ("train", "val", "test"):
        selected = select_split(args.corpus_dir / f"{split}.jsonl", directional_keys, args.rows_per_split)
        write_jsonl(args.output_dir / f"{split}.jsonl", selected)
        summary["splits"][split] = {
            "rows": len(selected),
            "labels": dict(Counter(str(valid_label(row)) for row in selected)),
            "directional_exact_matches": sum(row_key(row) in directional_keys for row in selected),
        }

    ranking = []
    for row in read_jsonl(args.corpus_dir / "ranking_train.jsonl"):
        positive = row.get("positive", {})
        negative = row.get("negative", {})
        if valid_label(positive) == 1 and valid_label(negative) == 0:
            ranking.append(row)
        if len(ranking) == args.ranking_pairs:
            break
    if len(ranking) != args.ranking_pairs:
        raise RuntimeError("Not enough valid ranking pairs")
    write_jsonl(args.output_dir / "ranking_train.jsonl", ranking)
    summary["ranking_pairs"] = len(ranking)
    summary["directional_source_rows"] = len(directional_rows)
    summary["directional_valid_keys"] = len(directional_keys)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
