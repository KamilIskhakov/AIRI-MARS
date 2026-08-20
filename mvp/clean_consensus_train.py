#!/usr/bin/env python3
"""Clean consensus train pairs while preserving annotation metadata."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def norm_literal(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", value).strip()


def norm_alnum(text: str) -> str:
    return re.sub(r"[^\w]+", "", norm_literal(text))


def norm_article(text: str) -> str:
    value = norm_literal(text).replace("&", "and")
    value = re.sub(r"^(the|a|an)\s+", "", value)
    value = re.sub(r"('s|’s)$", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def is_identity(entity: str, candidate: str) -> bool:
    return (
        norm_literal(entity) == norm_literal(candidate)
        or norm_alnum(entity) == norm_alnum(candidate)
        or norm_article(entity) == norm_article(candidate)
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    kept: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    summary: Counter[str] = Counter(input=len(rows))

    for row in rows:
        entity = str(row.get("entity", ""))
        candidate = str(row.get("candidate", ""))
        if is_identity(entity, candidate):
            summary["drop_identity"] += 1
            continue
        key = (
            row.get("label"),
            row.get("entity_id"),
            norm_literal(entity),
            norm_literal(candidate),
            row.get("left"),
            row.get("right"),
        )
        if key in seen:
            summary["drop_duplicate"] += 1
            continue
        seen.add(key)
        kept.append(row)
        summary[f"label_{row.get('label')}"] += 1
        summary[f"kind_{row.get('pair_kind', 'unknown')}"] += 1
        summary[f"type_{row.get('entity_type', 'unknown')}"] += 1

    summary["output"] = len(kept)
    write_jsonl(args.out, kept)
    args.summary_out.write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(dict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
