#!/usr/bin/env python3
"""Merge candidate JSONL files while removing duplicate model inputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    seen: set[tuple[str, str, str, str]] = set()
    rows = []
    for path in args.input:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    norm(row.get("original_context") or row.get("left")),
                    norm(row.get("candidate_context") or row.get("right")),
                    norm(row.get("entity")),
                    norm(row.get("candidate")),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            row["pair_id"] = f"merged{idx:09d}"
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"inputs": [str(path) for path in args.input], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
