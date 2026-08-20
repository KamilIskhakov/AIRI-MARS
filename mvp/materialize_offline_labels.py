#!/usr/bin/env python3
"""Turn local weak/rule outputs into explicitly marked training rows.

This file never calls a remote model. Labels are copied only from local
``weak_label`` or deterministic ``expected_score`` fields and are marked as
non-gold supervision in the metadata.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path)


def local_label(row: dict[str, Any]) -> int | None:
    value = row.get("label", row.get("weak_label"))
    if value in {0, 1, "0", "1"}:
        return int(value)
    expected = row.get("expected_score")
    if expected in {0, 1, 0.0, 1.0, "0", "1", "0.0", "1.0"}:
        return int(float(expected))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="NAME=PATH; repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    output: list[dict[str, Any]] = []
    skipped = Counter()
    sources = []
    for raw_input in args.input:
        source_name, path = parse_input(raw_input)
        sources.append({"name": source_name, "path": str(path)})
        for row in load(path):
            label = local_label(row)
            if label is None:
                skipped[f"{source_name}:missing_local_label"] += 1
                continue
            normalized = dict(row)
            normalized["label"] = label
            normalized["corpus_source"] = f"offline_{source_name}"
            normalized["annotation_source"] = normalized.get(
                "annotation_source", f"offline_{source_name}_weak_or_rule"
            )
            normalized["supervision_quality"] = "weak_or_rule_based"
            normalized["remote_judgment_used"] = False
            output.append(normalized)

    summary = {
        "inputs": sources,
        "rows": len(output),
        "labels": dict(Counter(str(row["label"]) for row in output)),
        "by_source": dict(Counter(str(row["corpus_source"]) for row in output)),
        "by_candidate_kind": dict(Counter(str(row.get("candidate_kind", "unknown")) for row in output)),
        "skipped": dict(skipped),
        "remote_judgment_used": False,
        "warning": "These labels are local weak/rule supervision, not gold annotations.",
    }
    write(args.out, output)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
