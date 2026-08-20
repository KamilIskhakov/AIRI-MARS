#!/usr/bin/env python3
"""Finalize Mistral substitution judgments into train-ready JSONL and a report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]], train_rows: list[dict[str, Any]], skipped: Counter[str]) -> dict[str, Any]:
    labels = Counter(row.get("judge_label", "missing") for row in rows)
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    expected_total: Counter[str] = Counter()
    expected_ok: Counter[str] = Counter()
    for row in rows:
        kind = row.get("candidate_kind", "unknown")
        label = row.get("judge_label", "missing")
        by_kind[kind][label] += 1
        expected = row.get("expected_score")
        if expected is None:
            continue
        expected_label = "preserved" if float(expected) >= 0.9 else "changed" if float(expected) <= 0.1 else "uncertain"
        expected_total[kind] += 1
        if label == expected_label:
            expected_ok[kind] += 1
    return {
        "total_pairs": len(rows),
        "train_pairs": len(train_rows),
        "skipped_labels": dict(skipped),
        "labels": dict(labels),
        "by_candidate_kind": {kind: dict(counts) for kind, counts in by_kind.items()},
        "expected_agreement": {
            kind: {
                "ok": expected_ok[kind],
                "total": expected_total[kind],
                "rate": round(expected_ok[kind] / max(expected_total[kind], 1), 3),
            }
            for kind in sorted(expected_total)
        },
    }


def train_item(row: dict[str, Any]) -> dict[str, Any] | None:
    label = row.get("judge_label")
    if label == "preserved":
        binary = 1
    elif label == "changed":
        binary = 0
    else:
        return None
    return {
        "left": row["original_context"],
        "right": row["candidate_context"],
        "label": binary,
        "score": row.get("judge_score"),
        "entity": row.get("entity"),
        "candidate": row.get("candidate"),
        "entity_id": row.get("entity_id"),
        "entity_type": row.get("fine_type"),
        "coarse_group": row.get("coarse_group"),
        "context_policy": row.get("context_policy"),
        "pair_kind": row.get("candidate_kind"),
        "branch": row.get("branch"),
        "annotation_source": "mistral_context_judge",
        "rationale": row.get("judge_rationale"),
        "pair_id": row.get("pair_id"),
    }


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Mistral Substitution Annotation",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Files",
        "",
        "- `probe_pairs.jsonl`: generated candidates.",
        "- `judge_mistral.jsonl`: raw judge output.",
        "- `judge_mistral_repair.jsonl`: repaired missing pair_ids, if any.",
        "- `judge_mistral_final.jsonl`: merged final judgments.",
        "- `train_pairs_mistral_judged.jsonl`: binary train-ready pairs.",
        "",
        "## Examples",
    ]
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[row.get("candidate_kind", "unknown")].append(row)
    for kind in sorted(by_kind):
        lines.extend(["", f"### {kind}", ""])
        for row in by_kind[kind][:8]:
            lines.append(
                f"- `{row.get('entity')}` -> `{row.get('candidate')}` | "
                f"judge={row.get('judge_label')} score={row.get('judge_score')}"
            )
            lines.append(f"  - {row.get('judge_rationale', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--out-final", required=True, type=Path)
    parser.add_argument("--out-train", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    args = parser.parse_args()

    raw_rows = load_jsonl(args.raw)
    repair_rows = load_jsonl(args.repair) if args.repair else []
    repairs = {
        row["pair_id"]: row
        for row in repair_rows if row.get("pair_id")
    }
    final_rows = [repairs.get(row.get("pair_id"), row) for row in raw_rows]
    write_jsonl(args.out_final, final_rows)

    train_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in final_rows:
        item = train_item(row)
        if item is None:
            skipped[str(row.get("judge_label", "missing"))] += 1
        else:
            train_rows.append(item)
    write_jsonl(args.out_train, train_rows)

    summary = summarize(final_rows, train_rows, skipped)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.report_out, summary, final_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
