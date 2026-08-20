#!/usr/bin/env python3
"""Combine common-noun consensus files and apply documented audit overrides."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_OVERRIDES = {
    ("eons", "moons"): (
        "changed",
        "The candidate produces the non-idiomatic phrase 'in moons'; semantic relatedness is insufficient.",
    ),
    ("hound", "hound dog"): (
        "changed",
        "The context uses 'Mr hound' as a person's name, so expanding it to the animal term changes reference.",
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

P
def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_item(row: dict[str, Any]) -> dict[str, Any] | None:
    label_name = row.get("final_label")
    if label_name not in {"preserved", "changed"}:
        return None
    return {
        "left": row["original_context"],
        "right": row["candidate_context"],
        "label": 1 if label_name == "preserved" else 0,
        "score": row.get("consensus_score"),
        "entity": row.get("entity"),
        "candidate": row.get("candidate"),
        "entity_id": row.get("entity_id"),
        "entity_type": row.get("fine_type"),
        "coarse_group": row.get("coarse_group"),
        "context_policy": row.get("context_policy"),
        "pair_kind": row.get("candidate_kind"),
        "branch": row.get("branch"),
        "dataset": row.get("dataset"),
        "text_id": row.get("text_id"),
        "mention_id": row.get("mention_id"),
        "cosine": row.get("cosine"),
        "shared_wordnet_synset": row.get("shared_wordnet_synset"),
        "annotation_source": "common_embedding_two_judges_audited",
        "rationale": row.get("final_rationale"),
        "pair_id": row.get("pair_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus", type=Path, nargs="+", required=True)
    parser.add_argument("--out-final", type=Path, required=True)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.consensus:
        rows.extend(load_jsonl(path))

    overridden = 0
    for row in rows:
        row["final_label"] = row.get("consensus_label")
        row["final_rationale"] = row.get("consensus_rationale")
        override = AUDIT_OVERRIDES.get((str(row.get("entity")), str(row.get("candidate"))))
        if override:
            row["final_label"], row["final_rationale"] = override
            row["manual_audit_override"] = True
            overridden += 1
        else:
            row["manual_audit_override"] = False

    train_rows = [item for row in rows if (item := train_item(row)) is not None]
    write_jsonl(args.out_final, rows)
    write_jsonl(args.out_train, train_rows)
    summary = {
        "total_pairs": len(rows),
        "manual_overrides": overridden,
        "final_labels": dict(Counter(str(row.get("final_label")) for row in rows)),
        "train_pairs": len(train_rows),
        "train_labels": dict(Counter(str(row["label"]) for row in train_rows)),
        "identity_pairs": sum(
            str(row.get("entity", "")).casefold() == str(row.get("candidate", "")).casefold()
            for row in rows
        ),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
