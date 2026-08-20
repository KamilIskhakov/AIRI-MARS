#!/usr/bin/env python3
"""Merge multiple substitution-judge outputs with a conservative consensus rule."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_name(path: Path, row: dict[str, Any]) -> str:
    provider = row.get("judge_provider")
    model = row.get("judge_model")
    if provider and model:
        return f"{provider}:{model}"
    if provider:
        return str(provider)
    name = path.stem
    if "mistral" in name:
        return "mistral"
    return name


def consensus_label(votes: list[str], positive_votes: int, negative_votes: int) -> str:
    counts = Counter(v for v in votes if v in {"preserved", "changed", "uncertain"})
    if counts["preserved"] >= positive_votes and counts["changed"] == 0:
        return "preserved"
    if counts["changed"] >= negative_votes and counts["preserved"] == 0:
        return "changed"
    return "uncertain"


def train_item(row: dict[str, Any]) -> dict[str, Any] | None:
    label = row.get("consensus_label")
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
        "score": row.get("consensus_score"),
        "entity": row.get("entity"),
        "candidate": row.get("candidate"),
        "entity_id": row.get("entity_id"),
        "entity_type": row.get("fine_type"),
        "coarse_group": row.get("coarse_group"),
        "context_policy": row.get("context_policy"),
        "pair_kind": row.get("candidate_kind"),
        "branch": row.get("branch"),
        "provider": row.get("provider"),
        "dataset": row.get("dataset"),
        "dataset_name": row.get("dataset_name"),
        "dataset_split": row.get("dataset_split"),
        "dataset_id": row.get("dataset_id"),
        "text_id": row.get("text_id"),
        "mention_id": row.get("mention_id"),
        "row_idx": row.get("row_idx"),
        "source_id": row.get("source_id"),
        "domain": row.get("domain"),
        "mask_idx": row.get("mask_idx"),
        "expected_score": row.get("expected_score"),
        "cosine": row.get("cosine"),
        "retrieval_bucket": row.get("retrieval_bucket"),
        "shared_wordnet_synset": row.get("shared_wordnet_synset"),
        "alias_relation": row.get("alias_relation"),
        "alias_confidence": row.get("alias_confidence"),
        "source_pair_id": row.get("source_pair_id"),
        "source_shard": row.get("source_shard"),
        "annotation_source": "multi_judge_consensus",
        "rationale": row.get("consensus_rationale"),
        "pair_id": row.get("pair_id"),
        "item_id": row.get("item_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", type=Path, nargs="+", required=True)
    parser.add_argument("--out-final", type=Path, required=True)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--positive-votes", type=int, default=2)
    parser.add_argument("--negative-votes", type=int, default=2)
    args = parser.parse_args()

    merged: dict[str, dict[str, Any]] = {}
    vote_rows: dict[str, list[dict[str, Any]]] = {}
    for path in args.judged:
        for row in load_jsonl(path):
            pair_id = str(row.get("pair_id"))
            if not pair_id:
                continue
            base = {key: value for key, value in row.items() if not key.startswith("judge_")}
            merged.setdefault(pair_id, base)
            vote_rows.setdefault(pair_id, [])
            vote_rows[pair_id].append(
                {
                    "source": source_name(path, row),
                    "label": row.get("judge_label", "missing"),
                    "score": row.get("judge_score"),
                    "rationale": row.get("judge_rationale"),
                }
            )

    final_rows: list[dict[str, Any]] = []
    for pair_id in sorted(merged):
        votes = vote_rows.get(pair_id, [])
        labels = [str(v.get("label")) for v in votes]
        label = consensus_label(labels, args.positive_votes, args.negative_votes)
        numeric_scores = [float(v["score"]) for v in votes if isinstance(v.get("score"), int | float)]
        row = dict(merged[pair_id])
        row["judge_votes"] = votes
        row["consensus_label"] = label
        row["consensus_score"] = round(sum(numeric_scores) / len(numeric_scores), 4) if numeric_scores else None
        row["consensus_rationale"] = "; ".join(
            f"{v['source']}={v.get('label')}" for v in votes
        )
        final_rows.append(row)

    train_rows = [item for row in final_rows if (item := train_item(row)) is not None]
    write_jsonl(args.out_final, final_rows)
    write_jsonl(args.out_train, train_rows)

    summary = {
        "total_pairs": len(final_rows),
        "train_pairs": len(train_rows),
        "consensus_labels": dict(Counter(row["consensus_label"] for row in final_rows)),
        "train_labels": dict(Counter(row["label"] for row in train_rows)),
        "by_candidate_kind": {},
        "positive_votes_required": args.positive_votes,
        "negative_votes_required": args.negative_votes,
    }
    by_kind: dict[str, Counter[str]] = {}
    for row in final_rows:
        kind = str(row.get("candidate_kind", "unknown"))
        by_kind.setdefault(kind, Counter())
        by_kind[kind][row["consensus_label"]] += 1
    summary["by_candidate_kind"] = {kind: dict(counts) for kind, counts in by_kind.items()}
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
