#!/usr/bin/env python3
"""Audit generated substitution annotations before using them for training."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


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


def is_identity(row: dict[str, Any]) -> bool:
    entity = str(row.get("entity", ""))
    candidate = str(row.get("candidate", ""))
    return (
        norm_literal(entity) == norm_literal(candidate)
        or norm_alnum(entity) == norm_alnum(candidate)
        or norm_article(entity) == norm_article(candidate)
    )


def compact_counts(rows: list[dict[str, Any]], key: str, top_n: int = 20) -> dict[str, int]:
    counts = Counter(str(row.get(key, "missing")) for row in rows)
    return dict(counts.most_common(top_n))


def first_useful_counts(rows: list[dict[str, Any]], keys: list[str], top_n: int = 20) -> dict[str, int]:
    for key in keys:
        counts = compact_counts(rows, key, top_n)
        if counts and set(counts) != {"missing"}:
            return counts
    return compact_counts(rows, keys[0], top_n) if keys else {}


def summarize(rows: list[dict[str, Any]], label_key: str) -> dict[str, Any]:
    identities = sum(1 for row in rows if is_identity(row))
    duplicate_keys: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        duplicate_keys[
            (
                row.get(label_key),
                row.get("entity_id"),
                norm_literal(str(row.get("entity", ""))),
                norm_literal(str(row.get("candidate", ""))),
                row.get("left") or row.get("original_context"),
                row.get("right") or row.get("candidate_context"),
            )
        ] += 1
    duplicate_rows = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    fields = [
        "dataset",
        "dataset_name",
        "dataset_split",
        "text_id",
        "mention_id",
        "source_id",
        "domain",
    ]
    field_presence = {
        field: sum(1 for row in rows if row.get(field) not in {None, ""})
        for field in fields
    }
    return {
        "rows": len(rows),
        "identity_rows": identities,
        "duplicate_rows": duplicate_rows,
        "labels": compact_counts(rows, label_key),
        "candidate_kind": first_useful_counts(rows, ["candidate_kind", "pair_kind"]),
        "pair_kind": first_useful_counts(rows, ["pair_kind", "candidate_kind"]),
        "entity_type": first_useful_counts(rows, ["fine_type", "entity_type"]),
        "datasets": compact_counts(rows, "dataset"),
        "domains": compact_counts(rows, "domain"),
        "unique_entities": len({str(row.get("entity_id")) for row in rows if row.get("entity_id") is not None}),
        "unique_texts": len({str(row.get("text_id")) for row in rows if row.get("text_id") not in {None, ""}}),
        "field_presence": field_presence,
    }


def examples(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("consensus_label", row.get("label", "missing")))
        key += "/" + str(row.get("candidate_kind", row.get("pair_kind", "unknown")))
        buckets[key].append(row)
    picked: list[dict[str, Any]] = []
    for key in sorted(buckets):
        sample = rng.sample(buckets[key], min(limit, len(buckets[key])))
        for row in sample:
            picked.append(
                {
                    "bucket": key,
                    "pair_id": row.get("pair_id"),
                    "entity_id": row.get("entity_id"),
                    "entity": row.get("entity"),
                    "candidate": row.get("candidate"),
                    "label": row.get("consensus_label", row.get("label")),
                    "candidate_kind": row.get("candidate_kind", row.get("pair_kind")),
                    "entity_type": row.get("fine_type", row.get("entity_type")),
                    "dataset": row.get("dataset"),
                    "domain": row.get("domain"),
                    "rationale": row.get("consensus_rationale", row.get("rationale")),
                    "identity_like": is_identity(row),
                    "left": row.get("original_context", row.get("left", ""))[:900],
                    "right": row.get("candidate_context", row.get("right", ""))[:900],
                }
            )
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--consensus", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--sample-out", type=Path)
    parser.add_argument("--examples-per-bucket", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    pairs = load_jsonl(args.pairs)
    consensus = load_jsonl(args.consensus)
    train = load_jsonl(args.train)
    report = {
        "pairs": summarize(pairs, "expected_score") if pairs else None,
        "consensus": summarize(consensus, "consensus_label") if consensus else None,
        "train": summarize(train, "label") if train else None,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.sample_out:
        sample_source = consensus or train or pairs
        with args.sample_out.open("w", encoding="utf-8") as out:
            for row in examples(sample_source, args.examples_per_bucket, args.seed):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
