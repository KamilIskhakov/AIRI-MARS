#!/usr/bin/env python3
"""Build an audited substitution corpus and entity/document-disjoint splits."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def entity_key(value: Any) -> str:
    text = normalize_text(value).replace("&", "and")
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"(?:'s|’s)$", "", text)
    return re.sub(r"[^\w]+", "", text)


def id_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def label_of(row: dict[str, Any]) -> int | None:
    value = row.get("label")
    if value in {0, 1, "0", "1"}:
        return int(value)
    value = row.get("final_label", row.get("consensus_label", row.get("judge_label")))
    if value == "preserved":
        return 1
    if value == "changed":
        return 0
    return None


def normalize_row(row: dict[str, Any], source_name: str) -> dict[str, Any] | None:
    label = label_of(row)
    left = row.get("left") or row.get("original_context")
    right = row.get("right") or row.get("candidate_context")
    entity = str(row.get("entity", "")).strip()
    candidate = str(row.get("candidate", "")).strip()
    if label is None or not left or not right or not entity or not candidate:
        return None
    return {
        **row,
        "left": str(left),
        "right": str(right),
        "label": label,
        "entity": entity,
        "candidate": candidate,
        "entity_id": id_text(row.get("entity_id")),
        "candidate_entity_id": id_text(row.get("candidate_entity_id")),
        "entity_type": str(row.get("entity_type", row.get("fine_type", "UNKNOWN"))),
        "coarse_group": str(row.get("coarse_group", "unknown")),
        "pair_kind": str(row.get("pair_kind", row.get("candidate_kind", "unknown"))),
        "branch": str(row.get("branch", "unknown")),
        "source_id": id_text(row.get("source_id")),
        "text_id": id_text(row.get("text_id")),
        "mention_id": id_text(row.get("mention_id")),
        "dataset": str(row.get("dataset", row.get("dataset_name", "unknown"))),
        "domain": str(row.get("domain", "unknown")),
        "corpus_source": source_name,
    }


def model_input_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        normalize_text(row["left"]),
        normalize_text(row["right"]),
        normalize_text(row["entity"]),
        normalize_text(row["candidate"]),
    )


def numeric_signature(value: str) -> tuple[str, ...]:
    return tuple(match.replace(",", "") for match in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", value))


def known_non_equivalent(left: str, right: str) -> bool:
    pair = frozenset({entity_key(left), entity_key(right)})
    return pair in {
        frozenset({"biannual", "biennial"}),
        frozenset({"annual", "biennial"}),
        frozenset({"annual", "biannual"}),
    }


def invalid_reason(row: dict[str, Any]) -> str | None:
    if entity_key(row["entity"]) == entity_key(row["candidate"]):
        return "identity_or_normalized_identity"
    if normalize_text(row["left"]) == normalize_text(row["right"]):
        return "identical_model_inputs"
    if normalize_text(row["entity"]) not in normalize_text(row["left"]):
        return "entity_missing_from_left"
    if normalize_text(row["candidate"]) not in normalize_text(row["right"]):
        return "candidate_missing_from_right"
    if int(row["label"]) == 1 and known_non_equivalent(row["entity"], row["candidate"]):
        return "known_non_equivalent_positive"
    if int(row["label"]) == 1 and row.get("coarse_group") == "numeric":
        source_numbers = numeric_signature(row["entity"])
        candidate_numbers = numeric_signature(row["candidate"])
        if source_numbers and candidate_numbers and source_numbers != candidate_numbers:
            return "numeric_value_changed_positive"
    return None


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left = self.find(left)
        right = self.find(right)
        if left != right:
            if self.size[left] < self.size[right]:
                left, right = right, left
            self.parent[right] = left
            self.size[left] += self.size[right]


def split_components(
    rows: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    dsu = DisjointSet()
    for idx, row in enumerate(rows):
        row_node = f"row:{idx}"
        source_entity = row["entity_id"] or entity_key(row["entity"]) or str(idx)
        dsu.union(row_node, f"entity:{source_entity}")
        document = row["source_id"] or row["text_id"]
        if document:
            dsu.union(row_node, f"document:{document}")

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        components[dsu.find(f"row:{idx}")].append(row)

    rng = random.Random(seed)
    groups = list(components.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    targets = {
        "train": len(rows) * (1 - val_ratio - test_ratio),
        "val": len(rows) * val_ratio,
        "test": len(rows) * test_ratio,
    }
    output: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    source_totals = Counter(str(row.get("corpus_source")) for row in rows)
    rare_sources = {
        source for source, count in source_totals.items()
        if count < max(500, int(len(rows) * 0.05))
    }
    rare_targets = {
        name: sum(source_totals[source] for source in rare_sources) * ratio
        for name, ratio in {
            "train": 1 - val_ratio - test_ratio,
            "val": val_ratio,
            "test": test_ratio,
        }.items()
    }
    rare_counts = Counter()
    rare_groups = [
        group for group in groups
        if any(str(row.get("corpus_source")) in rare_sources for row in group)
    ]
    rare_group_ids = {id(group) for group in rare_groups}
    regular_groups = [group for group in groups if id(group) not in rare_group_ids]

    for group in rare_groups:
        group_rare = sum(str(row.get("corpus_source")) in rare_sources for row in group)
        destination = min(
            output,
            key=lambda name: (
                rare_counts[name] / max(rare_targets[name], 1.0),
                len(output[name]) / max(targets[name], 1.0),
                {"test": 0, "val": 1, "train": 2}[name],
            ),
        )
        output[destination].extend(group)
        rare_counts[destination] += group_rare

    for group in regular_groups:
        destination = min(
            output,
            key=lambda name: (
                len(output[name]) / max(targets[name], 1.0),
                {"test": 0, "val": 1, "train": 2}[name],
            ),
        )
        output[destination].extend(group)
    return output


def count_values(rows: list[dict[str, Any]], field: str, limit: int = 30) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "missing")) for row in rows).most_common(limit))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": count_values(rows, "label"),
        "sources": count_values(rows, "corpus_source"),
        "coarse_groups": count_values(rows, "coarse_group"),
        "entity_types": count_values(rows, "entity_type"),
        "pair_kinds": count_values(rows, "pair_kind"),
        "domains": count_values(rows, "domain"),
        "unique_entities": len({row["entity_id"] or entity_key(row["entity"]) for row in rows}),
        "unique_documents": len({row["source_id"] or row["text_id"] for row in rows if row["source_id"] or row["text_id"]}),
        "unique_candidates": len({entity_key(row["candidate"]) for row in rows}),
    }


def overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, int]:
    def values(rows: list[dict[str, Any]], kind: str) -> set[Any]:
        if kind == "entities":
            return {row["entity_id"] or entity_key(row["entity"]) for row in rows}
        if kind == "documents":
            return {row["source_id"] or row["text_id"] for row in rows if row["source_id"] or row["text_id"]}
        if kind == "candidates":
            return {entity_key(row["candidate"]) for row in rows}
        return {model_input_key(row) for row in rows}

    return {
        kind: len(values(left, kind) & values(right, kind))
        for kind in ("entities", "documents", "candidates", "model_inputs")
    }


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="NAME=PATH; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--review-limit", type=int, default=5000)
    args = parser.parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Invalid validation/test ratios")

    raw_count = 0
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for value in args.input:
        name, path = parse_input(value)
        source_rows = load_jsonl(path)
        raw_count += len(source_rows)
        for source_row in source_rows:
            row = normalize_row(source_row, name)
            if row is None:
                rejected.append({**source_row, "review_reason": "invalid_schema_or_label", "corpus_source": name})
                continue
            reason = invalid_reason(row)
            if reason:
                rejected.append({**row, "review_reason": reason})
            else:
                normalized.append(row)

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[model_input_key(row)].append(row)
    clean: list[dict[str, Any]] = []
    duplicate_rows = 0
    conflict_rows = 0
    for rows in grouped.values():
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            conflict_rows += len(rows)
            rejected.extend({**row, "review_reason": "conflicting_duplicate_labels"} for row in rows)
            continue
        clean.append(rows[0])
        duplicate_rows += len(rows) - 1

    splits = split_components(clean, args.val_ratio, args.test_ratio, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "all.jsonl", clean)
    for name, rows in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)
    write_jsonl(args.output_dir / "review.jsonl", rejected[: args.review_limit])

    report = {
        "raw_rows": raw_count,
        "accepted_rows": len(clean),
        "rejected_rows": len(rejected),
        "exact_duplicates_removed": duplicate_rows,
        "conflicting_rows_removed": conflict_rows,
        "rejection_reasons": dict(Counter(str(row.get("review_reason")) for row in rejected)),
        "all": summarize(clean),
        "splits": {name: summarize(rows) for name, rows in splits.items()},
        "overlap": {
            "train_val": overlap(splits["train"], splits["val"]),
            "train_test": overlap(splits["train"], splits["test"]),
            "val_test": overlap(splits["val"], splits["test"]),
        },
        "split_policy": "connected components over source entity_id and source_id/text_id",
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
