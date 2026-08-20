#!/usr/bin/env python3
"""Audit substitution data before candidate judging or model training.

The audit is deliberately dependency-free so it can run on a login node or on
the laptop before expensive MLM/LLM stages. It reports critical row defects,
label balance, explicit generator agreement, and coverage by entity type.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def norm(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", value).strip()


def compact(value: Any) -> str:
    return re.sub(r"[^\w]+", "", norm(value))


def label(row: dict[str, Any]) -> str:
    value = row.get("label")
    if value in {0, "0"}:
        return "changed"
    if value in {1, "1"}:
        return "preserved"
    value = row.get("consensus_label", row.get("judge_label"))
    return str(value or "missing")


def replace_nth(text: str, needle: str, replacement: str, index: int) -> str | None:
    start = 0
    for _ in range(index + 1):
        pos = text.find(needle, start)
        if pos < 0:
            return None
        start = pos + len(needle)
    return text[:pos] + replacement + text[pos + len(needle) :]


def replace_matching_occurrence(text: str, needle: str, replacement: str, target: str) -> str | None:
    if not needle:
        return None
    for match in re.finditer(re.escape(needle), text, flags=re.IGNORECASE):
        rebuilt = text[:match.start()] + replacement + text[match.end():]
        if norm(rebuilt) == norm(target):
            return rebuilt
    return None


def materialized(row: dict[str, Any]) -> tuple[str, str, str]:
    left = str(row.get("left") or row.get("original_context") or "")
    right = str(row.get("right") or row.get("candidate_context") or "")
    entity = str(row.get("entity") or "")
    candidate = str(row.get("candidate") or "")
    if "<mask>" in left and candidate:
        try:
            preferred_idx = int(row.get("mask_idx", -1))
        except (TypeError, ValueError):
            preferred_idx = -1
        candidate_left = replace_nth(left, "<mask>", candidate, preferred_idx)
        if candidate_left is not None and (candidate_left == right or (len(candidate_left) == len(right) and norm(candidate_left) == norm(right))):
            return replace_nth(left, "<mask>", entity, preferred_idx) or left, right, "mask_diff"
        # Older rows may have a local mask index but a shortened context. Keep
        # the fallback bounded; full-document normalization is expensive.
        for idx in range(min(left.count("<mask>"), 24)):
            if idx == preferred_idx:
                continue
            trial = replace_nth(left, "<mask>", candidate, idx)
            if trial is not None and (trial == right or (len(trial) == len(right) and norm(trial) == norm(right))):
                return replace_nth(left, "<mask>", entity, idx) or left, right, "mask_diff"
        if 0 <= preferred_idx < left.count("<mask>"):
            return replace_nth(left, "<mask>", entity, preferred_idx) or left, right, "mask_idx_fallback"
    if entity and candidate:
        rebuilt = replace_matching_occurrence(right, candidate, entity, left)
        if rebuilt is not None:
            return rebuilt, right, "candidate_diff"
        rebuilt = replace_matching_occurrence(left, entity, candidate, right)
        if rebuilt is not None:
            return left, rebuilt, "entity_diff"
        match = re.search(re.escape(candidate), right, flags=re.IGNORECASE)
        if match:
            rebuilt = right[:match.start()] + entity + right[match.end():]
            return rebuilt, right, "candidate_reverse_replace"
    return left, right, "unresolved"


def row_flags(row: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    entity = str(row.get("entity") or "")
    candidate = str(row.get("candidate") or "")
    left = str(row.get("left") or row.get("original_context") or "")
    right = str(row.get("right") or row.get("candidate_context") or "")
    model_left, model_right, method = materialized(row)
    flags: list[str] = []
    if not entity or not candidate or not left or not right:
        flags.append("missing_required_field")
    numeric_format = (
        str(row.get("coarse_group")) == "numeric"
        and str(row.get("candidate_kind")) == "numeric_equivalent_format"
    )
    if norm(entity) == norm(candidate):
        flags.append("identity_candidate")
    elif compact(entity) == compact(candidate):
        flags.append("numeric_same_value_format" if numeric_format else "identity_candidate")
    if norm(left) == norm(right):
        flags.append("identical_model_inputs")
    if entity and norm(entity) not in norm(model_left):
        flags.append("entity_missing_after_materialization")
    if candidate and norm(candidate) not in norm(model_right):
        flags.append("candidate_missing_after_materialization")
    if method == "unresolved":
        flags.append("target_location_unresolved")
    return flags, {
        "model_left": model_left,
        "model_right": model_right,
        "context_materialization": method,
    }


def summary(rows: list[dict[str, Any]], field: str | None = None) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "unknown") if field else "all"].append(row)
    output: dict[str, Any] = {}
    for name, items in groups.items():
        output[name] = {
            "rows": len(items),
            "labels": dict(Counter(label(row) for row in items)),
            "identity": sum("identity_candidate" in row["_flags"] for row in items),
            "critical": sum(bool(row["_critical_flags"]) for row in items),
            "materialization": dict(Counter(row["materialization"] for row in items)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="NAME=PATH, repeatable")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-out", type=Path)
    parser.add_argument("--sample-per-bucket", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--require-both-labels-by", choices=["coarse_group", "entity_type", "candidate_kind"])
    parser.add_argument("--min-group-rows", type=int, default=5)
    parser.add_argument("--fail-on-critical", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {"inputs": {}, "schema": {"critical_rows": 0, "flag_counts": {}}}
    samples: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    for spec in args.input:
        name, raw_path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        path = Path(raw_path)
        source_rows = load(path)
        audited: list[dict[str, Any]] = []
        for row in source_rows:
            flags, extra = row_flags(row)
            audited.append({**row, "_flags": flags, "_critical_flags": [f for f in flags if f not in {"target_location_unresolved", "numeric_same_value_format"}], "materialization": extra["context_materialization"]})
            for flag in flags:
                report["schema"]["flag_counts"][flag] = report["schema"]["flag_counts"].get(flag, 0) + 1
        critical = [row for row in audited if row["_critical_flags"]]
        report["schema"]["critical_rows"] += len(critical)
        balance_failures: list[str] = []
        if args.require_both_labels_by:
            grouped = defaultdict(list)
            for row in audited:
                grouped[str(row.get(args.require_both_labels_by) or "unknown")].append(row)
            for group_name, group_rows in grouped.items():
                if len(group_rows) >= args.min_group_rows:
                    group_labels = {label(row) for row in group_rows}
                    if not {"changed", "preserved"}.issubset(group_labels):
                        balance_failures.append(group_name)
        report["inputs"][name] = {
            "path": str(path),
            "rows": len(audited),
            "binary_rows": sum(label(row) in {"changed", "preserved"} for row in audited),
            "uncertain_or_missing": sum(label(row) not in {"changed", "preserved"} for row in audited),
            "labels": dict(Counter(label(row) for row in audited)),
            "by_coarse_group": summary(audited, "coarse_group"),
            "by_entity_type": summary(audited, "entity_type" if any(row.get("entity_type") for row in audited) else "fine_type"),
            "by_candidate_kind": summary(audited, "candidate_kind" if any(row.get("candidate_kind") for row in audited) else "pair_kind"),
            "explicit_expected_agreement": {
                "rows": sum(row.get("expected_score") is not None for row in audited),
                "matches": sum(
                    (float(row.get("expected_score")) >= 0.9 and label(row) == "preserved")
                    or (float(row.get("expected_score")) <= 0.1 and label(row) == "changed")
                    for row in audited if row.get("expected_score") is not None
                ),
            },
            "balance_failures": balance_failures,
        }
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in audited:
            bucket = f"{row.get('coarse_group', row.get('branch', 'unknown'))}/{row.get('candidate_kind', row.get('pair_kind', 'unknown'))}/{label(row)}"
            buckets[bucket].append(row)
        for bucket, items in buckets.items():
            for row in rng.sample(items, min(args.sample_per_bucket, len(items))):
                samples.append({
                    "source": name,
                    "bucket": bucket,
                    "pair_id": row.get("pair_id"),
                    "entity": row.get("entity"),
                    "candidate": row.get("candidate"),
                    "label": label(row),
                    "flags": row["_flags"],
                    "context_materialization": row["materialization"],
                    "left": str(row.get("left") or row.get("original_context") or "")[:1200],
                    "right": str(row.get("right") or row.get("candidate_context") or "")[:1200],
                })

    report["schema"]["critical_rows"] = int(report["schema"]["critical_rows"])
    balance_failures = [
        f"{name}:{group}"
        for name, item in report["inputs"].items()
        for group in item.get("balance_failures", [])
    ]
    report["balance_failures"] = balance_failures
    report["recommendation"] = (
        "do_not_train_until_fixed" if report["schema"]["critical_rows"] or balance_failures else
        "train_only_binary_consensus_rows; keep uncertain rows for review"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.sample_out:
        args.sample_out.parent.mkdir(parents=True, exist_ok=True)
        with args.sample_out.open("w", encoding="utf-8") as fh:
            for row in samples:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_critical and (report["schema"]["critical_rows"] or balance_failures):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
