#!/usr/bin/env python3
"""Assign conservative weak labels to local common-noun retrieval pairs.

Cosine similarity is used for retrieval, not as a semantic truth signal. A
positive weak label requires a shared WordNet synset and a high enough score;
nearby candidates without that relation become hard negatives. Morphological
variants are excluded instead of being mislabeled as positives.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm(value))


def morphology_key(value: str) -> str:
    tokens = norm(value).split()
    if not tokens:
        return ""
    last = tokens[-1]
    for suffix in ("ies", "ves", "ses", "xes", "zes", "ches", "shes", "s"):
        if len(last) > len(suffix) + 2 and last.endswith(suffix):
            last = last[: -len(suffix)] + ("y" if suffix == "ies" else "f" if suffix == "ves" else "")
            break
    tokens[-1] = last
    return " ".join(tokens)


def morphological_variant(left: str, right: str) -> bool:
    if compact(left) == compact(right):
        return True
    return morphology_key(left) == morphology_key(right)


def wordnet_synsets(surface: str) -> set[str]:
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return set()
    try:
        tokens = re.findall(r"[a-z]+", norm(surface))
        if not tokens:
            return set()
        # The existing generator treats the head noun as the lexical unit.
        return {item.name() for item in wn.synsets(tokens[-1], pos=wn.NOUN)}
    except LookupError:
        return set()


def shared_synset(row: dict[str, Any]) -> bool:
    if row.get("shared_wordnet_synset") is not None:
        return bool(row.get("shared_wordnet_synset"))
    left = wordnet_synsets(str(row.get("entity", "")))
    right = wordnet_synsets(str(row.get("candidate", "")))
    return bool(left and right and left & right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--positive-min", type=float, default=0.90)
    parser.add_argument("--hard-min", type=float, default=0.90)
    parser.add_argument("--hard-max", type=float, default=0.97)
    parser.add_argument("--include-medium", action="store_true")
    parser.add_argument("--medium-min", type=float, default=0.82)
    args = parser.parse_args()
    if not 0 <= args.medium_min < args.hard_min <= args.hard_max <= 1.0:
        raise ValueError("Expected medium_min < hard_min <= hard_max <= 1")

    rows: list[dict[str, Any]] = []
    excluded_counts: Counter[str] = Counter()
    seen: set[tuple[str, str, str, str]] = set()
    for path in args.input:
        for source in load(path):
            entity = str(source.get("entity", ""))
            candidate = str(source.get("candidate", ""))
            key = (norm(entity), norm(candidate), norm(source.get("original_context") or source.get("left")), norm(source.get("candidate_context") or source.get("right")))
            if key in seen:
                continue
            seen.add(key)
            row = dict(source)
            row["weak_label"] = None
            row["weak_label_reason"] = None
            row["weak_label_confidence"] = None
            row["weak_excluded_reason"] = None
            score = float(row.get("cosine", -1.0))
            if not entity or not candidate or norm(entity) == norm(candidate):
                row["weak_excluded_reason"] = "identity"
            elif morphological_variant(entity, candidate):
                row["weak_excluded_reason"] = "morphological_variant"
            else:
                direct_wordnet = row.get("candidate_kind") == "common_wordnet_synonym"
                same = direct_wordnet or shared_synset(row)
                if direct_wordnet or (same and score >= args.positive_min):
                    row.update(
                        weak_label=1,
                        weak_label_reason="shared_wordnet_synset_high_cosine",
                        weak_label_confidence=0.82,
                        candidate_kind="common_weak_synonym",
                        expected_score=1.0,
                    )
                elif args.hard_min <= score < args.hard_max and not same:
                    row.update(
                        weak_label=0,
                        weak_label_reason="high_cosine_without_shared_synset",
                        weak_label_confidence=0.76,
                        candidate_kind="common_weak_hard_negative",
                        expected_score=0.0,
                    )
                elif args.include_medium and args.medium_min <= score < args.hard_min and not same:
                    row.update(
                        weak_label=0,
                        weak_label_reason="medium_cosine_without_shared_synset",
                        weak_label_confidence=0.64,
                        candidate_kind="common_weak_medium_negative",
                        expected_score=0.0,
                    )
                else:
                    row["weak_excluded_reason"] = "outside_conservative_band_or_unresolved_relation"
            if row["weak_label"] is not None:
                row["label"] = row["weak_label"]
                row["annotation_source"] = "local_encoder_wordnet_weak_label"
                rows.append(row)
            else:
                excluded_counts[str(row["weak_excluded_reason"])] += 1

    summary = {
        "inputs": [str(path) for path in args.input],
        "output_rows": len(rows),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "candidate_kinds": dict(Counter(str(row.get("candidate_kind")) for row in rows)),
        "weak_reasons": dict(Counter(str(row.get("weak_label_reason")) for row in rows)),
        "excluded": dict(excluded_counts),
        "thresholds": {
            "positive_min": args.positive_min,
            "hard_min": args.hard_min,
            "hard_max": args.hard_max,
            "medium_min": args.medium_min if args.include_medium else None,
        },
        "warning": "weak labels require calibration; do not treat cosine alone as semantic ground truth",
    }
    write(args.out, rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
