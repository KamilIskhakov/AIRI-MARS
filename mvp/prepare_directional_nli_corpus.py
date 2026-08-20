#!/usr/bin/env python3
"""Convert agent-reviewed directional relations into ordered NLI examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID = {"entailment", "neutral", "contradiction"}


def stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    args = parser.parse_args()

    examples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            confidence = float(row.get("directional_confidence", row.get("confidence", 0.0)))
            if confidence < args.min_confidence:
                skipped["low_confidence"] += 1
                continue
            left = row.get("left") or row.get("original_context")
            right = row.get("right") or row.get("candidate_context")
            if not left or not right:
                skipped["missing_text"] += 1
                continue
            forward = row.get("directional_a_to_b", row.get("a_to_b"))
            backward = row.get("directional_b_to_a", row.get("b_to_a"))
            if forward not in VALID or backward not in VALID:
                skipped["uncertain_relation"] += 1
                continue
            relation = str(row.get("directional_relation", row.get("relation", "unknown")))
            relation_counts[relation] += 1
            base = {
                "source_pair_id": row.get("pair_id"),
                "entity": row.get("entity"),
                "candidate": row.get("candidate"),
                "entity_type": row.get("entity_type"),
                "coarse_group": row.get("coarse_group"),
                "mention_id": row.get("mention_id"),
                "entity_id": row.get("entity_id"),
                "text_id": row.get("text_id"),
                "dataset": row.get("dataset"),
                "domain": row.get("domain"),
                "relation": relation,
                "confidence": confidence,
                "rationale": row.get("directional_rationale", row.get("rationale")),
            }
            for direction, premise, hypothesis, label in (
                ("a_to_b", left, right, forward),
                ("b_to_a", right, left, backward),
            ):
                item = dict(base)
                item.update(
                    {
                        "nli_id": stable_id(row.get("pair_id"), direction),
                        "direction": direction,
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "label": label,
                    }
                )
                examples.append(item)
                label_counts[label] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for item in examples:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "input": str(args.input),
        "examples": len(examples),
        "source_pairs": len(examples) // 2,
        "min_confidence": args.min_confidence,
        "labels": dict(sorted(label_counts.items())),
        "relations": dict(sorted(relation_counts.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
