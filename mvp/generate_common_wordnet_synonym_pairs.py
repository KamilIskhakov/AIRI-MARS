#!/usr/bin/env python3
"""Generate positive common-noun candidates from direct WordNet synonyms.

This stage uses no remote model and complements embedding retrieval, which is
excellent at finding close negatives but often misses a clean positive pair.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from audit_entity_tags_and_probe import connect_ro, load_tags, one_mention
from generate_common_embedding_pairs import (
    clean_surface,
    inventory_items,
    make_pair,
    simple_inflection_pair,
    unique_items,
    write_jsonl,
)


def source_items(args: argparse.Namespace, conn: Any) -> list[dict[str, Any]]:
    if args.source == "inventory":
        return inventory_items(
            conn,
            args.inventory_limit,
            args.min_mentions,
            {"common_entity", "domain_term"},
            True,
        )
    rows = unique_items(load_tags(args.tag_files))
    return [
        row for row in rows
        if row.get("coarse_group") == "common_entity"
        and row.get("fine_type") == "COMMON_NOUN"
        and clean_surface(str(row.get("entity", "")), strict_lexical=True)
    ]


def synonyms(surface: str) -> list[tuple[str, float, str]]:
    try:
        from nltk.corpus import wordnet as wn
    except ImportError as exc:
        raise RuntimeError("WordNet requires nltk; install nltk and its wordnet data") from exc
    tokens = re.findall(r"[a-z]+", surface.casefold())
    if not tokens:
        return []
    source = tokens[-1]
    output: dict[str, tuple[float, str]] = {}
    try:
        synsets = wn.synsets(source, pos=wn.NOUN)
    except LookupError as exc:
        raise RuntimeError("Run: python -m nltk.downloader wordnet") from exc
    for synset in synsets:
        for lemma in synset.lemmas():
            candidate = re.sub(r"[_-]+", " ", lemma.name()).casefold().strip()
            candidate = re.sub(r"\s+", " ", candidate)
            if not clean_surface(candidate, strict_lexical=True):
                continue
            if candidate == surface.casefold() or simple_inflection_pair(surface, candidate):
                continue
            # Lemma frequency is a useful deterministic preference, not a label.
            score = float(lemma.count())
            previous = output.get(candidate)
            if previous is None or score > previous[0]:
                output[candidate] = (score, synset.name())
    return sorted(((candidate, score, synset) for candidate, (score, synset) in output.items()), key=lambda item: (-item[1], item[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["inventory", "tags"], default="inventory")
    parser.add_argument("--tag-files", type=Path, nargs="*", default=[])
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--synonyms-per-source", type=int, default=3)
    parser.add_argument("--mentions-per-source", type=int, default=1)
    parser.add_argument("--inventory-limit", type=int, default=8000)
    parser.add_argument("--min-mentions", type=int, default=2)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    if args.source == "tags" and not args.tag_files:
        raise ValueError("--tag-files is required for --source=tags")

    rng = random.Random(args.seed)
    conn = connect_ro(args.db)
    sources = source_items(args, conn)
    rng.shuffle(sources)
    if args.max_sources:
        sources = sources[: args.max_sources]
    pairs: list[dict[str, Any]] = []
    skipped = Counter()
    for source in sources:
        candidates = synonyms(str(source["entity"]))[: args.synonyms_per_source]
        if not candidates:
            skipped["no_wordnet_synonyms"] += 1
            continue
        mentions = []
        for _ in range(args.mentions_per_source):
            mention = one_mention(conn, str(source["entity_id"]), rng)
            if mention:
                mentions.append(mention)
        if not mentions:
            skipped["no_mentions"] += 1
            continue
        for mention in mentions:
            for candidate, lemma_count, synset_name in candidates:
                candidate_row = {
                    "entity_id": f"wordnet:{candidate}",
                    "entity": candidate,
                    "fine_type": source.get("fine_type", "COMMON_NOUN"),
                    "coarse_group": source.get("coarse_group", "common_entity"),
                }
                row = make_pair(
                    conn, source, candidate_row, mention, 1.0,
                    "common_wordnet_synonym", f"cs{len(pairs) + 1:07d}", 360,
                )
                row.update({
                    "branch": "common_wordnet_synonym",
                    "candidate_kind": "common_wordnet_synonym",
                    "expected_score": 1.0,
                    "label": 1,
                    "weak_label": 1,
                    "weak_label_reason": "direct_wordnet_synonym",
                    "weak_label_confidence": 0.78,
                    "wordnet_synset": synset_name,
                    "wordnet_lemma_count": lemma_count,
                    "annotation_source": "local_wordnet_weak_label",
                })
                pairs.append(row)

    summary = {
        "source": args.source,
        "sources": len(sources),
        "pairs": len(pairs),
        "labels": dict(Counter(str(row["label"]) for row in pairs)),
        "skipped": dict(skipped),
        "warning": "WordNet positives are weak labels and require a small calibration audit",
    }
    write_jsonl(args.out, pairs)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
