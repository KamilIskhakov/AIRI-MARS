#!/usr/bin/env python3
"""Retrieve common-noun substitutions from WordNet with a local encoder."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from audit_entity_tags_and_probe import connect_ro, encode_with_local_modernbert, load_tags, one_mention
from generate_common_embedding_pairs import (
    clean_surface,
    has_wordnet_noun,
    make_pair,
    norm_key,
    simple_inflection_pair,
    unique_items,
    write_jsonl,
)


def wordnet_nouns() -> tuple[list[str], dict[str, set[str]]]:
    try:
        from nltk.corpus import wordnet as wn
    except LookupError as exc:
        raise RuntimeError("Run: python -m nltk.downloader wordnet") from exc

    surfaces: set[str] = set()
    synsets_by_surface: dict[str, set[str]] = {}
    for synset in wn.all_synsets(pos=wn.NOUN):
        synset_name = synset.name()
        for lemma in synset.lemmas():
            surface = lemma.name().replace("_", " ").replace("-", " ").casefold()
            surface = re.sub(r"\s+", " ", surface).strip()
            if not clean_surface(surface, strict_lexical=True):
                continue
            if not has_wordnet_noun(surface):
                continue
            surfaces.add(surface)
            synsets_by_surface.setdefault(surface, set()).add(synset_name)
    return sorted(surfaces), synsets_by_surface


def source_items(paths: list[Path], confidence: float) -> list[dict[str, Any]]:
    rows = unique_items(load_tags(paths))
    return sorted(
        (
            row
            for row in rows
            if row.get("coarse_group") == "common_entity"
            and row.get("fine_type") == "COMMON_NOUN"
            and float(row.get("confidence", 0)) >= confidence
            and clean_surface(str(row["entity"]), strict_lexical=True)
            and has_wordnet_noun(str(row["entity"]))
        ),
        key=lambda row: str(row["entity"]).casefold(),
    )


def source_synsets(surface: str) -> set[str]:
    from nltk.corpus import wordnet as wn

    head = re.findall(r"[a-z]+", surface.casefold())[-1]
    return {synset.name() for synset in wn.synsets(head, pos=wn.NOUN)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-files", type=Path, nargs="+", required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--neighbors-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--very-high-min", type=float, default=0.97)
    parser.add_argument("--hard-min", type=float, default=0.90)
    parser.add_argument("--hard-max", type=float, default=0.97)
    parser.add_argument("--neighbors", type=int, default=80)
    parser.add_argument("--pairs-per-bucket", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--context-chars", type=int, default=500)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--max-wordnet-items", type=int, default=0)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    if not (0 <= args.hard_min < args.hard_max <= args.very_high_min <= 1.01):
        raise ValueError("Expected hard_min < hard_max <= very_high_min")

    rng = random.Random(args.seed)
    sources = source_items(args.tag_files, args.min_confidence)
    candidates, synsets_by_surface = wordnet_nouns()
    if args.max_wordnet_items:
        candidates = candidates[: args.max_wordnet_items]
    if not sources or not candidates:
        raise RuntimeError("No clean sources or WordNet candidates")

    source_texts = [f"query: {row['entity']}" for row in sources]
    candidate_texts = [f"passage: {surface}" for surface in candidates]
    vectors = encode_with_local_modernbert(source_texts + candidate_texts, args.model, args.batch_size)
    source_vectors = vectors[: len(sources)]
    candidate_vectors = vectors[len(sources) :]
    similarities = source_vectors @ candidate_vectors.T

    conn = connect_ro(args.db)
    pairs: list[dict[str, Any]] = []
    neighbors: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    skipped_no_mention = 0
    for source_idx, source in enumerate(sources):
        order = np.argsort(-similarities[source_idx])[: args.neighbors]
        source_key = norm_key(str(source["entity"]))
        source_synset_names = source_synsets(str(source["entity"]))
        selected: Counter[str] = Counter()
        neighbor_rows: list[dict[str, Any]] = []
        mention = one_mention(conn, str(source["entity_id"]), rng)
        for candidate_idx in order:
            candidate = candidates[int(candidate_idx)]
            score = float(similarities[source_idx, candidate_idx])
            if norm_key(candidate) == source_key or simple_inflection_pair(str(source["entity"]), candidate):
                continue
            bucket = (
                "common_wordnet_very_high_cosine"
                if score >= args.very_high_min
                else "common_wordnet_high_cosine_hard"
                if args.hard_min <= score < args.hard_max
                else "below_threshold"
            )
            shared_synset = bool(source_synset_names & synsets_by_surface.get(candidate, set()))
            neighbor_rows.append(
                {
                    "candidate": candidate,
                    "cosine": round(score, 6),
                    "bucket": bucket,
                    "shared_wordnet_synset": shared_synset,
                }
            )
            if bucket == "below_threshold" or selected[bucket] >= args.pairs_per_bucket:
                continue
            if mention is None:
                skipped_no_mention += 1
                break
            candidate_row = {
                "entity_id": f"wordnet:{norm_key(candidate)}",
                "entity": candidate,
                "fine_type": "COMMON_NOUN",
                "coarse_group": "common_entity",
            }
            pair = make_pair(
                conn,
                source,
                candidate_row,
                mention,
                score,
                bucket,
                f"cw{len(pairs) + 1:07d}",
                args.context_chars,
            )
            pair["branch"] = "common_wordnet_embedding_retrieval"
            pair["shared_wordnet_synset"] = shared_synset
            pairs.append(pair)
            selected[bucket] += 1
            bucket_counts[bucket] += 1
        neighbors.append(
            {
                "entity_id": str(source["entity_id"]),
                "entity": source["entity"],
                "neighbors": neighbor_rows,
            }
        )

    summary = {
        "sources": len(sources),
        "wordnet_candidates": len(candidates),
        "pairs": len(pairs),
        "pairs_by_bucket": dict(bucket_counts),
        "shared_wordnet_synset_pairs": sum(bool(row["shared_wordnet_synset"]) for row in pairs),
        "skipped_no_mention": skipped_no_mention,
        "thresholds": {
            "very_high_min": args.very_high_min,
            "hard_min": args.hard_min,
            "hard_max": args.hard_max,
        },
    }
    write_jsonl(args.out, pairs)
    write_jsonl(args.neighbors_out, neighbors)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
