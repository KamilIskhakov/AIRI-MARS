#!/usr/bin/env python3
"""Retrieve high-cosine common-entity candidates and attach short contexts."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from audit_entity_tags_and_probe import connect_ro, encode_with_local_modernbert, load_tags, one_mention, short_window
from prepare_pairs import crop_around_entity, normalize_space, replace_nth_mask
from run_small_substitution_probe import TEMPORAL_OR_MEASURE_RE


GENERIC_STOP = {
    "more", "less", "other", "same", "this", "that", "these", "those",
    "something", "anything", "everything", "someone", "anyone", "people",
}
ALLOWED_FINE_TYPES = {"COMMON_NOUN", "DOMAIN_TERM"}
LEADING_STOP = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "each", "every", "either",
    "neither", "many", "much", "few", "several", "all", "both", "another", "as",
}
NON_LEXICAL_TOKENS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
    "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
    "age", "second", "seconds", "minute", "minutes", "hour", "hours", "day",
    "days", "week", "weeks", "month", "months", "year", "years", "decade",
    "decades", "century", "centuries", "season", "spring", "summer", "fall",
    "autumn", "winter", "date", "time", "era", "today", "tomorrow", "yesterday",
    "morning", "afternoon", "evening", "night", "weekend", "weekday", "semester",
    "daily", "weekly", "monthly", "annual", "annually", "quarterly", "midnight",
    "meter", "meters", "metre", "metres", "mile", "miles", "foot", "feet",
    "inch", "inches", "kilometer", "kilometers", "kilometre", "kilometres",
    "gram", "grams", "kilogram", "kilograms", "pound", "pounds", "percent",
    "percentage", "dollar", "dollars", "euro", "euros", "million", "millions",
    "billion", "billions", "trillion", "trillions", "hundred", "hundreds",
    "thousand", "thousands", "twenties", "thirties", "forties", "fifties",
    "sixties", "seventies", "eighties", "nineties", "noon", "midday", "dawn",
    "dusk", "daytime", "nighttime", "midweek", "midmorning", "halftime",
    "lunchtime", "now", "later", "before", "after", "recent", "last", "next",
    "ago", "early", "late", "earlier", "shortly", "moment", "moments", "term",
    "up", "more", "less", "least", "most", "about", "around", "nearly", "only",
    "roughly", "approximately", "over", "under", "times", "gallon", "gallons",
    "millimeter", "millimeters", "millimetre", "millimetres", "centimeter",
    "centimeters", "centimetre", "centimetres", "millisecond", "milliseconds",
    "ton", "tons", "weeknight", "weeknights", "workweek", "workweeks", "fortnight",
    "fortnightly", "weekends", "holiday", "holidays", "tomorrows", "yesterdays",
    "nights", "mornings", "afternoons", "evenings", "millennium", "millenniums",
    "millennia", "age", "ages", "aged", "thirds", "threes", "triples", "billionth",
    "millionth", "thirtieth", "sixtieth",
}
NON_ENGLISH_FUNCTION_TOKENS = {
    "ac", "ar", "bod", "bobl", "cael", "cymryd", "dywedodd", "ei", "eu", "fydd",
    "gan", "gyda", "hyn", "hynny", "iechyd", "ifanc", "llywodraeth", "llefarydd",
    "mae", "meddai", "nifer", "nhw", "oedd", "pawb", "rhan", "rhaid", "sydd",
    "wedi", "wnaeth", "yna", "ychwanegol",
}
POLAR_PREFIX_RE = re.compile(r"^(anti|pro|non|un|pre|post|ex|pan|sub)-", re.IGNORECASE)
LEXICAL_SURFACE_RE = re.compile(r"^[a-z]+(?:['-][a-z]+)*(?: [a-z]+(?:['-][a-z]+)*){0,3}$")
NUMBER_WORD_RE = re.compile(
    r"\b(zero(?:s)?|one(?:s)?|two(?:s)?|three(?:s)?|four(?:s|ths)?|five(?:s)?|"
    r"six(?:es|ths)?|seven(?:s|ths)?|eight(?:s|hs)?|nine(?:s|ths)?|ten(?:s|ths)?|eleven(?:s|ths)?|twelve(?:s|ths)?|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion|trillion|dozen|half|quarter|hundredth|thousandth|millionth|billionth)\b",
    re.IGNORECASE,
)
TEMPORAL_EXTRA_RE = re.compile(
    r"\b(age|today|tomorrow|yesterday|morning|afternoon|evening|night|weekend|"
    r"weekday|semester|daily|weekly|monthly|annual|annually|quarterly)\b",
    re.IGNORECASE,
)


def norm_key(text: str) -> str:
    return re.sub(r"[^a-zа-я]+", "", normalize_space(text).casefold())


def simple_inflection_pair(left: str, right: str) -> bool:
    a = normalize_space(left).casefold()
    b = normalize_space(right).casefold()
    if " " in a or " " in b:
        return False
    pairs = {(a, b), (b, a)}
    return any(
        plural == singular + suffix
        or (singular.endswith("y") and plural == singular[:-1] + "ies")
        or (singular.endswith("f") and plural == singular[:-1] + "ves")
        for singular, plural in pairs
        for suffix in ("s", "es")
    )


def clean_surface(text: str, strict_lexical: bool = False) -> bool:
    text = normalize_space(text)
    if len(text) < 3 or len(text) > 64:
        return False
    if TEMPORAL_OR_MEASURE_RE.search(text) or re.search(r"\d", text):
        return False
    if NUMBER_WORD_RE.search(text) or TEMPORAL_EXTRA_RE.search(text):
        return False
    if text.casefold() in GENERIC_STOP:
        return False
    if not re.search(r"[A-Za-zА-Яа-я]", text):
        return False
    if strict_lexical:
        if text != text.casefold() or not LEXICAL_SURFACE_RE.fullmatch(text):
            return False
        tokens = re.findall(r"[a-z]+", text)
        if not tokens or tokens[0] in LEADING_STOP:
            return False
        if any(token in NON_LEXICAL_TOKENS for token in tokens):
            return False
        if any(token in NON_ENGLISH_FUNCTION_TOKENS for token in tokens):
            return False
        if POLAR_PREFIX_RE.match(text):
            return False
    return len(norm_key(text)) >= 3


@lru_cache(maxsize=100_000)
def has_wordnet_noun(surface: str) -> bool:
    try:
        from nltk.corpus import wordnet as wn

        head = re.findall(r"[a-z]+", surface)[-1]
        return bool(wn.synsets(head, pos=wn.NOUN))
    except LookupError as exc:
        raise RuntimeError(
            "WordNet data is missing; run: python -m nltk.downloader wordnet"
        ) from exc


def unique_items(tags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_surface: dict[str, dict[str, Any]] = {}
    for tag in tags:
        if tag.get("coarse_group") not in {"common_entity", "domain_term"}:
            continue
        if tag.get("fine_type") not in ALLOWED_FINE_TYPES:
            continue
        surface = normalize_space(str(tag.get("entity", "")))
        if not clean_surface(surface):
            continue
        key = norm_key(surface)
        current = by_surface.get(key)
        if current is None or float(tag.get("confidence", 0)) > float(current.get("confidence", 0)):
            by_surface[key] = {**tag, "entity": surface}
    return list(by_surface.values())


def inventory_items(
    conn: Any,
    limit: int,
    min_mentions: int,
    groups: set[str],
    require_wordnet_noun: bool,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        select id, surface, heuristic_group, context_policy, mention_count,
               observed_types_json, domains_json, datasets_json
        from entities
        where heuristic_group in ('common_entity', 'domain_term')
          and mention_count >= ?
        order by mention_count desc, id
        """,
        (min_mentions,),
    ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        surface = normalize_space(str(row["surface"]))
        if not clean_surface(surface, strict_lexical=True):
            continue
        if require_wordnet_noun and not has_wordnet_noun(surface):
            continue
        group = str(row["heuristic_group"])
        if group not in groups:
            continue
        items.append(
            {
                "entity_id": str(row["id"]),
                "entity": surface,
                "coarse_group": group,
                "fine_type": "COMMON_NOUN" if group == "common_entity" else "DOMAIN_TERM",
                "provider": "inventory_heuristic",
                "mention_count": int(row["mention_count"]),
                "context_policy": str(row["context_policy"]),
                "observed_types": json.loads(row["observed_types_json"] or "{}"),
                "domains": json.loads(row["domains_json"] or "{}"),
                "datasets": json.loads(row["datasets_json"] or "{}"),
            }
        )
        if limit and len(items) >= limit:
            break
    return items


def fill_contexts(masked_text: str, mask_idx: int, entity: str, candidate: str, chars: int) -> tuple[str, str]:
    original = replace_nth_mask(masked_text, mask_idx, entity)
    substituted = replace_nth_mask(masked_text, mask_idx, candidate)
    return crop_around_entity(original, entity, chars), crop_around_entity(substituted, candidate, chars)


TARGET_MARKER = "__AIRI_TARGET_ENTITY__"


def reconstructed_contexts(
    conn: Any,
    mention: dict[str, Any],
    entity: str,
    candidate: str,
    chars: int,
) -> tuple[str, str, str]:
    mention_rows = conn.execute(
        """
        select m.mask_idx, e.surface
        from mentions m
        join entities e on e.id = m.entity_id
        where m.text_id = ?
        order by m.mask_idx
        """,
        (int(mention["text_id"]),),
    ).fetchall()
    surfaces = {int(row["mask_idx"]): str(row["surface"]) for row in mention_rows}
    target_idx = int(mention["mask_idx"])
    masked_text = str(mention["masked_text"])
    parts = masked_text.split("<mask>")
    if len(parts) - 1 != len(surfaces) or target_idx not in surfaces:
        original, substituted = fill_contexts(masked_text, target_idx, entity, candidate, chars)
        return original, substituted, short_window(masked_text, target_idx, entity, 2)

    rebuilt: list[str] = [parts[0]]
    for idx, suffix in enumerate(parts[1:]):
        rebuilt.append(TARGET_MARKER if idx == target_idx else surfaces[idx])
        rebuilt.append(suffix)
    marked = "".join(rebuilt)
    marker_pos = marked.index(TARGET_MARKER)
    lo = max(0, marker_pos - chars)
    hi = min(len(marked), marker_pos + len(TARGET_MARKER) + chars)
    crop = marked[lo:hi]
    original = normalize_space(crop.replace(TARGET_MARKER, entity, 1))
    substituted = normalize_space(crop.replace(TARGET_MARKER, candidate, 1))

    left = re.findall(r"[A-Za-zА-Яа-я0-9]+", marked[:marker_pos])[-2:]
    right_start = marker_pos + len(TARGET_MARKER)
    right = re.findall(r"[A-Za-zА-Яа-я0-9]+", marked[right_start:])[:2]
    short = " ".join(left + [f"[{entity}]"] + right)
    return original, substituted, short


def make_pair(
    conn: Any,
    source: dict[str, Any],
    candidate: dict[str, Any],
    mention: dict[str, Any],
    score: float,
    bucket: str,
    pair_id: str,
    context_chars: int,
) -> dict[str, Any]:
    original, substituted, short = reconstructed_contexts(
        conn,
        mention,
        source["entity"],
        candidate["entity"],
        context_chars,
    )
    return {
        "pair_id": pair_id,
        "branch": "common_embedding_retrieval",
        "candidate_kind": bucket,
        "retrieval_bucket": bucket,
        "expected_score": None,
        "cosine": round(score, 6),
        "entity_id": str(source["entity_id"]),
        "candidate_entity_id": str(candidate["entity_id"]),
        "entity": source["entity"],
        "candidate": candidate["entity"],
        "fine_type": source.get("fine_type"),
        "candidate_fine_type": candidate.get("fine_type"),
        "coarse_group": source.get("coarse_group"),
        "candidate_coarse_group": candidate.get("coarse_group"),
        "context_policy": "short_window",
        "provider": source.get("provider"),
        "dataset": f"{mention['dataset_name']}/{mention['dataset_split']}",
        "dataset_name": mention.get("dataset_name"),
        "dataset_split": mention.get("dataset_split"),
        "dataset_id": mention.get("dataset_id"),
        "text_id": mention.get("text_id"),
        "mention_id": mention.get("mention_id"),
        "row_idx": mention.get("row_idx"),
        "source_id": mention.get("source_id"),
        "domain": mention.get("domain"),
        "mask_idx": mention.get("mask_idx"),
        "short_2w": short,
        "original_context": original,
        "candidate_context": substituted,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def tagging_inventory(
    conn: Any,
    items: list[dict[str, Any]],
    rng: random.Random,
    context_chars: int,
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for item in items:
        mention = one_mention(conn, str(item["entity_id"]), rng)
        examples: list[str] = []
        if mention is not None:
            context, _, _ = reconstructed_contexts(
                conn,
                mention,
                str(item["entity"]),
                str(item["entity"]),
                context_chars,
            )
            examples.append(context)
        exported.append(
            {
                "entity_id": str(item["entity_id"]),
                "entity": item["entity"],
                "count": int(item.get("mention_count", 0)),
                "observed_types": item.get("observed_types", {}),
                "domains": item.get("domains", {}),
                "datasets": item.get("datasets", {}),
                "heuristic_group": item.get("coarse_group"),
                "heuristic_context_policy": item.get("context_policy", "no_context_embedding"),
                "heuristic_type": item.get("fine_type"),
                "examples": examples,
            }
        )
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("tags", "inventory"), default="tags")
    parser.add_argument("--tag-files", type=Path, nargs="*", default=[])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--neighbors-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--items-out", type=Path)
    parser.add_argument("--very-high-min", type=float, default=0.97)
    parser.add_argument("--hard-min", type=float, default=0.93)
    parser.add_argument("--hard-max", type=float, default=0.97)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--pairs-per-bucket", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-chars", type=int, default=360)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--inventory-limit", type=int, default=8000)
    parser.add_argument("--min-mentions", type=int, default=2)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("common_entity", "domain_term"),
        default=["common_entity", "domain_term"],
    )
    parser.add_argument("--allow-cross-group", action="store_true")
    parser.add_argument("--require-wordnet-noun", action="store_true")
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    if not (0 <= args.hard_min < args.hard_max <= args.very_high_min <= 1.01):
        raise ValueError("Expected hard_min < hard_max <= very_high_min")

    rng = random.Random(args.seed)
    conn = connect_ro(args.db)
    tags = load_tags(args.tag_files) if args.tag_files else []
    if args.source == "tags":
        if not tags:
            raise ValueError("--tag-files is required when --source=tags")
        items = [row for row in unique_items(tags) if row.get("coarse_group") in set(args.groups)]
        if args.require_wordnet_noun:
            items = [row for row in items if has_wordnet_noun(str(row["entity"]))]
    else:
        items = inventory_items(
            conn,
            args.inventory_limit,
            args.min_mentions,
            set(args.groups),
            args.require_wordnet_noun,
        )
    items.sort(key=lambda row: str(row["entity"]).casefold())
    if args.max_items:
        rng.shuffle(items)
        items = items[: args.max_items]
    if len(items) < 2:
        raise RuntimeError("Not enough clean common/domain surfaces")
    if args.items_out:
        write_jsonl(args.items_out, tagging_inventory(conn, items, rng, args.context_chars))

    vectors = encode_with_local_modernbert([row["entity"] for row in items], args.model, args.batch_size)
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -1.0)
    if not args.allow_cross_group:
        groups = np.asarray([row["coarse_group"] for row in items])
        similarities[groups[:, None] != groups[None, :]] = -1.0

    neighbors: list[dict[str, Any]] = []
    selected: list[tuple[int, int, float, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    bucket_counts: Counter[str] = Counter()
    top1: list[float] = []
    for source_idx, source in enumerate(items):
        order = np.argsort(-similarities[source_idx])[: args.neighbors]
        source_neighbors: list[dict[str, Any]] = []
        per_bucket: Counter[str] = Counter()
        for candidate_idx in order:
            candidate = items[int(candidate_idx)]
            if simple_inflection_pair(source["entity"], candidate["entity"]):
                continue
            score = float(similarities[source_idx, candidate_idx])
            bucket = (
                "common_very_high_cosine"
                if score >= args.very_high_min
                else "common_high_cosine_hard"
                if args.hard_min <= score < args.hard_max
                else "below_threshold"
            )
            source_neighbors.append(
                {
                    "candidate_entity_id": str(candidate["entity_id"]),
                    "candidate": candidate["entity"],
                    "candidate_coarse_group": candidate.get("coarse_group"),
                    "candidate_fine_type": candidate.get("fine_type"),
                    "cosine": round(score, 6),
                    "bucket": bucket,
                }
            )
            if bucket == "below_threshold" or per_bucket[bucket] >= args.pairs_per_bucket:
                continue
            key = tuple(sorted((str(source["entity_id"]), str(candidate["entity_id"]))))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            selected.append((source_idx, int(candidate_idx), score, bucket))
            per_bucket[bucket] += 1
            bucket_counts[bucket] += 1
        top1.append(float(similarities[source_idx, order[0]]))
        neighbors.append(
            {
                "entity_id": str(source["entity_id"]),
                "entity": source["entity"],
                "coarse_group": source.get("coarse_group"),
                "fine_type": source.get("fine_type"),
                "neighbors": source_neighbors,
            }
        )

    pairs: list[dict[str, Any]] = []
    skipped_no_mention = 0
    rng.shuffle(selected)
    for source_idx, candidate_idx, score, bucket in selected:
        source = items[source_idx]
        mention = one_mention(conn, str(source["entity_id"]), rng)
        if mention is None:
            skipped_no_mention += 1
            continue
        pairs.append(
            make_pair(
                conn,
                source,
                items[candidate_idx],
                mention,
                score,
                bucket,
                f"ce{len(pairs) + 1:07d}",
                args.context_chars,
            )
        )

    summary = {
        "source": args.source,
        "tag_rows": len(tags),
        "clean_items": len(items),
        "selected_neighbor_pairs": len(selected),
        "context_pairs": len(pairs),
        "skipped_no_mention": skipped_no_mention,
        "thresholds": {
            "very_high_min": args.very_high_min,
            "hard_min": args.hard_min,
            "hard_max": args.hard_max,
        },
        "selected_by_bucket": dict(bucket_counts),
        "context_pairs_by_bucket": dict(Counter(row["retrieval_bucket"] for row in pairs)),
        "items_by_group": dict(Counter(str(row.get("coarse_group")) for row in items)),
        "top1_bins": dict(
            Counter(
                ">=very_high" if score >= args.very_high_min else "hard_range" if args.hard_min <= score < args.hard_max else "below"
                for score in top1
            )
        ),
        "top1_mean": round(float(np.mean(top1)), 6),
        "top1_median": round(float(np.median(top1)), 6),
    }
    write_jsonl(args.neighbors_out, neighbors)
    write_jsonl(args.out, pairs)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
