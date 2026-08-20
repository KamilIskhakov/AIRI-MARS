#!/usr/bin/env python3
"""Generate equivalent-format and close-but-changed numeric substitutions."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from audit_entity_tags_and_probe import connect_ro, load_tags, one_mention
from generate_common_embedding_pairs import make_pair, write_jsonl
from prepare_pairs import normalize_space


NUMERIC_TYPES = {"DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT", "QUANTITY"}
MONTHS = {
    "January": "Jan.", "February": "Feb.", "March": "Mar.", "April": "Apr.",
    "June": "Jun.", "July": "Jul.", "August": "Aug.", "September": "Sept.",
    "October": "Oct.", "November": "Nov.", "December": "Dec.",
}
NUMBER_WORD_NEIGHBORS = {
    "zero": "one", "one": "two", "two": "three", "three": "four", "four": "five",
    "five": "six", "six": "seven", "seven": "eight", "eight": "nine", "nine": "ten",
    "ten": "eleven", "eleven": "twelve", "twelve": "thirteen", "thirteen": "fourteen",
    "fourteen": "fifteen", "fifteen": "sixteen", "sixteen": "seventeen",
    "seventeen": "eighteen", "eighteen": "nineteen", "nineteen": "twenty",
    "twenty": "thirty", "thirty": "forty", "forty": "fifty", "fifty": "sixty",
    "sixty": "seventy", "seventy": "eighty", "eighty": "ninety", "ninety": "one hundred",
    "hundred": "thousand", "thousand": "million", "million": "billion",
}


def unique_numeric(tags: list[dict[str, Any]], confidence: float) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in tags:
        if row.get("coarse_group") != "numeric" or row.get("fine_type") not in NUMERIC_TYPES:
            continue
        if float(row.get("confidence", 0)) < confidence:
            continue
        entity = normalize_space(str(row.get("entity", "")))
        if not entity:
            continue
        key = entity.casefold()
        if key not in best or float(row.get("confidence", 0)) > float(best[key].get("confidence", 0)):
            best[key] = {**row, "entity": entity}
    return list(best.values())


def equivalent_variants(text: str, fine_type: str) -> list[str]:
    variants: list[str] = []
    if re.search(r"\d,\d", text):
        variants.append(re.sub(r"(?<=\d),(?=\d)", "", text))
    if re.search(r"\b\d+\.0\b", text):
        variants.append(re.sub(r"\b(\d+)\.0\b", r"\1", text))
    if fine_type == "PERCENT":
        if "%" in text:
            variants.append(text.replace("%", " percent"))
            variants.append(text.replace("%", " per cent"))
        elif re.search(r"\bpercent\b", text, re.IGNORECASE):
            variants.append(re.sub(r"\s*percent\b", "%", text, flags=re.IGNORECASE))
        elif re.search(r"\bper cent\b", text, re.IGNORECASE):
            variants.append(re.sub(r"\s*per cent\b", "%", text, flags=re.IGNORECASE))
    if fine_type == "MONEY":
        currency = {"$": " dollars", "£": " pounds", "€": " euros"}
        for symbol, word in currency.items():
            if symbol in text:
                variants.append(text.replace(symbol, "", 1).strip() + word)
    if fine_type == "DATE":
        for full, short in MONTHS.items():
            if re.search(rf"\b{full}\b", text, re.IGNORECASE):
                variants.append(re.sub(rf"\b{full}\b", short, text, count=1, flags=re.IGNORECASE))
            elif re.search(rf"\b{re.escape(short.rstrip('.'))}(?:\.)?", text, re.IGNORECASE):
                variants.append(re.sub(rf"\b{re.escape(short.rstrip('.'))}(?:\.)?", full, text, count=1, flags=re.IGNORECASE))
    return list(dict.fromkeys(normalize_space(value) for value in variants if normalize_space(value) != normalize_space(text)))


def format_number(raw: str, value: float) -> str:
    if "." in raw:
        decimals = len(raw.split(".")[-1])
        return f"{value:.{decimals}f}"
    integer = int(round(value))
    return f"{integer:,}" if "," in raw else str(integer)


def changed_variants(text: str, fine_type: str) -> list[str]:
    variants: list[str] = []
    match = re.search(r"[-+]?\d(?:[\d,]*\d)?(?:\.\d+)?", text)
    if match:
        raw = match.group(0)
        value = float(raw.replace(",", ""))
        if fine_type == "DATE" and value.is_integer() and 1500 <= value <= 2200:
            values = [value - 1, value + 1]
        elif fine_type == "DATE" and value.is_integer() and 1 <= value <= 31:
            values = [max(1, value - 1), min(31, value + 1)]
        elif fine_type in {"PERCENT", "ORDINAL"}:
            values = [max(0, value - 1), value + 1]
        elif value == 0:
            values = [1, 2]
        else:
            delta = max(1.0, round(abs(value) * 0.1, 2))
            values = [max(0, value - delta), value + delta]
        for new_value in values:
            replacement = format_number(raw, new_value)
            variants.append(text[: match.start()] + replacement + text[match.end() :])
    else:
        for word, replacement in NUMBER_WORD_NEIGHBORS.items():
            word_match = re.search(rf"\b{word}\b", text, re.IGNORECASE)
            if word_match:
                variants.append(text[: word_match.start()] + replacement + text[word_match.end() :])
                break
    if fine_type == "MONEY":
        for source, target in (("$", "€"), ("€", "$"), ("£", "$")):
            if source in text:
                variants.append(text.replace(source, target, 1))
    return list(dict.fromkeys(normalize_space(value) for value in variants if normalize_space(value) != normalize_space(text)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-files", type=Path, nargs="+", required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--mentions-per-entity", type=int, default=1)
    parser.add_argument("--equivalent-per-mention", type=int, default=1)
    parser.add_argument("--changed-per-mention", type=int, default=2)
    parser.add_argument("--context-chars", type=int, default=360)
    parser.add_argument("--max-entities", type=int, default=0)
    parser.add_argument("--seed", type=int, default=79)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    items = unique_numeric(load_tags(args.tag_files), args.min_confidence)
    rng.shuffle(items)
    if args.max_entities:
        items = items[: args.max_entities]
    conn = connect_ro(args.db)
    pairs: list[dict[str, Any]] = []
    skipped = Counter()
    for source in items:
        entity = str(source["entity"])
        fine_type = str(source["fine_type"])
        candidates = [
            (candidate, "numeric_equivalent_format", 1.0)
            for candidate in equivalent_variants(entity, fine_type)[: args.equivalent_per_mention]
        ] + [
            (candidate, "numeric_close_changed", 0.0)
            for candidate in changed_variants(entity, fine_type)[: args.changed_per_mention]
        ]
        if not candidates:
            skipped["no_variants"] += 1
            continue
        mention_rows = conn.execute(
            """
            select m.id as mention_id, m.text_id, m.mask_idx, t.row_idx, t.source_id,
                   t.masked_text, d.id as dataset_id, d.name as dataset_name,
                   d.split as dataset_split, d.domain
            from mentions m
            join texts t on t.id = m.text_id
            join datasets d on d.id = t.dataset_id
            where m.entity_id = ?
            limit 30
            """,
            (int(source["entity_id"]),),
        ).fetchall()
        mentions = [dict(row) for row in mention_rows]
        rng.shuffle(mentions)
        for mention in mentions[: args.mentions_per_entity]:
            for candidate, kind, expected in candidates:
                candidate_row = {
                    "entity_id": f"generated:{candidate.casefold()}",
                    "entity": candidate,
                    "fine_type": fine_type,
                    "coarse_group": "numeric",
                }
                pair = make_pair(
                    conn,
                    source,
                    candidate_row,
                    mention,
                    0.0,
                    kind,
                    f"nu{len(pairs) + 1:08d}",
                    args.context_chars,
                )
                pair["branch"] = "numeric_context_rules"
                pair["candidate_kind"] = kind
                pair["retrieval_bucket"] = None
                pair["expected_score"] = expected
                pair["context_policy"] = "short_window_generate_full_context_judge"
                pairs.append(pair)

    summary = {
        "source_entities": len(items),
        "pairs": len(pairs),
        "labels_expected": dict(Counter(str(row["expected_score"]) for row in pairs)),
        "by_type": dict(Counter(str(row["fine_type"]) for row in pairs)),
        "by_kind": dict(Counter(str(row["candidate_kind"]) for row in pairs)),
        "skipped": dict(skipped),
        "identity_pairs": sum(
            normalize_space(str(row["entity"])).casefold() == normalize_space(str(row["candidate"])).casefold()
            for row in pairs
        ),
    }
    write_jsonl(args.out, pairs)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
