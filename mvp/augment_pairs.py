#!/usr/bin/env python3
"""Generate stronger substitution examples for the contextual scorer.

The script creates two outputs:
- labeled training pairs: strict positives and safe negatives;
- optional review tasks: context-sensitive candidates that should be labeled by an agent/LLM/human.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from prepare_pairs import (
    MASK_RE,
    crop_around_entity,
    iter_arrow_rows,
    normalize_space,
    replace_nth_mask,
    surface_variants,
)


MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


def norm_key(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[^\w%$£€]+", "", text)
    return text


def infer_entity_type(entity: str, given: str | None = None) -> str:
    if given and given != "UNKNOWN":
        return given

    text = normalize_space(entity)
    low = text.lower()
    if re.search(r"[$£€]\s*\d|\b(dollars?|euros?|pounds?)\b", low):
        return "MONEY"
    if "%" in text or re.search(r"\bpercent\b", low):
        return "PERCENT"
    if re.search(r"\b(19|20)\d{2}\b", text) or any(m in low for m in MONTHS):
        return "DATE"
    if low in {"today", "yesterday", "tomorrow", "tonight", "last month", "next month"}:
        return "DATE"
    if re.fullmatch(r"[\d,.\s]+", text):
        return "CARDINAL"
    return given or "UNTYPED"


def load_aliases(path: Path) -> dict[str, dict[str, list[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, dict[str, list[str]]] = {}
    for group in data.get("groups", []):
        strict = list(dict.fromkeys(group.get("strict_aliases", [])))
        review = list(dict.fromkeys(group.get("review_aliases", [])))
        all_names = set(strict) | set(review) | {group.get("canonical", "")}
        for name in all_names:
            if not name:
                continue
            index[norm_key(name)] = {"strict": strict, "review": review}
    return index


def alias_candidates(entity: str, alias_index: dict[str, dict[str, list[str]]]) -> tuple[list[str], list[str]]:
    item = alias_index.get(norm_key(entity))
    if not item:
        return [], []
    entity_norm = norm_key(entity)
    strict = [x for x in item["strict"] if norm_key(x) != entity_norm]
    review = [x for x in item["review"] if norm_key(x) != entity_norm]
    return strict, review


def perturb_number_like(entity: str) -> list[str]:
    text = normalize_space(entity)
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return []

    raw = match.group(0)
    value = float(raw.replace(",", ""))
    has_decimal = "." in raw
    month_day = bool(
        re.search(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+"
            + re.escape(raw)
            + r"\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if month_day and value.is_integer():
        values = [max(1, int(value) - 1), min(28, int(value) + 1)]
    elif value == 0:
        values = [1, 10]
    elif 1900 <= value <= 2100 and float(value).is_integer():
        values = [int(value) - 1, int(value) + 1]
    elif not has_decimal and float(value).is_integer():
        delta = max(1, round(abs(value) * 0.1))
        values = [int(value) + delta, max(0, int(value) - delta)]
    else:
        delta = max(1, abs(value) * 0.1)
        values = [value + delta, max(0, value - delta)]

    variants = []
    for new_value in values:
        if not has_decimal or float(new_value).is_integer():
            replacement = f"{int(new_value):,}" if "," in raw else str(int(new_value))
        else:
            replacement = f"{new_value:.2f}"
        variants.append(text[: match.start()] + replacement + text[match.end() :])

    if "$" in text:
        variants.append(text.replace("$", "€", 1))
    elif "€" in text:
        variants.append(text.replace("€", "$", 1))
    elif "£" in text:
        variants.append(text.replace("£", "$", 1))
    return list(dict.fromkeys(v for v in variants if v != text))


def perturb_month(entity: str) -> list[str]:
    text = normalize_space(entity)
    low = text.lower()
    variants = []
    for idx, month in enumerate(MONTHS):
        if month in low:
            replacement = MONTHS[(idx + 1) % len(MONTHS)].capitalize()
            variants.append(re.sub(month, replacement, text, flags=re.IGNORECASE))
            break
    return variants


def collect_pool(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    pool: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        words = row.get("demasked_words") or []
        types = row.get("entity_types") or []
        for idx, raw in enumerate(words):
            entity = normalize_space(str(raw))
            if not entity:
                continue
            given = str(types[idx]) if idx < len(types) and types[idx] else None
            entity_type = infer_entity_type(entity, given)
            pool[entity_type].add(entity)
            pool["__ALL__"].add(entity)
    return {k: sorted(v) for k, v in pool.items()}


def choose_same_type_negative(
    rng: random.Random,
    entity: str,
    entity_type: str,
    pool: dict[str, list[str]],
) -> str | None:
    entity_norm = norm_key(entity)
    candidates = [x for x in pool.get(entity_type, []) if norm_key(x) != entity_norm]
    if not candidates:
        candidates = [x for x in pool.get("__ALL__", []) if norm_key(x) != entity_norm]
    return rng.choice(candidates) if candidates else None


def make_pair(
    masked_text: str,
    row_idx: int,
    mask_idx: int,
    entity: str,
    candidate: str,
    label: int,
    entity_type: str,
    pair_kind: str,
    annotation_source: str,
    rationale: str,
    window_chars: int,
) -> dict[str, Any]:
    left = replace_nth_mask(masked_text, mask_idx, entity)
    right = replace_nth_mask(masked_text, mask_idx, candidate)
    return {
        "left": crop_around_entity(left, entity, window_chars),
        "right": crop_around_entity(right, candidate, window_chars),
        "label": label,
        "entity": entity,
        "candidate": candidate,
        "entity_type": entity_type,
        "row_idx": row_idx,
        "mask_idx": mask_idx,
        "pair_kind": pair_kind,
        "annotation_source": annotation_source,
        "rationale": rationale,
    }


def make_review_task(
    masked_text: str,
    row_idx: int,
    mask_idx: int,
    entity: str,
    candidate: str,
    entity_type: str,
    pair_kind: str,
    window_chars: int,
) -> dict[str, Any]:
    left = replace_nth_mask(masked_text, mask_idx, entity)
    right = replace_nth_mask(masked_text, mask_idx, candidate)
    return {
        "left": crop_around_entity(left, entity, window_chars),
        "right": crop_around_entity(right, candidate, window_chars),
        "entity": entity,
        "candidate": candidate,
        "entity_type": entity_type,
        "row_idx": row_idx,
        "mask_idx": mask_idx,
        "pair_kind": pair_kind,
        "annotation_source": "agent_review_required",
        "question": "Does replacing entity with candidate preserve the meaning/fact in this context? Return label 1, 0.5, or 0 and a short reason.",
    }


def generate_examples(
    rows: list[dict[str, Any]],
    alias_index: dict[str, dict[str, list[str]]],
    negatives_per_entity: int,
    review_candidates_per_entity: int,
    window_chars: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    pool = collect_pool(rows)
    labeled: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    seen_labeled: set[tuple[int, int, str, int]] = set()
    seen_review: set[tuple[int, int, str]] = set()

    def add_labeled(pair: dict[str, Any]) -> None:
        key = (pair["row_idx"], pair["mask_idx"], norm_key(pair["candidate"]), pair["label"])
        if key not in seen_labeled:
            seen_labeled.add(key)
            labeled.append(pair)

    def add_review(task: dict[str, Any]) -> None:
        key = (task["row_idx"], task["mask_idx"], norm_key(task["candidate"]))
        if key not in seen_review:
            seen_review.add(key)
            review.append(task)

    for row_idx, row in enumerate(rows):
        masked_text = row.get("masked_text") or ""
        words = row.get("demasked_words") or []
        types = row.get("entity_types") or []
        usable = min(len(MASK_RE.findall(masked_text)), len(words))

        for mask_idx in range(usable):
            entity = normalize_space(str(words[mask_idx]))
            if not entity:
                continue
            given = str(types[mask_idx]) if mask_idx < len(types) and types[mask_idx] else None
            entity_type = infer_entity_type(entity, given)

            add_labeled(
                make_pair(
                    masked_text,
                    row_idx,
                    mask_idx,
                    entity,
                    entity,
                    1,
                    entity_type,
                    "positive_identity",
                    "rule",
                    "The original entity is always a valid replacement for itself.",
                    window_chars,
                )
            )

            for candidate in surface_variants(entity):
                add_labeled(
                    make_pair(
                        masked_text,
                        row_idx,
                        mask_idx,
                        entity,
                        candidate,
                        1,
                        entity_type,
                        "positive_surface",
                        "rule",
                        "Surface-normalized variant of the same entity.",
                        window_chars,
                    )
                )

            strict_aliases, review_aliases = alias_candidates(entity, alias_index)
            for candidate in strict_aliases[:4]:
                add_labeled(
                    make_pair(
                        masked_text,
                        row_idx,
                        mask_idx,
                        entity,
                        candidate,
                        1,
                        entity_type,
                        "positive_alias",
                        "alias_dict",
                        "Strict alias group from local aliases.json.",
                        window_chars,
                    )
                )
            for candidate in review_aliases[:3]:
                add_review(
                    make_review_task(
                        masked_text,
                        row_idx,
                        mask_idx,
                        entity,
                        candidate,
                        entity_type,
                        "context_sensitive_alias_candidate",
                        window_chars,
                    )
                )

            if entity_type in {"DATE", "CARDINAL", "MONEY", "PERCENT", "QUANTITY"}:
                candidates = perturb_number_like(entity) + perturb_month(entity)
                for candidate in candidates[:3]:
                    add_labeled(
                        make_pair(
                            masked_text,
                            row_idx,
                            mask_idx,
                            entity,
                            candidate,
                            0,
                            entity_type,
                            "negative_numeric_or_date_perturbation",
                            "rule",
                            "Small date/number/currency change usually changes the fact.",
                            window_chars,
                        )
                    )

            for _ in range(negatives_per_entity):
                candidate = choose_same_type_negative(rng, entity, entity_type, pool)
                if candidate:
                    add_labeled(
                        make_pair(
                            masked_text,
                            row_idx,
                            mask_idx,
                            entity,
                            candidate,
                            0,
                            entity_type,
                            "negative_same_type",
                            "synthetic",
                            "Different entity of the same inferred type.",
                            window_chars,
                        )
                    )

            for _ in range(review_candidates_per_entity):
                candidate = choose_same_type_negative(rng, entity, entity_type, pool)
                if candidate:
                    add_review(
                        make_review_task(
                            masked_text,
                            row_idx,
                            mask_idx,
                            entity,
                            candidate,
                            entity_type,
                            "agent_hard_same_type_candidate",
                            window_chars,
                        )
                    )

    rng.shuffle(labeled)
    rng.shuffle(review)
    return labeled, review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--agent-review-out", type=Path)
    parser.add_argument("--aliases", type=Path, default=Path("mvp/resources/aliases.json"))
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--negatives-per-entity", type=int, default=3)
    parser.add_argument("--review-candidates-per-entity", type=int, default=1)
    parser.add_argument("--window-chars", type=int, default=450)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = list(iter_arrow_rows(args.dataset_dir, args.max_rows))
    alias_index = load_aliases(args.aliases)
    labeled, review = generate_examples(
        rows,
        alias_index,
        args.negatives_per_entity,
        args.review_candidates_per_entity,
        args.window_chars,
        args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for item in labeled:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    if args.agent_review_out:
        args.agent_review_out.parent.mkdir(parents=True, exist_ok=True)
        with args.agent_review_out.open("w", encoding="utf-8") as fh:
            for item in review:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    positives = sum(1 for x in labeled if x["label"] == 1)
    negatives = sum(1 for x in labeled if x["label"] == 0)
    print(f"rows={len(rows)} labeled={len(labeled)} positives={positives} negatives={negatives}")
    print(f"review_tasks={len(review)}")
    print(f"wrote {args.out}")
    if args.agent_review_out:
        print(f"wrote {args.agent_review_out}")


if __name__ == "__main__":
    main()
