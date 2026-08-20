#!/usr/bin/env python3
"""Build positive/negative substitution pairs from masked Arrow datasets."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.ipc as ipc


MASK_RE = re.compile(r"<mask>|\[MASK\]|\[ENT_MASK\]", re.IGNORECASE)


def iter_arrow_rows(dataset_dir: Path, max_rows: int | None = None):
    files = sorted(dataset_dir.glob("data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"No data-*.arrow files found in {dataset_dir}")

    seen = 0
    for path in files:
        with path.open("rb") as fh:
            try:
                reader = ipc.open_stream(fh)
            except Exception:
                fh.seek(0)
                reader = ipc.open_file(fh)

            for batch in reader:
                names = batch.schema.names
                columns = {name: batch.column(name).to_pylist() for name in names}
                for i in range(batch.num_rows):
                    yield {name: columns[name][i] for name in names}
                    seen += 1
                    if max_rows is not None and seen >= max_rows:
                        return


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def replace_nth_mask(text: str, mask_index: int, value: str) -> str:
    current = -1

    def repl(match: re.Match[str]) -> str:
        nonlocal current
        current += 1
        return value if current == mask_index else match.group(0)

    return MASK_RE.sub(repl, text, count=0)


def crop_around_entity(text: str, entity: str, window_chars: int) -> str:
    text = normalize_space(text)
    if len(text) <= window_chars * 2:
        return text
    idx = text.find(entity)
    if idx < 0:
        return text[: window_chars * 2]
    start = max(0, idx - window_chars)
    end = min(len(text), idx + len(entity) + window_chars)
    return text[start:end].strip()


def surface_variants(entity: str) -> list[str]:
    entity = normalize_space(entity)
    variants: list[str] = []
    if not entity:
        return variants

    us_variant = re.sub(r"\bU\.S\.", "US", entity)
    if us_variant != entity:
        variants.append(us_variant)

    prop_variant = re.sub(r"\bProp\.\s+(\d+)\b", r"Prop \1", entity)
    if prop_variant != entity:
        variants.append(prop_variant)

    suffix_variant = re.sub(r"\b(Jr|Sr)\.$", r"\1", entity)
    if suffix_variant != entity:
        variants.append(suffix_variant)

    no_commas = entity.replace(",", "")
    if no_commas != entity and "." not in entity:
        variants.append(no_commas)

    return list(dict.fromkeys(v for v in variants if v and v != entity))


def collect_entity_pool(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    pool: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        words = row.get("demasked_words") or []
        types = row.get("entity_types") or []
        for idx, entity in enumerate(words):
            entity = normalize_space(str(entity))
            if not entity:
                continue
            entity_type = str(types[idx]) if idx < len(types) and types[idx] else "UNTYPED"
            pool[entity_type].append(entity)
            pool["__ALL__"].append(entity)
    return {k: sorted(set(v)) for k, v in pool.items()}


def choose_negative(
    rng: random.Random,
    entity: str,
    entity_type: str,
    pool: dict[str, list[str]],
) -> str | None:
    entity_norm = normalize_space(entity).lower()
    candidates = pool.get(entity_type) or []
    candidates = [c for c in candidates if normalize_space(c).lower() != entity_norm]
    if len(candidates) < 1:
        candidates = [
            c for c in pool.get("__ALL__", []) if normalize_space(c).lower() != entity_norm
        ]
    if not candidates:
        return None
    return rng.choice(candidates)


def make_pairs(
    rows: list[dict[str, Any]],
    negatives_per_positive: int,
    window_chars: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool = collect_entity_pool(rows)
    pairs: list[dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        masked_text = row.get("masked_text") or ""
        words = row.get("demasked_words") or []
        types = row.get("entity_types") or []
        mask_count = len(MASK_RE.findall(masked_text))
        usable = min(mask_count, len(words))

        for mask_idx in range(usable):
            entity = normalize_space(str(words[mask_idx]))
            if not entity:
                continue

            entity_type = str(types[mask_idx]) if mask_idx < len(types) and types[mask_idx] else "UNTYPED"
            left = replace_nth_mask(masked_text, mask_idx, entity)
            left = crop_around_entity(left, entity, window_chars)

            variants = surface_variants(entity)
            positive_candidate = variants[0] if variants else entity
            right_pos = replace_nth_mask(masked_text, mask_idx, positive_candidate)
            right_pos = crop_around_entity(right_pos, positive_candidate, window_chars)
            pairs.append(
                {
                    "left": left,
                    "right": right_pos,
                    "label": 1,
                    "entity": entity,
                    "candidate": positive_candidate,
                    "entity_type": entity_type,
                    "row_idx": row_idx,
                    "mask_idx": mask_idx,
                    "pair_kind": "positive_surface" if variants else "positive_identity",
                }
            )

            for _ in range(negatives_per_positive):
                negative = choose_negative(rng, entity, entity_type, pool)
                if not negative:
                    continue
                right_neg = replace_nth_mask(masked_text, mask_idx, negative)
                right_neg = crop_around_entity(right_neg, negative, window_chars)
                pairs.append(
                    {
                        "left": left,
                        "right": right_neg,
                        "label": 0,
                        "entity": entity,
                        "candidate": negative,
                        "entity_type": entity_type,
                        "row_idx": row_idx,
                        "mask_idx": mask_idx,
                        "pair_kind": "negative_same_type"
                        if entity_type != "UNTYPED"
                        else "negative_random",
                    }
                )

    rng.shuffle(pairs)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--window-chars", type=int, default=450)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = list(iter_arrow_rows(args.dataset_dir, args.max_rows))
    pairs = make_pairs(rows, args.negatives_per_positive, args.window_chars, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    positives = sum(1 for p in pairs if p["label"] == 1)
    negatives = sum(1 for p in pairs if p["label"] == 0)
    print(f"rows={len(rows)} pairs={len(pairs)} positives={positives} negatives={negatives}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
