#!/usr/bin/env python3
"""Extract unique masked entities with compact context examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from augment_pairs import infer_entity_type, norm_key
from prepare_pairs import MASK_RE, crop_around_entity, iter_arrow_rows, normalize_space, replace_nth_mask


TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[$£€]?\d[\d,.%]*|[^\w\s]", re.UNICODE)


def domain_from_path(path: Path) -> str:
    raw = str(path).lower()
    if "billsum" in raw:
        return "legal"
    if "samsum" in raw:
        return "dialogue"
    if "multi_news" in raw:
        return "multi_news"
    if "cnn_dailymail" in raw:
        return "news"
    if "mars_test_200" in raw:
        return "news"
    if "xsum" in raw:
        return "news_short"
    return "unknown"


def stable_id(entity: str) -> str:
    return hashlib.sha1(norm_key(entity).encode("utf-8")).hexdigest()[:16]


def local_word_window(text: str, entity: str, words_each_side: int) -> str:
    text = normalize_space(text)
    idx = text.find(entity)
    if idx < 0:
        return crop_around_entity(text, entity, 120)

    snippet_start = max(0, idx - 220)
    snippet_end = min(len(text), idx + len(entity) + 220)
    snippet = text[snippet_start:snippet_end]
    relative_idx = idx - snippet_start

    spans = [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(snippet)]
    entity_token_idx = None
    for i, (_, start, end) in enumerate(spans):
        if start <= relative_idx < end or relative_idx <= start < relative_idx + len(entity):
            entity_token_idx = i
            break
    if entity_token_idx is None:
        return crop_around_entity(text, entity, 120)

    start_i = max(0, entity_token_idx - words_each_side)
    end_i = min(len(spans), entity_token_idx + words_each_side + 1)
    return " ".join(tok for tok, _, _ in spans[start_i:end_i])


def heuristic_group(entity: str, inferred_type: str) -> str:
    numeric_types = {"DATE", "TIME", "CARDINAL", "MONEY", "PERCENT", "QUANTITY"}
    proper_types = {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "FAC",
        "NORP",
        "EVENT",
        "LAW",
        "PRODUCT",
        "WORK_OF_ART",
    }
    if inferred_type in numeric_types:
        return "numeric"
    if inferred_type in proper_types:
        return "proper_name"

    text = normalize_space(entity)
    if len(text) <= 1:
        return "junk"
    if re.fullmatch(r"[\W_]+", text):
        return "junk"
    if text.isupper() and len(text) > 1:
        return "proper_name"
    if any(part[:1].isupper() for part in text.split()) and not text.islower():
        return "proper_name"
    if len(text.split()) >= 2:
        return "domain_term"
    return "common_entity"


def default_context_policy(group: str) -> str:
    if group == "numeric":
        return "short_window"
    if group == "proper_name":
        return "full_context"
    if group in {"common_entity", "domain_term"}:
        return "no_context_embedding"
    if group == "junk":
        return "drop"
    return "agent_review"


def dataset_label(path: Path) -> str:
    parts = path.parts
    for part in reversed(parts):
        if part.endswith("_with_mask") or part == "mars_test_200_split":
            return part
    return path.name


def collect_inventory(
    dataset_dirs: list[Path],
    max_rows_per_dataset: int | None,
    max_samples_per_entity: int,
    short_context_words: int,
    full_context_chars: int,
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}

    for dataset_dir in dataset_dirs:
        domain = domain_from_path(dataset_dir)
        dataset = dataset_label(dataset_dir)
        for row_idx, row in enumerate(iter_arrow_rows(dataset_dir, max_rows_per_dataset)):
            masked_text = row.get("masked_text") or ""
            words = row.get("demasked_words") or []
            types = row.get("entity_types") or []
            usable = min(len(MASK_RE.findall(masked_text)), len(words))

            for mask_idx in range(usable):
                entity = normalize_space(str(words[mask_idx]))
                if not entity:
                    continue

                given = str(types[mask_idx]) if mask_idx < len(types) and types[mask_idx] else None
                inferred_type = infer_entity_type(entity, given)
                group = heuristic_group(entity, inferred_type)
                entity_id = stable_id(entity)
                item = inventory.setdefault(
                    entity_id,
                    {
                        "entity_id": entity_id,
                        "entity": entity,
                        "count": 0,
                        "observed_types": Counter(),
                        "domains": Counter(),
                        "datasets": Counter(),
                        "heuristic_group": group,
                        "heuristic_context_policy": default_context_policy(group),
                        "examples": [],
                    },
                )
                item["count"] += 1
                item["observed_types"][inferred_type] += 1
                item["domains"][domain] += 1
                item["datasets"][dataset] += 1

                if len(item["examples"]) < max_samples_per_entity:
                    filled = replace_nth_mask(masked_text, mask_idx, entity)
                    item["examples"].append(
                        {
                            "dataset": dataset,
                            "domain": domain,
                            "row_idx": row_idx,
                            "mask_idx": mask_idx,
                            "short_context": local_word_window(
                                filled,
                                entity,
                                short_context_words,
                            ),
                            "full_context": crop_around_entity(
                                filled,
                                entity,
                                full_context_chars,
                            ),
                        }
                    )

    return inventory


def serializable_item(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out["observed_types"] = dict(item["observed_types"].most_common())
    out["domains"] = dict(item["domains"].most_common())
    out["datasets"] = dict(item["datasets"].most_common())
    return out


def write_summary(path: Path, items: list[dict[str, Any]]) -> None:
    by_group = Counter(x["heuristic_group"] for x in items)
    by_policy = Counter(x["heuristic_context_policy"] for x in items)
    by_type = Counter()
    for item in items:
        if item["observed_types"]:
            by_type[next(iter(item["observed_types"].keys()))] += 1
    summary = {
        "unique_entities": len(items),
        "total_mentions": sum(x["count"] for x in items),
        "by_heuristic_group": dict(by_group.most_common()),
        "by_context_policy": dict(by_policy.most_common()),
        "by_primary_type": dict(by_type.most_common()),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--max-rows-per-dataset", type=int, default=200)
    parser.add_argument("--max-samples-per-entity", type=int, default=3)
    parser.add_argument("--short-context-words", type=int, default=2)
    parser.add_argument("--full-context-chars", type=int, default=700)
    args = parser.parse_args()

    inventory = collect_inventory(
        args.dataset_dir,
        args.max_rows_per_dataset,
        args.max_samples_per_entity,
        args.short_context_words,
        args.full_context_chars,
    )
    items = [serializable_item(x) for x in inventory.values()]
    items.sort(key=lambda x: (-x["count"], x["entity"].lower()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        write_summary(args.summary_out, items)

    print(f"unique_entities={len(items)} mentions={sum(x['count'] for x in items)}")
    print(f"wrote {args.out}")
    if args.summary_out:
        print(f"wrote {args.summary_out}")


if __name__ == "__main__":
    main()
