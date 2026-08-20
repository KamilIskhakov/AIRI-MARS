#!/usr/bin/env python3
"""Generate context-fitting one-token common-noun candidates with ModernBERT MLM."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from audit_entity_tags_and_probe import connect_ro, load_tags, one_mention
from generate_common_embedding_pairs import (
    TARGET_MARKER,
    clean_surface,
    has_wordnet_noun,
    simple_inflection_pair,
    unique_items,
    write_jsonl,
)
from prepare_pairs import normalize_space


def marked_context(conn: Any, mention: dict[str, Any], chars: int) -> str | None:
    rows = conn.execute(
        """
        select m.mask_idx, e.surface
        from mentions m
        join entities e on e.id = m.entity_id
        where m.text_id = ?
        order by m.mask_idx
        """,
        (int(mention["text_id"]),),
    ).fetchall()
    surfaces = {int(row["mask_idx"]): str(row["surface"]) for row in rows}
    target_idx = int(mention["mask_idx"])
    parts = str(mention["masked_text"]).split("<mask>")
    if len(parts) - 1 != len(surfaces) or target_idx not in surfaces:
        return None
    rebuilt = [parts[0]]
    for idx, suffix in enumerate(parts[1:]):
        rebuilt.append(TARGET_MARKER if idx == target_idx else surfaces[idx])
        rebuilt.append(suffix)
    text = "".join(rebuilt)
    position = text.index(TARGET_MARKER)
    lo = max(0, position - chars)
    hi = min(len(text), position + len(TARGET_MARKER) + chars)
    return normalize_space(text[lo:hi])


def source_items(paths: list[Path], confidence: float) -> list[dict[str, Any]]:
    rows = unique_items(load_tags(paths))
    return sorted(
        (
            row for row in rows
            if row.get("coarse_group") == "common_entity"
            and row.get("fine_type") == "COMMON_NOUN"
            and float(row.get("confidence", 0)) >= confidence
            and clean_surface(str(row["entity"]), strict_lexical=True)
            and has_wordnet_noun(str(row["entity"]))
        ),
        key=lambda row: str(row["entity"]).casefold(),
    )


def valid_candidate(candidate: str, source: str) -> bool:
    candidate = candidate.strip()
    return (
        candidate == candidate.casefold()
        and clean_surface(candidate, strict_lexical=True)
        and has_wordnet_noun(candidate)
        and candidate.casefold() != source.casefold()
        and not simple_inflection_pair(source, candidate)
        and bool(re.fullmatch(r"[a-z]+(?:['-][a-z]+)*", candidate))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-files", type=Path, nargs="+", required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--candidates-per-mention", type=int, default=5)
    parser.add_argument("--mentions-per-entity", type=int, default=1)
    parser.add_argument("--min-log-prob", type=float, default=-12.0)
    parser.add_argument("--max-relative-log-prob", type=float, default=5.0)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--context-chars", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-entities", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    items = source_items(args.tag_files, args.min_confidence)
    if args.max_entities:
        rng.shuffle(items)
        items = items[: args.max_entities]
    conn = connect_ro(args.db)
    device = (
        torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
        else torch.device("mps") if args.device == "auto" and torch.backends.mps.is_available()
        else torch.device("cpu") if args.device == "auto"
        else torch.device(args.device)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=Path(args.model).exists())
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, local_files_only=Path(args.model).exists()
    ).to(device)
    model.eval()

    pairs: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    with torch.inference_mode():
        for source in items:
            source_id = str(source["entity_id"])
            mentions = conn.execute(
                """
                select m.id as mention_id, m.text_id, m.mask_idx, t.row_idx, t.source_id,
                       t.masked_text, d.id as dataset_id, d.name as dataset_name,
                       d.split as dataset_split, d.domain
                from mentions m
                join texts t on t.id = m.text_id
                join datasets d on d.id = t.dataset_id
                where m.entity_id = ?
                limit 40
                """,
                (int(source_id),),
            ).fetchall()
            mention_items = [dict(row) for row in mentions]
            rng.shuffle(mention_items)
            for mention in mention_items[: args.mentions_per_entity]:
                marked = marked_context(conn, mention, args.context_chars)
                if not marked:
                    stats["skip_reconstruction"] += 1
                    continue
                masked = marked.replace(TARGET_MARKER, tokenizer.mask_token, 1)
                encoded = tokenizer(
                    masked,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                )
                mask_positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=False)
                if len(mask_positions) != 1:
                    stats["skip_mask_alignment"] += 1
                    continue
                mask_idx = int(mask_positions[0, 0])
                output = model(**{key: value.to(device) for key, value in encoded.items()})
                log_probs = torch.log_softmax(output.logits[0, mask_idx], dim=-1)
                values, token_ids = torch.topk(log_probs, min(args.top_k, log_probs.shape[-1]))
                top_log_prob = float(values[0].item())
                accepted = 0
                seen: set[str] = set()
                for rank, (value, token_id) in enumerate(zip(values.tolist(), token_ids.tolist()), start=1):
                    candidate = tokenizer.decode([token_id], clean_up_tokenization_spaces=False).strip()
                    key = candidate.casefold()
                    if key in seen or not valid_candidate(candidate, str(source["entity"])):
                        continue
                    seen.add(key)
                    relative = top_log_prob - float(value)
                    if value < args.min_log_prob or relative > args.max_relative_log_prob:
                        continue
                    original_context = marked.replace(TARGET_MARKER, str(source["entity"]), 1)
                    candidate_context = marked.replace(TARGET_MARKER, candidate, 1)
                    pairs.append(
                        {
                            "pair_id": f"cm{len(pairs) + 1:08d}",
                            "branch": "common_mlm_generation",
                            "candidate_kind": "common_mlm_context_fit",
                            "expected_score": None,
                            "entity_id": source_id,
                            "candidate_entity_id": None,
                            "entity": source["entity"],
                            "candidate": candidate,
                            "fine_type": "COMMON_NOUN",
                            "candidate_fine_type": "COMMON_NOUN",
                            "coarse_group": "common_entity",
                            "candidate_coarse_group": "common_entity",
                            "context_policy": "full_context_judge",
                            "mlm_log_prob": round(float(value), 6),
                            "mlm_probability": round(math.exp(float(value)), 10),
                            "mlm_rank": rank,
                            "mlm_relative_log_prob": round(relative, 6),
                            "mlm_model": args.model,
                            "dataset": f"{mention['dataset_name']}/{mention['dataset_split']}",
                            "dataset_name": mention["dataset_name"],
                            "dataset_split": mention["dataset_split"],
                            "dataset_id": mention["dataset_id"],
                            "text_id": mention["text_id"],
                            "mention_id": mention["mention_id"],
                            "row_idx": mention["row_idx"],
                            "source_id": mention["source_id"],
                            "domain": mention["domain"],
                            "mask_idx": mention["mask_idx"],
                            "original_context": original_context,
                            "candidate_context": candidate_context,
                        }
                    )
                    accepted += 1
                    if accepted >= args.candidates_per_mention:
                        break
                stats["mentions_processed"] += 1
                stats["pairs"] += accepted

    summary = {
        "source_entities": len(items),
        "pairs": len(pairs),
        "unique_entities_with_pairs": len({row["entity_id"] for row in pairs}),
        "parameters": {
            "top_k": args.top_k,
            "candidates_per_mention": args.candidates_per_mention,
            "mentions_per_entity": args.mentions_per_entity,
            "min_log_prob": args.min_log_prob,
            "max_relative_log_prob": args.max_relative_log_prob,
        },
        "stats": dict(stats),
        "mlm_log_prob": {
            "min": min((row["mlm_log_prob"] for row in pairs), default=None),
            "max": max((row["mlm_log_prob"] for row in pairs), default=None),
        },
    }
    write_jsonl(args.out, pairs)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
