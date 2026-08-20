#!/usr/bin/env python3
"""Small end-to-end probe for entity substitution strategies on local data."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from audit_entity_tags_and_probe import (
    connect_ro,
    context_crop,
    encode_with_local_modernbert,
    load_tags,
    one_mention,
    perturb_number_like,
    short_window,
)
from prepare_pairs import crop_around_entity, normalize_space, replace_nth_mask, surface_variants
from tag_entities_mistral import load_env_file


TEMPORAL_OR_MEASURE_RE = re.compile(
    r"\d|"
    r"\b(year|years|month|months|week|weeks|day|days|hour|hours|minute|minutes|"
    r"second|seconds|mile|miles|inch|inches|meter|meters|metre|metres|"
    r"century|centuries|decade|decades|season|seasons)\b",
    re.IGNORECASE,
)


class Judgment(BaseModel):
    pair_id: str
    label: Literal["preserved", "changed", "uncertain"]
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=240)


class JudgmentBatch(BaseModel):
    items: list[Judgment]


JUDGE_PROMPT = """You judge contextual entity substitutions.

Given an original context and a substituted context, decide whether replacing the entity with the candidate preserves the same factual meaning in this context.

Use a strict bidirectional test: label preserved only if the original statement entails
the substituted statement and the substituted statement entails the original statement
in the same situation. If either direction loses or adds factual specificity, label changed.

Labels:
- preserved: same entity/fact, true alias, harmless spelling/surface variant.
- changed: materially changes the person, organization, place, number, date, quantity, product, or claim.
- uncertain: context is insufficient or the answer genuinely cannot be determined.

For common nouns and domain terms, relatedness is not equivalence. Hypernyms, hyponyms,
antonyms, neighboring concepts, and singular/plural changes are "changed" unless the two
forms are genuinely interchangeable in this exact sentence without changing reference,
cardinality, grammatical role, or the stated fact. A spelling variant or true contextual
synonym may be "preserved".

The same rule applies to named entities. A subset/superset, part/whole, demonym/place,
institution/location, event/edition, or broad name/more specific name pair is "changed"
unless the context proves that both expressions denote exactly the same referent. Do not
infer equivalence merely from topical relatedness or plausible world knowledge. Check that
possessives, articles, number, and surrounding grammar remain valid after replacement.
Be careful with confusable quantitative terms: for example, biannual (twice per year) and
biennial (once every two years) are changed, not spelling variants.

Be strict. For numbers, dates, quantities, and proper names, small changes usually mean
"changed". The score is confidence in the chosen label, not a semantic-similarity score:
use values near 1 only when the label is clear and lower values when judgment is uncertain.
Return only the structured JSON.
"""


def load_mistral_client():
    try:
        from mistralai import Mistral
    except Exception:
        from mistralai.client import Mistral  # type: ignore

    load_env_file(Path("mvp/.env"))
    load_env_file(Path(".env"))
    api_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY or OPENAI_API_KEY")
    server_url = os.environ.get("MISTRAL_BASE_URL")
    if server_url:
        return Mistral(api_key=api_key, server_url=server_url)
    return Mistral(api_key=api_key)


def norm_key(text: str) -> str:
    return re.sub(r"\W+", "", normalize_space(text).lower())


def is_clean_common_surface(text: str) -> bool:
    text = normalize_space(text)
    if len(text) < 3 or len(text) > 40:
        return False
    if TEMPORAL_OR_MEASURE_RE.search(text):
        return False
    if text.lower() in {"more", "less", "other", "same", "this", "that"}:
        return False
    return bool(re.search(r"[A-Za-zА-Яа-я]", text))


def fill_contexts(masked_text: str, mask_idx: int, entity: str, candidate: str, window_chars: int) -> tuple[str, str]:
    original = replace_nth_mask(masked_text, mask_idx, entity)
    substituted = replace_nth_mask(masked_text, mask_idx, candidate)
    return crop_around_entity(original, entity, window_chars), crop_around_entity(
        substituted,
        candidate,
        window_chars,
    )


def make_pair(
    tag: dict[str, Any],
    mention: dict[str, Any],
    candidate: str,
    branch: str,
    candidate_kind: str,
    expected_score: float | None,
    window_chars: int,
) -> dict[str, Any]:
    mask_idx = int(mention["mask_idx"])
    original_context, candidate_context = fill_contexts(
        mention["masked_text"],
        mask_idx,
        tag["entity"],
        candidate,
        window_chars,
    )
    return {
        "pair_id": "",
        "branch": branch,
        "candidate_kind": candidate_kind,
        "expected_score": expected_score,
        "entity_id": str(tag["entity_id"]),
        "entity": tag["entity"],
        "candidate": candidate,
        "fine_type": tag.get("fine_type"),
        "coarse_group": tag.get("coarse_group"),
        "context_policy": tag.get("context_policy"),
        "provider": tag.get("provider"),
        "dataset": f"{mention['dataset_name']}/{mention['dataset_split']}",
        "domain": mention["domain"],
        "short_2w": short_window(mention["masked_text"], mask_idx, tag["entity"], 2),
        "original_context": original_context,
        "candidate_context": candidate_context,
    }


def candidate_pool_by_fine_type(tags: list[dict[str, Any]]) -> dict[str, list[str]]:
    pool: dict[str, set[str]] = defaultdict(set)
    for tag in tags:
        if tag.get("coarse_group") != "proper_name":
            continue
        surface = normalize_space(str(tag.get("entity", "")))
        if len(surface) < 3:
            continue
        pool[str(tag.get("fine_type", "UNKNOWN"))].add(surface)
    return {k: sorted(v) for k, v in pool.items()}


def add_ids(pairs: list[dict[str, Any]]) -> None:
    for idx, pair in enumerate(pairs, start=1):
        pair["pair_id"] = f"p{idx:04d}"


def generate_common_embedding_pairs(
    tags: list[dict[str, Any]],
    conn: sqlite3.Connection,
    rng: random.Random,
    model_dir: Path,
    max_surfaces: int,
    high_n: int,
    mid_n: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    by_surface: dict[str, dict[str, Any]] = {}
    for tag in tags:
        if tag.get("coarse_group") != "common_entity":
            continue
        surface = normalize_space(str(tag.get("entity", "")))
        if not is_clean_common_surface(surface):
            continue
        by_surface.setdefault(norm_key(surface), {**tag, "entity": surface})

    items = list(by_surface.values())
    rng.shuffle(items)
    items = items[:max_surfaces]
    if len(items) < 4:
        return []

    surfaces = [item["entity"] for item in items]
    emb = encode_with_local_modernbert(surfaces, model_dir, batch_size)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)

    high: list[tuple[int, int, float]] = []
    mid: list[tuple[int, int, float]] = []
    for i in range(len(items)):
        order = np.argsort(-sim[i])
        for j in order[:8]:
            score = float(sim[i, j])
            if score >= 0.95:
                high.append((i, int(j), score))
                break
            if 0.80 <= score < 0.90:
                mid.append((i, int(j), score))
                break

    rng.shuffle(high)
    rng.shuffle(mid)
    pairs: list[dict[str, Any]] = []
    for bucket, rows, n, expected in [
        ("common_high_cosine", high, high_n, None),
        ("common_mid_cosine", mid, mid_n, 0.0),
    ]:
        for i, j, score in rows[:n]:
            tag = items[i]
            mention = one_mention(conn, str(tag["entity_id"]), rng)
            if not mention:
                continue
            pair = make_pair(
                tag,
                mention,
                items[j]["entity"],
                "common",
                bucket,
                expected,
                220,
            )
            pair["cosine"] = round(score, 4)
            pairs.append(pair)
    return pairs


def generate_pairs(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    tags = load_tags(args.tag_files)
    conn = connect_ro(args.db)

    mistral_tags = [t for t in tags if "mistral" in t["provider"]]
    primary = tags if args.tag_selection == "all" else (mistral_tags or tags)

    pairs: list[dict[str, Any]] = []

    controls = [t for t in primary if t.get("coarse_group") in {"numeric", "proper_name", "common_entity"}]
    rng.shuffle(controls)
    for tag in controls[: args.identity_controls]:
        mention = one_mention(conn, str(tag["entity_id"]), rng)
        if mention:
            pairs.append(make_pair(tag, mention, tag["entity"], "control", "identity", 1.0, 220))

    numeric = [t for t in primary if t.get("coarse_group") == "numeric"]
    rng.shuffle(numeric)
    for tag in numeric:
        candidates = perturb_number_like(tag["entity"])
        if not candidates:
            continue
        mention = one_mention(conn, str(tag["entity_id"]), rng)
        if not mention:
            continue
        pairs.append(
            make_pair(tag, mention, candidates[0], "numeric", "numeric_perturb", 0.0, 100)
        )
        if sum(1 for p in pairs if p["candidate_kind"] == "numeric_perturb") >= args.numeric_n:
            break

    proper = [t for t in primary if t.get("coarse_group") == "proper_name"]
    proper_pool = candidate_pool_by_fine_type(primary)
    rng.shuffle(proper)
    for tag in proper:
        candidates = [
            c
            for c in proper_pool.get(str(tag.get("fine_type", "UNKNOWN")), [])
            if norm_key(c) != norm_key(tag["entity"])
        ]
        if not candidates:
            continue
        mention = one_mention(conn, str(tag["entity_id"]), rng)
        if not mention:
            continue
        pairs.append(
            make_pair(
                tag,
                mention,
                rng.choice(candidates),
                "proper_name",
                "same_fine_type_random",
                0.0,
                320,
            )
        )
        if sum(1 for p in pairs if p["candidate_kind"] == "same_fine_type_random") >= args.proper_n:
            break

    surface_variant_count = 0
    rng.shuffle(proper)
    for tag in proper:
        variants = surface_variants(tag["entity"])
        if not variants:
            continue
        mention = one_mention(conn, str(tag["entity_id"]), rng)
        if not mention:
            continue
        pairs.append(
            make_pair(
                tag,
                mention,
                variants[0],
                "proper_name",
                "surface_variant",
                1.0,
                260,
            )
        )
        surface_variant_count += 1
        if surface_variant_count >= args.surface_variant_n:
            break

    pairs.extend(
        generate_common_embedding_pairs(
            primary,
            conn,
            rng,
            args.embedding_model,
            args.common_pool,
            args.common_high_n,
            args.common_mid_n,
            args.embedding_batch_size,
        )
    )

    add_ids(pairs)
    return pairs


def judge_batch(client, model: str, batch: list[dict[str, Any]], max_tokens: int) -> JudgmentBatch:
    payload = [
        {
            "pair_id": p["pair_id"],
            "branch": p["branch"],
            "entity": p["entity"],
            "candidate": p["candidate"],
            "original_context": p.get("original_context") or p.get("left"),
            "candidate_context": p.get("candidate_context") or p.get("right"),
        }
        for p in batch
    ]
    response = client.chat.parse(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": "Judge these substitutions. Keep pair_id unchanged.\n"
                + json.dumps(payload, ensure_ascii=False),
            },
        ],
        response_format=JudgmentBatch,
        temperature=0,
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        return parsed
    return JudgmentBatch.model_validate_json(message.content)


def run_judge(args: argparse.Namespace, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    client = load_mistral_client()
    model = args.judge_model or os.environ.get("MISTRAL_MODEL") or "mistral-small-latest"
    judged: list[dict[str, Any]] = []
    for start in range(0, len(pairs), args.judge_batch_size):
        batch = pairs[start : start + args.judge_batch_size]
        result = judge_batch(client, model, batch, args.judge_max_tokens)
        by_id = {item.pair_id: item for item in result.items}
        for pair in batch:
            item = by_id.get(pair["pair_id"])
            out = dict(pair)
            if item:
                out["judge_label"] = item.label
                out["judge_score"] = item.score
                out["judge_rationale"] = item.rationale
            else:
                out["judge_label"] = "missing"
                out["judge_score"] = None
                out["judge_rationale"] = "missing pair_id in judge response"
            judged.append(out)
        print(f"judged {min(start + len(batch), len(pairs))}/{len(pairs)}")
        if args.judge_sleep:
            time.sleep(args.judge_sleep)
    return judged


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    expected_total: Counter[str] = Counter()
    expected_ok: Counter[str] = Counter()
    for row in rows:
        kind = row["candidate_kind"]
        label = str(row.get("judge_label", "unjudged"))
        by_kind[kind][label] += 1
        expected = row.get("expected_score")
        if expected is None or "judge_score" not in row or row["judge_score"] is None:
            continue
        expected_total[kind] += 1
        expected_label = "preserved" if expected >= 0.9 else "changed" if expected <= 0.1 else "uncertain"
        if label == expected_label:
            expected_ok[kind] += 1
    return {
        "total": len(rows),
        "by_candidate_kind": {k: dict(v) for k, v in by_kind.items()},
        "expected_agreement": {
            k: {
                "ok": expected_ok[k],
                "total": expected_total[k],
                "rate": round(expected_ok[k] / max(expected_total[k], 1), 3),
            }
            for k in sorted(expected_total)
        },
    }


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Small Substitution Probe",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Examples",
    ]
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[row["candidate_kind"]].append(row)
    for kind in sorted(by_kind):
        lines.extend(["", f"### {kind}", ""])
        for row in by_kind[kind][:8]:
            lines.extend(
                [
                    f"- `{row['entity']}` -> `{row['candidate']}` | judge={row.get('judge_label')} score={row.get('judge_score')} expected={row.get('expected_score')} cosine={row.get('cosine')}",
                    f"  - short: {row.get('short_2w', '')}",
                    f"  - reason: {row.get('judge_rationale', '')}",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/entity_inventory.sqlite"))
    parser.add_argument(
        "--tag-files",
        type=Path,
        nargs="+",
        default=[
            Path("data/entity_tags_top50k_chunk01.mistral.jsonl"),
            Path("data/entity_tags_top50k_chunk03.openrouter.jsonl"),
        ],
    )
    parser.add_argument(
        "--tag-selection",
        choices=["mistral_preferred", "all"],
        default="mistral_preferred",
        help="mistral_preferred keeps older behavior; all uses every passed tag file.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/substitution_probe"))
    parser.add_argument("--identity-controls", type=int, default=10)
    parser.add_argument("--numeric-n", type=int, default=20)
    parser.add_argument("--proper-n", type=int, default=20)
    parser.add_argument("--surface-variant-n", type=int, default=10)
    parser.add_argument("--common-high-n", type=int, default=20)
    parser.add_argument("--common-mid-n", type=int, default=20)
    parser.add_argument("--common-pool", type=int, default=320)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=Path("models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000"),
    )
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--judge-max-tokens", type=int, default=4500)
    parser.add_argument("--judge-sleep", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = generate_pairs(args)
    write_jsonl(args.out_dir / "probe_pairs.jsonl", pairs)
    print(f"generated pairs={len(pairs)}")

    rows = pairs if args.skip_judge else run_judge(args, pairs)
    write_jsonl(args.out_dir / "probe_judged.jsonl", rows)
    summary = summarize(rows)
    (args.out_dir / "probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.out_dir / "probe_report.md", rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
