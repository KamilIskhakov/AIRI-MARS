#!/usr/bin/env python3
"""Audit entity tags and probe MVP substitution strategies."""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MASK_RE = re.compile(r"<mask>")
WORD_RE = re.compile(r"[A-Za-zА-Яа-я0-9]+(?:[.,:/-][A-Za-zА-Яа-я0-9]+)*|[$€£%]+")
NUMERIC_FINE = {"DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT", "QUANTITY"}


def load_tags(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        provider = path.name.replace("entity_tags_top50k_", "").replace(".jsonl", "")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = (provider, str(item["entity_id"]))
                if key in seen:
                    continue
                seen.add(key)
                item["provider"] = provider
                rows.append(item)
    return rows


def connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def one_mention(conn: sqlite3.Connection, entity_id: str, rng: random.Random) -> dict[str, Any] | None:
    rows = conn.execute(
        """
        select
            m.id as mention_id,
            m.text_id,
            m.mask_idx,
            t.row_idx,
            t.source_id,
            d.name as dataset_name,
            d.split as dataset_split,
            d.id as dataset_id,
            d.domain,
            t.title,
            t.masked_text,
            t.original_text,
            t.summary
        from mentions m
        join texts t on t.id = m.text_id
        join datasets d on d.id = t.dataset_id
        where m.entity_id = ?
        limit 20
        """,
        (int(entity_id),),
    ).fetchall()
    if not rows:
        return None
    return dict(rng.choice(rows))


def sample_mentions(
    conn: sqlite3.Connection,
    entity_id: str,
    rng: random.Random,
    limit: int,
    pool_limit: int = 80,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        select
            m.id as mention_id,
            m.text_id,
            m.mask_idx,
            t.row_idx,
            t.source_id,
            d.name as dataset_name,
            d.split as dataset_split,
            d.id as dataset_id,
            d.domain,
            t.title,
            t.masked_text,
            t.original_text,
            t.summary
        from mentions m
        join texts t on t.id = m.text_id
        join datasets d on d.id = t.dataset_id
        where m.entity_id = ?
        limit ?
        """,
        (int(entity_id), int(pool_limit)),
    ).fetchall()
    if not rows:
        return []
    items = [dict(row) for row in rows]
    rng.shuffle(items)
    return items[: max(1, limit)]


def mask_span(masked_text: str, mask_idx: int) -> tuple[int, int] | None:
    matches = list(MASK_RE.finditer(masked_text))
    if mask_idx < 0 or mask_idx >= len(matches):
        return None
    match = matches[mask_idx]
    return match.start(), match.end()


def short_window(masked_text: str, mask_idx: int, entity: str, words: int = 2) -> str:
    span = mask_span(masked_text, mask_idx)
    if not span:
        return ""
    start, end = span
    left = WORD_RE.findall(masked_text[:start])[-words:]
    right = WORD_RE.findall(masked_text[end:])[:words]
    return " ".join(left + [f"[{entity}]"] + right)


def context_crop(masked_text: str, mask_idx: int, entity: str, chars: int = 240) -> str:
    span = mask_span(masked_text, mask_idx)
    if not span:
        return masked_text[: chars * 2]
    start, end = span
    lo = max(0, start - chars)
    hi = min(len(masked_text), end + chars)
    crop = masked_text[lo:start] + f"[{entity}]" + masked_text[end:hi]
    return re.sub(r"\s+", " ", crop).strip()


def perturb_number_like(entity: str) -> list[str]:
    text = entity.strip()
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    variants: list[str] = []
    if match:
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
        elif 1500 <= value <= 2200 and value.is_integer():
            values = [int(value) - 1, int(value) + 1]
        elif not has_decimal and value.is_integer():
            delta = max(1, round(abs(value) * 0.1))
            values = [int(value) + delta, max(0, int(value) - delta)]
        else:
            delta = max(1, abs(value) * 0.1)
            values = [value + delta, max(0, value - delta)]
        for new_value in values:
            if not has_decimal or float(new_value).is_integer():
                repl = f"{int(new_value):,}" if "," in raw else str(int(new_value))
            else:
                repl = f"{new_value:.2f}"
            variants.append(text[: match.start()] + repl + text[match.end() :])
    swaps = [("$", "€"), ("€", "$"), ("£", "$"), ("percent", "points"), ("%", " percent")]
    for a, b in swaps:
        if a in text:
            variants.append(text.replace(a, b, 1))
    return list(dict.fromkeys(v for v in variants if v and v != text))


def sample_by_group(tags: list[dict[str, Any]], per_group: int, rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tag in tags:
        buckets[tag.get("coarse_group", "unknown")].append(tag)
    sampled: list[dict[str, Any]] = []
    for group in sorted(buckets):
        items = buckets[group]
        sampled.extend(rng.sample(items, min(per_group, len(items))))
    return sampled


def summarize_tags(tags: list[dict[str, Any]]) -> dict[str, Any]:
    by_provider = Counter(t["provider"] for t in tags)
    by_group = Counter(t.get("coarse_group", "?") for t in tags)
    by_policy = Counter(t.get("context_policy", "?") for t in tags)
    confidence_bins = Counter()
    for tag in tags:
        conf = float(tag.get("confidence", -1))
        if conf == 0:
            confidence_bins["0"] += 1
        elif conf < 0.5:
            confidence_bins["<0.5"] += 1
        elif conf < 0.8:
            confidence_bins["0.5-0.8"] += 1
        else:
            confidence_bins[">=0.8"] += 1
    return {
        "total": len(tags),
        "by_provider": dict(by_provider),
        "by_group": dict(by_group),
        "by_policy": dict(by_policy),
        "confidence_bins": dict(confidence_bins),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_with_local_modernbert(texts: list[str], model_dir: Path, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=48, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors)


def embedding_probe(
    tags: list[dict[str, Any]],
    model_dir: Path,
    max_items: int,
    batch_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [
        t
        for t in tags
        if t.get("context_policy") == "no_context_embedding"
        or t.get("coarse_group") in {"common_entity", "domain_term"}
    ]
    unique: dict[str, dict[str, Any]] = {}
    for tag in candidates:
        surface = re.sub(r"\s+", " ", str(tag["entity"]).strip())
        if len(surface) < 3:
            continue
        unique.setdefault(surface.lower(), {**tag, "entity": surface})
    items = list(unique.values())
    rng.shuffle(items)
    items = items[:max_items]
    if len(items) < 3:
        return [], {"error": "not enough embedding candidates", "candidates": len(items)}

    texts = [i["entity"] for i in items]
    emb = encode_with_local_modernbert(texts, model_dir, batch_size)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)

    rows: list[dict[str, Any]] = []
    top_scores: list[float] = []
    for idx, item in enumerate(items):
        order = np.argsort(-sim[idx])[:5]
        neighbors = [
            {
                "entity": items[j]["entity"],
                "coarse_group": items[j].get("coarse_group"),
                "fine_type": items[j].get("fine_type"),
                "cosine": round(float(sim[idx, j]), 4),
            }
            for j in order
        ]
        top_scores.append(float(sim[idx, order[0]]))
        rows.append(
            {
                "entity_id": item["entity_id"],
                "entity": item["entity"],
                "coarse_group": item.get("coarse_group"),
                "fine_type": item.get("fine_type"),
                "provider": item["provider"],
                "neighbors": neighbors,
            }
        )

    examples = []
    for lo, hi, name in [(0.95, 1.01, ">=0.95"), (0.9, 0.95, "0.90-0.95"), (0.8, 0.9, "0.80-0.90")]:
        pool = [r for r in rows if lo <= r["neighbors"][0]["cosine"] < hi]
        examples.extend({**r, "bin": name} for r in pool[:8])

    summary = {
        "items": len(items),
        "top1_cosine_bins": dict(
            Counter(
                ">=0.95" if x >= 0.95 else "0.90-0.95" if x >= 0.9 else "0.80-0.90" if x >= 0.8 else "<0.80"
                for x in top_scores
            )
        ),
        "top1_mean": round(float(np.mean(top_scores)), 4),
        "top1_median": round(float(np.median(top_scores)), 4),
    }
    return examples, summary


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
    parser.add_argument("--out-dir", type=Path, default=Path("data/probes"))
    parser.add_argument("--audit-per-group", type=int, default=8)
    parser.add_argument("--numeric-sample", type=int, default=80)
    parser.add_argument("--proper-sample", type=int, default=80)
    parser.add_argument("--embedding-sample", type=int, default=600)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=Path("models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000"),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-embedding", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tags = load_tags(args.tag_files)
    conn = connect_ro(args.db)

    audit_rows: list[dict[str, Any]] = []
    for tag in sample_by_group(tags, args.audit_per_group, rng):
        mention = one_mention(conn, tag["entity_id"], rng)
        row = dict(tag)
        if mention:
            row.update(
                {
                    "dataset": f"{mention['dataset_name']}/{mention['dataset_split']}",
                    "domain": mention["domain"],
                    "short_2w": short_window(mention["masked_text"], int(mention["mask_idx"]), tag["entity"], 2),
                    "context_crop": context_crop(mention["masked_text"], int(mention["mask_idx"]), tag["entity"], 220),
                }
            )
        audit_rows.append(row)
    write_jsonl(args.out_dir / "random_tag_audit.jsonl", audit_rows)

    numeric_tags = [
        t for t in tags if t.get("coarse_group") == "numeric" or t.get("fine_type") in NUMERIC_FINE
    ]
    numeric_rows: list[dict[str, Any]] = []
    for tag in rng.sample(numeric_tags, min(args.numeric_sample, len(numeric_tags))):
        mention = one_mention(conn, tag["entity_id"], rng)
        if not mention:
            continue
        numeric_rows.append(
            {
                **tag,
                "dataset": f"{mention['dataset_name']}/{mention['dataset_split']}",
                "short_1w": short_window(mention["masked_text"], int(mention["mask_idx"]), tag["entity"], 1),
                "short_2w": short_window(mention["masked_text"], int(mention["mask_idx"]), tag["entity"], 2),
                "short_4w": short_window(mention["masked_text"], int(mention["mask_idx"]), tag["entity"], 4),
                "negative_candidates": perturb_number_like(tag["entity"])[:4],
                "context_crop": context_crop(mention["masked_text"], int(mention["mask_idx"]), tag["entity"], 180),
            }
        )
    write_jsonl(args.out_dir / "numeric_short_window_probe.jsonl", numeric_rows)

    proper_tags = [t for t in tags if t.get("coarse_group") == "proper_name"]
    proper_rows: list[dict[str, Any]] = []
    for tag in rng.sample(proper_tags, min(args.proper_sample, len(proper_tags))):
        mention = one_mention(conn, tag["entity_id"], rng)
        if not mention:
            continue
        proper_rows.append(
            {
                **tag,
                "dataset": f"{mention['dataset_name']}/{mention['dataset_split']}",
                "short_2w": short_window(mention["masked_text"], int(mention["mask_idx"]), tag["entity"], 2),
                "context_crop": context_crop(mention["masked_text"], int(mention["mask_idx"]), tag["entity"], 260),
            }
        )
    write_jsonl(args.out_dir / "proper_full_context_probe.jsonl", proper_rows)

    embedding_examples: list[dict[str, Any]] = []
    embedding_summary: dict[str, Any] = {"skipped": bool(args.skip_embedding)}
    if not args.skip_embedding:
        embedding_examples, embedding_summary = embedding_probe(
            tags,
            args.embedding_model,
            args.embedding_sample,
            args.embedding_batch_size,
            rng,
        )
        write_jsonl(args.out_dir / "common_embedding_neighbors.jsonl", embedding_examples)

    confidence_by_provider: dict[str, Counter[str]] = defaultdict(Counter)
    for tag in tags:
        conf = float(tag.get("confidence", -1))
        bucket = "0" if conf == 0 else "<0.5" if conf < 0.5 else "0.5-0.8" if conf < 0.8 else ">=0.8"
        confidence_by_provider[tag["provider"]][bucket] += 1

    summary = {
        "tag_summary": summarize_tags(tags),
        "confidence_by_provider": {k: dict(v) for k, v in confidence_by_provider.items()},
        "audit_rows": len(audit_rows),
        "numeric_probe_rows": len(numeric_rows),
        "proper_probe_rows": len(proper_rows),
        "embedding_summary": embedding_summary,
    }
    (args.out_dir / "probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# Entity Tag Audit And Probe",
        "",
        f"Tag files: {', '.join(str(p) for p in args.tag_files)}",
        f"Total tags: {summary['tag_summary']['total']}",
        "",
        "## Distribution",
        f"- Providers: {summary['tag_summary']['by_provider']}",
        f"- Groups: {summary['tag_summary']['by_group']}",
        f"- Policies: {summary['tag_summary']['by_policy']}",
        f"- Confidence bins: {summary['tag_summary']['confidence_bins']}",
        f"- Confidence by provider: {summary['confidence_by_provider']}",
        "",
        "## Probe Outputs",
        "- random_tag_audit.jsonl: random manual audit examples with short and cropped contexts.",
        "- numeric_short_window_probe.jsonl: numeric examples with 1/2/4-word windows and perturbation negatives.",
        "- proper_full_context_probe.jsonl: proper-name examples showing short window vs larger context.",
        "- common_embedding_neighbors.jsonl: nearest neighbors for no_context_embedding/common/domain surfaces.",
        "",
        "## Embedding Summary",
        json.dumps(embedding_summary, ensure_ascii=False, indent=2),
    ]
    (args.out_dir / "entity_tag_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
