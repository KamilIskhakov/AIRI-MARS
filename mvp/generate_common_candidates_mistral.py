#!/usr/bin/env python3
"""Generate contextual common-noun synonyms and hard negatives with Mistral."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from tag_entities_mistral import load_env_file


class CommonCandidate(BaseModel):
    candidate: str = Field(min_length=1, max_length=80)
    relation: Literal["contextual_synonym", "spelling_variant", "hypernym", "hyponym", "near_concept", "contrast"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=240)


class CommonItem(BaseModel):
    item_id: str
    entity: str
    positive_candidates: list[CommonCandidate]
    hard_negative_candidates: list[CommonCandidate]


class CommonBatch(BaseModel):
    items: list[CommonItem]


PROMPT = """Generate contextual substitutions for an ordinary noun or domain term.

positive_candidates must preserve the full factual meaning in this exact sentence in both
directions. Use only a genuine contextual synonym or harmless spelling variant. A related
word, hypernym, hyponym, part/whole, singular/plural change, or change of grammatical role
is not positive.
Confusable frequency words are not variants: biannual means twice per year, while biennial
means once every two years.

hard_negative_candidates must be grammatical and topically close, but definitely change
the stated fact. Prefer a nearby concept, hypernym, hyponym, or contrast that a weak model
could confuse with the source. Do not return absurd or unrelated words.

Do not copy the source string with case or whitespace changes. It is better to return an
empty positive list than invent a synonym. Keep item_id and entity unchanged. Return only
the structured response.
"""


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text: str) -> str:
    return re.sub(r"\W+", "", text.casefold())


def replace_first(text: str, source: str, candidate: str) -> str | None:
    match = re.search(re.escape(source), text, flags=re.IGNORECASE)
    if not match:
        return None
    return text[: match.start()] + candidate + text[match.end() :]


def stable_id(*parts: object) -> str:
    return hashlib.sha1("\x1f".join(str(x) for x in parts).encode()).hexdigest()[:20]


def get_client():
    try:
        from mistralai import Mistral
    except ImportError:
        from mistralai.client import Mistral
    load_env_file(Path("mvp/.env"))
    load_env_file(Path(".env"))
    key_name = os.environ.get("MISTRAL_COMMON_KEY", "MISTRAL_API_KEY_2")
    key = os.environ.get(key_name) or os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError(f"Missing {key_name} or MISTRAL_API_KEY")
    return Mistral(api_key=key)


def call(client, model: str, rows: list[dict[str, Any]], max_tokens: int) -> CommonBatch:
    payload = [
        {
            "item_id": row["item_id"],
            "entity": row["entity"],
            "context": row["original_context"],
        }
        for row in rows
    ]
    response = client.chat.parse(
        model=model,
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        response_format=CommonBatch,
        temperature=0,
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    return parsed if parsed is not None else CommonBatch.model_validate_json(message.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--raw-out", type=Path)
    parser.add_argument("--model", default="mistral-medium-latest")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=8.0)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    unique: dict[str, dict[str, Any]] = {}
    for row in load_rows(args.input):
        key = str(row.get("mention_id") or stable_id(row.get("entity_id"), row.get("original_context")))
        if key not in unique and row.get("entity") and row.get("original_context"):
            unique[key] = {**row, "item_id": key}
    rows = list(unique.values())
    if args.limit:
        rows = rows[: args.limit]

    client = get_client()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_out or args.out.with_suffix(".raw.jsonl")
    generated = 0
    rejected = 0
    with args.out.open("w", encoding="utf-8") as out, raw_path.open("w", encoding="utf-8") as raw:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            for attempt in range(args.retries + 1):
                try:
                    response = call(client, args.model, batch, args.max_tokens)
                    break
                except Exception:
                    if attempt >= args.retries:
                        raise
                    time.sleep(args.retry_sleep * min(2 ** attempt, 8))
            by_id = {item.item_id: item for item in response.items}
            for source in batch:
                result = by_id.get(source["item_id"])
                if result is None:
                    continue
                raw.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")
                candidates = [
                    (candidate, "common_agent_synonym", 1.0)
                    for candidate in result.positive_candidates
                ] + [
                    (candidate, "common_agent_hard_negative", 0.0)
                    for candidate in result.hard_negative_candidates
                ]
                seen = {norm(source["entity"])}
                for candidate, kind, expected in candidates:
                    value = " ".join(candidate.candidate.split())
                    key = norm(value)
                    context = replace_first(source["original_context"], source["entity"], value)
                    if not key or key in seen or context is None or context == source["original_context"]:
                        rejected += 1
                        continue
                    seen.add(key)
                    generated += 1
                    row = dict(source)
                    row.update(
                        {
                            "pair_id": f"ca{generated:08d}",
                            "candidate": value,
                            "candidate_context": context,
                            "right": context,
                            "branch": "common_agent_contextual",
                            "candidate_kind": kind,
                            "pair_kind": kind,
                            "expected_score": expected,
                            "generator_relation": candidate.relation,
                            "generator_confidence": candidate.confidence,
                            "generator_rationale": candidate.rationale,
                            "candidate_entity_id": f"generated:{key}",
                        }
                    )
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            raw.flush()
            print(f"common_agent={min(start + len(batch), len(rows))}/{len(rows)} pairs={generated}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    print(json.dumps({"source_mentions": len(rows), "pairs": generated, "rejected": rejected}, indent=2))


if __name__ == "__main__":
    main()
