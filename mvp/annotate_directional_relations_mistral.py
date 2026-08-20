#!/usr/bin/env python3
"""Annotate bidirectional entailment for entity substitutions with Mistral."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from tag_entities_mistral import load_env_file


class DirectionalJudgment(BaseModel):
    pair_id: str
    a_to_b: Literal["entailment", "neutral", "contradiction", "uncertain"]
    b_to_a: Literal["entailment", "neutral", "contradiction", "uncertain"]
    relation: Literal[
        "equivalence",
        "generalization",
        "specialization",
        "substitution",
        "contradiction",
        "uncertain",
    ]
    preserves_meaning: Literal["preserved", "changed", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=320)


class DirectionalBatch(BaseModel):
    items: list[DirectionalJudgment]


SYSTEM_PROMPT = """You annotate directional semantic relations caused by one entity substitution.

A is the original context and B is the substituted context. Judge both directions independently:
- entailment: if the premise is true, the hypothesis must be true in the same situation;
- neutral: the hypothesis may be true, but does not follow from the premise;
- contradiction: the hypothesis is incompatible with the premise;
- uncertain: context is insufficient or the text is malformed.

Map directions to relation:
- equivalence: A entails B and B entails A;
- generalization: A entails B, but B does not entail A;
- specialization: B entails A, but A does not entail B;
- substitution: neither direction entails and there is no direct contradiction;
- contradiction: at least one direction is genuinely incompatible;
- uncertain: confidence is insufficient.

Do not treat a different named entity as contradiction merely because it is different: often it is neutral.
Relatedness, similar wording, or grammatical fit do not imply entailment. Preserve pair_id unchanged.
If B adds specificity, A normally does not entail B, while B may entail A. If B removes
specificity, A may entail B, while B normally does not entail A.
Return only the structured response.
"""


def derive_relation(a_to_b: str, b_to_a: str) -> tuple[str, str]:
    if "uncertain" in {a_to_b, b_to_a}:
        return "uncertain", "uncertain"
    if "contradiction" in {a_to_b, b_to_a}:
        return "contradiction", "changed"
    if a_to_b == "entailment" and b_to_a == "entailment":
        return "equivalence", "preserved"
    if a_to_b == "entailment":
        return "generalization", "changed"
    if b_to_a == "entailment":
        return "specialization", "changed"
    return "substitution", "changed"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("pair_id")) for row in load_jsonl(path) if row.get("pair_id")}


def get_client():
    try:
        from mistralai import Mistral
    except ImportError:
        from mistralai.client import Mistral

    load_env_file(Path("mvp/.env"))
    load_env_file(Path(".env"))
    key_name = os.environ.get("MISTRAL_DIRECTIONAL_KEY", "MISTRAL_API_KEY_2")
    api_key = os.environ.get(key_name) or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing {key_name} or MISTRAL_API_KEY")
    return Mistral(api_key=api_key)


def call_batch(client, model: str, rows: list[dict[str, Any]], max_tokens: int) -> DirectionalBatch:
    payload = [
        {
            "pair_id": row["pair_id"],
            "entity": row.get("entity"),
            "candidate": row.get("candidate"),
            "A_original": row.get("left") or row.get("original_context"),
            "B_substituted": row.get("right") or row.get("candidate_context"),
        }
        for row in rows
    ]
    response = client.chat.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format=DirectionalBatch,
        temperature=0,
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    return parsed if parsed is not None else DirectionalBatch.model_validate_json(message.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="mistral-medium-latest")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=8.0)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    done = existing_ids(args.out) if args.resume else set()
    rows = [row for row in rows if str(row.get("pair_id")) not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    client = get_client()
    mode = "a" if args.resume and args.out.exists() else "w"
    with args.out.open(mode, encoding="utf-8") as out:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            last_error: Exception | None = None
            for attempt in range(args.retries + 1):
                try:
                    result = call_batch(client, args.model, batch, args.max_tokens)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= args.retries:
                        raise
                    time.sleep(args.retry_sleep * min(2 ** attempt, 8))
            else:
                raise RuntimeError(str(last_error))
            by_id = {item.pair_id: item for item in result.items}
            for row in batch:
                judgment = by_id.get(str(row["pair_id"]))
                item = dict(row)
                if judgment is None:
                    item["directional_error"] = "missing_pair_id"
                else:
                    raw = judgment.model_dump()
                    relation, preserves = derive_relation(raw["a_to_b"], raw["b_to_a"])
                    item["directional_a_to_b"] = raw["a_to_b"]
                    item["directional_b_to_a"] = raw["b_to_a"]
                    item["directional_relation"] = relation
                    item["directional_preserves_meaning"] = preserves
                    item["directional_confidence"] = raw["confidence"]
                    item["directional_rationale"] = raw["rationale"]
                    item["directional_agent_relation"] = raw["relation"]
                    item["directional_agent_preserves_meaning"] = raw["preserves_meaning"]
                    item["directional_consistent"] = (
                        raw["relation"] == relation and raw["preserves_meaning"] == preserves
                    )
                    item["directional_model"] = args.model
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            print(f"directional={min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
