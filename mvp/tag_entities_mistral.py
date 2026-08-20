#!/usr/bin/env python3
"""Tag entity inventory with Mistral structured output and Pydantic validation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from entity_schema import EntityTag, EntityTagBatch


SYSTEM_PROMPT = """You label masked entities for a contextual substitution project.

Return only the required structured JSON.

Definitions:
- proper_name: concrete named entity, person, organization, place, law, event, product, artwork, nationality group.
- numeric: date, time, cardinal number, money, percent, quantity.
- common_entity: ordinary common noun or simple generic concept.
- domain_term: domain-specific term or multi-word concept that is not a concrete named entity.
- ambiguous: cannot decide from surface form alone.
- junk: punctuation, malformed extraction, section numbering with no semantic value.

Context policy:
- short_window: enough to use 1-2 words before/after; usually numeric/date/quantity.
- full_context: needs whole local context; usually concrete names, people, organizations, places.
- no_context_embedding: can first generate candidates by embedding similarity without full context; usually common words/domain terms.
- agent_review: too ambiguous; ask another agent/human with context.
- drop: do not use for training.

Be conservative. If a surface form can be a proper name and a common word, choose ambiguous or full_context.
"""


class BatchAlignmentError(ValueError):
    pass


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_mistral_client():
    try:
        from mistralai import Mistral
    except Exception:
        from mistralai.client import Mistral  # type: ignore

    load_env_file(Path("mvp/.env"))
    load_env_file(Path(".env"))

    api_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set MISTRAL_API_KEY or OPENAI_API_KEY before running this script.")
    server_url = os.environ.get("MISTRAL_BASE_URL")
    if server_url:
        return Mistral(api_key=api_key, server_url=server_url)
    return Mistral(api_key=api_key)


def load_inventory(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            items.append(item)
            if limit and len(items) >= limit:
                break
    return items


def compact_item(item: dict[str, Any], include_context: bool) -> dict[str, Any]:
    out = {
        "entity_id": item["entity_id"],
        "entity": item["entity"],
        "count": item["count"],
        "observed_types": item.get("observed_types", {}),
        "heuristic_group": item.get("heuristic_group"),
        "heuristic_context_policy": item.get("heuristic_context_policy"),
        "domains": item.get("domains", {}),
    }
    if include_context:
        out["examples"] = item.get("examples", [])[:2]
    return out


def call_mistral(client, model: str, batch: list[dict[str, Any]], include_context: bool, max_tokens: int):
    payload = [compact_item(item, include_context) for item in batch]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Classify these entities. Keep entity_id and entity unchanged. "
                "Return one tag object per input item in the same order.\n\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]
    response = client.chat.parse(
        model=model,
        messages=messages,
        response_format=EntityTagBatch,
        temperature=0,
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        return parsed
    return EntityTagBatch.model_validate_json(message.content)


def normalized_surface(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def align_and_repair_batch(
    result: EntityTagBatch,
    batch: list[dict[str, Any]],
    batch_idx: int,
) -> tuple[EntityTagBatch, list[dict[str, Any]]]:
    if len(result.items) != len(batch):
        raise BatchAlignmentError(
            f"Batch {batch_idx}: got {len(result.items)} tags for {len(batch)} inputs"
        )

    by_id: dict[str, EntityTag] = {}
    for tag in result.items:
        if tag.entity_id in by_id:
            raise BatchAlignmentError(f"Batch {batch_idx}: duplicate entity_id {tag.entity_id!r}")
        by_id[tag.entity_id] = tag

    expected_ids = [str(source["entity_id"]) for source in batch]
    if set(by_id) != set(expected_ids):
        missing = [entity_id for entity_id in expected_ids if entity_id not in by_id]
        extra = [entity_id for entity_id in by_id if entity_id not in set(expected_ids)]
        raise BatchAlignmentError(
            f"Batch {batch_idx}: entity_id set mismatch missing={missing[:5]} extra={extra[:5]}"
        )

    aligned: list[EntityTag] = []
    warnings: list[dict[str, Any]] = []
    for pos, source in enumerate(batch, start=1):
        tag = by_id[str(source["entity_id"])]
        normalized_tag = normalized_surface(tag.entity)
        normalized_source = normalized_surface(source["entity"])
        if tag.entity != source["entity"] and normalized_tag != normalized_source:
            warnings.append(
                {
                    "batch_idx": batch_idx,
                    "item_pos": pos,
                    "entity_id": source["entity_id"],
                    "source_entity": source["entity"],
                    "model_entity": tag.entity,
                    "action": "repaired_entity_surface_from_source_inventory",
                }
            )
        tag.entity = source["entity"]
        aligned.append(tag)
    return EntityTagBatch(items=aligned), warnings


def heuristic_fallback_tag(source: dict[str, Any], reason: str) -> EntityTag:
    group = source.get("heuristic_group") or "ambiguous"
    if group not in {"proper_name", "numeric", "common_entity", "domain_term", "ambiguous", "junk"}:
        group = "ambiguous"

    observed_types = source.get("observed_types") or {}
    fine_type = next(iter(observed_types.keys()), "UNKNOWN") if isinstance(observed_types, dict) else "UNKNOWN"
    allowed_fine_types = {
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
        "DATE",
        "TIME",
        "CARDINAL",
        "ORDINAL",
        "MONEY",
        "PERCENT",
        "QUANTITY",
        "COMMON_NOUN",
        "DOMAIN_TERM",
        "OTHER",
        "UNKNOWN",
        "JUNK",
    }
    if fine_type not in allowed_fine_types:
        fine_type = "UNKNOWN"

    policy = source.get("heuristic_context_policy") or "agent_review"
    if policy not in {"short_window", "full_context", "no_context_embedding", "agent_review", "drop"}:
        policy = "agent_review"

    return EntityTag(
        entity_id=str(source["entity_id"]),
        entity=str(source["entity"]),
        coarse_group=group,
        fine_type=fine_type,
        context_policy=policy,
        confidence=0.35,
        rationale=f"Heuristic fallback after provider/alignment failure: {reason}"[:240],
    )


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Cannot resume: invalid JSON at {path}:{line_no}") from exc
            entity_id = item.get("entity_id")
            if entity_id:
                ids.add(str(entity_id))
    return ids


def call_with_retries(
    client,
    model: str,
    batch: list[dict[str, Any]],
    include_context: bool,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
    batch_idx: int,
):
    attempt = 0
    last_error: Exception | None = None
    while True:
        try:
            result = call_mistral(client, model, batch, include_context, max_tokens)
            return align_and_repair_batch(result, batch, batch_idx)
        except Exception as exc:
            last_error = exc
            attempt += 1
            if attempt > retries:
                raise last_error
            delay = retry_sleep * min(2 ** (attempt - 1), 8)
            print(
                f"provider=mistral batch={batch_idx} retry={attempt}/{retries} "
                f"sleep={delay:.1f}s error={type(exc).__name__}: {exc}"
            )
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fallback-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warnings-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file(Path("mvp/.env"))
    load_env_file(Path(".env"))
    model = args.model or os.environ.get("MISTRAL_MODEL") or os.environ.get("OPENAI_MODEL") or "ministral-8b-latest"

    items = load_inventory(args.inventory, args.limit or None)
    existing_ids = load_existing_ids(args.out) if args.resume else set()
    if existing_ids:
        before = len(items)
        items = [item for item in items if str(item["entity_id"]) not in existing_ids]
        print(f"resume=true existing={len(existing_ids)} remaining={len(items)} skipped_from_input={before - len(items)}")
    batches = [items[i : i + args.batch_size] for i in range(0, len(items), args.batch_size)]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        preview = {
            "system": SYSTEM_PROMPT,
            "first_batch": [compact_item(x, args.include_context) for x in batches[0]] if batches else [],
            "schema": EntityTagBatch.model_json_schema(),
            "model": model,
        }
        args.out.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"dry-run wrote {args.out}")
        return

    client = get_mistral_client()
    written = 0
    started = time.perf_counter()
    mode = "a" if args.resume and args.out.exists() else "w"
    warnings_path = args.warnings_out or args.out.with_suffix(args.out.suffix + ".warnings.jsonl")
    with args.out.open(mode, encoding="utf-8") as fh:
        warn_mode = "a" if args.resume and warnings_path.exists() else "w"
        warnings_fh = warnings_path.open(warn_mode, encoding="utf-8")
        for idx, batch in enumerate(batches, start=1):
            t0 = time.perf_counter()
            try:
                result, warnings = call_with_retries(
                    client,
                    model,
                    batch,
                    args.include_context,
                    args.max_tokens,
                    args.retries,
                    args.retry_sleep,
                    idx,
                )
            except Exception as exc:
                if not args.fallback_on_failure:
                    warnings_fh.close()
                    raise
                warnings = [
                    {
                        "batch_idx": idx,
                        "error": f"{type(exc).__name__}: {exc}",
                        "action": "heuristic_fallback_for_entire_batch",
                    }
                ]
                result = EntityTagBatch(
                    items=[heuristic_fallback_tag(source, str(exc)) for source in batch]
                )
            for warning in warnings:
                warnings_fh.write(json.dumps(warning, ensure_ascii=False) + "\n")
            for tag in result.items:
                fh.write(tag.model_dump_json() + "\n")
                written += 1
            fh.flush()
            warnings_fh.flush()
            dt = time.perf_counter() - t0
            print(
                f"batch={idx}/{len(batches)} items={len(batch)} seconds={dt:.2f} "
                f"written={written} warnings={len(warnings)}"
            )
            if args.sleep:
                time.sleep(args.sleep)
        warnings_fh.close()

    total = time.perf_counter() - started
    print(f"written={written} total_seconds={total:.2f} items_per_sec={written / max(total, 1e-9):.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
