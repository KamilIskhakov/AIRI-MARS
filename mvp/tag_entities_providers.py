#!/usr/bin/env python3
"""Tag entity inventory with different LLM providers and validate via Pydantic."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from entity_schema import EntityTagBatch
from tag_entities_mistral import (
    SYSTEM_PROMPT,
    align_and_repair_batch,
    compact_item,
    heuristic_fallback_tag,
    load_env_file,
    load_inventory,
)


PROVIDERS = {
    "openrouter": {
        "api_key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
        "model": "OPENAI_MODEL",
        "default_base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4.1-mini",
    },
    "groq": {
        "api_key": "GROQ_API_KEY",
        "base_url": "GROQ_BASE_URL",
        "model": "GROQ_MODEL",
        "schema_model": "GROQ_STRONG_MODEL",
        "default_base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-120b",
    },
    "groq_fast": {
        "api_key": "GROQ_API_KEY",
        "base_url": "GROQ_BASE_URL",
        "model": "GROQ_FAST_MODEL",
        "default_base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-8b-instant",
    },
    "cerebras": {
        "api_key": "CEREBRAS_API_KEY",
        "base_url": "CEREBRAS_BASE_URL",
        "model": "CEREBRAS_MODEL",
        "schema_model": "CEREBRAS_STRONG_MODEL",
        "default_base_url": "https://api.cerebras.ai/v1",
        "default_model": "gpt-oss-120b",
    },
}


SCHEMA_INSTRUCTIONS = """
Required output JSON:
{
  "items": [
    {
      "entity_id": "same id as input",
      "entity": "same entity string as input",
      "coarse_group": "proper_name|numeric|common_entity|domain_term|ambiguous|junk",
      "fine_type": "PERSON|ORG|GPE|LOC|FAC|NORP|EVENT|LAW|PRODUCT|WORK_OF_ART|DATE|TIME|CARDINAL|ORDINAL|MONEY|PERCENT|QUANTITY|COMMON_NOUN|DOMAIN_TERM|OTHER|UNKNOWN|JUNK",
      "context_policy": "short_window|full_context|no_context_embedding|agent_review|drop",
      "confidence": 0.0,
      "rationale": "short reason"
    }
  ]
}
Do not use fields named "type", "category", or "label".
Do not omit confidence, rationale, fine_type, coarse_group, or context_policy.
"""


API_SCHEMA_UNSUPPORTED_KEYS = {
    "title",
    "description",
    "default",
    "examples",
    "maxLength",
    "minLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
}


def api_compatible_schema(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: api_compatible_schema(value)
            for key, value in node.items()
            if key not in API_SCHEMA_UNSUPPORTED_KEYS
        }
    if isinstance(node, list):
        return [api_compatible_schema(value) for value in node]
    return node


def provider_config(provider: str, model_override: str | None = None, schema_mode: str = "json_schema") -> tuple[str, str, str]:
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["api_key"])
    if not api_key:
        raise RuntimeError(f"Missing {cfg['api_key']} in environment")
    base_url = os.environ.get(cfg["base_url"]) or cfg["default_base_url"]
    schema_model_key = cfg.get("schema_model") if schema_mode == "json_schema" else None
    model = (
        model_override
        or (os.environ.get(schema_model_key) if schema_model_key else None)
        or os.environ.get(cfg["model"])
        or cfg["default_model"]
    )
    return api_key, base_url.rstrip("/"), model


def parse_json_content(content: str) -> EntityTagBatch:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start : end + 1]
    return EntityTagBatch.model_validate_json(content)


def normalize_surface_for_alignment(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def pydantic_response_format(schema_mode: str) -> dict[str, Any]:
    if schema_mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_tag_batch",
            "strict": True,
            "schema": api_compatible_schema(EntityTagBatch.model_json_schema()),
        },
    }


def call_openai_compatible(
    provider: str,
    batch: list[dict[str, Any]],
    include_context: bool,
    model_override: str | None,
    timeout: int,
    max_tokens: int,
    schema_mode: str,
) -> EntityTagBatch:
    api_key, base_url, model = provider_config(provider, model_override, schema_mode)
    payload_items = [compact_item(item, include_context) for item in batch]
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": pydantic_response_format(schema_mode),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Classify these entities. Keep entity_id and entity unchanged. "
                    "Return exactly one item per input in the same order. "
                    "The API response_format contains the Pydantic-derived JSON Schema.\n"
                    + SCHEMA_INSTRUCTIONS
                    + "\nInput:\n"
                    + json.dumps(payload_items, ensure_ascii=False)
                ),
            },
        ],
    }
    headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "airi-mars-entity-tagger/0.1",
            "HTTP-Referer": "http://localhost",
            "X-Title": "AIRI MARS entity tagging",
    }
    try:
        import httpx

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{base_url}/chat/completions", headers=headers, json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"{provider} HTTP {resp.status_code}: {resp.text[:1000]}")
        data = resp.json()
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"{provider} request failed: {exc}") from exc
    content = data["choices"][0]["message"]["content"]
    return parse_json_content(content)


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
    provider: str,
    batch: list[dict[str, Any]],
    include_context: bool,
    model_override: str | None,
    timeout: int,
    max_tokens: int,
    schema_mode: str,
    retries: int,
    retry_sleep: float,
    batch_idx: int,
) -> tuple[EntityTagBatch, list[dict[str, Any]]]:
    attempt = 0
    last_error: Exception | None = None
    while True:
        try:
            result = call_openai_compatible(
                provider,
                batch,
                include_context,
                model_override,
                timeout,
                max_tokens,
                schema_mode,
            )
            return align_and_repair_batch(result, batch, batch_idx)
        except Exception as exc:
            last_error = exc
            attempt += 1
            if attempt > retries:
                raise last_error
            delay = retry_sleep * min(2 ** (attempt - 1), 8)
            print(
                f"provider={provider} batch={batch_idx} retry={attempt}/{retries} "
                f"sleep={delay:.1f}s error={type(exc).__name__}: {exc}"
            )
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--schema-mode", choices=["json_schema", "json_object"], default="json_schema")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fallback-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warnings-out", type=Path)
    args = parser.parse_args()

    load_env_file(Path("mvp/.env"))
    load_env_file(Path(".env"))
    items = load_inventory(args.inventory, args.limit or None)
    existing_ids = load_existing_ids(args.out) if args.resume else set()
    if existing_ids:
        before = len(items)
        items = [item for item in items if str(item["entity_id"]) not in existing_ids]
        print(f"resume=true existing={len(existing_ids)} remaining={len(items)} skipped_from_input={before - len(items)}")
    batches = [items[i : i + args.batch_size] for i in range(0, len(items), args.batch_size)]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    written = 0
    mode = "a" if args.resume and args.out.exists() else "w"
    warnings_path = args.warnings_out or args.out.with_suffix(args.out.suffix + ".warnings.jsonl")
    with args.out.open(mode, encoding="utf-8") as fh:
        warn_mode = "a" if args.resume and warnings_path.exists() else "w"
        warnings_fh = warnings_path.open(warn_mode, encoding="utf-8")
        for idx, batch in enumerate(batches, start=1):
            t0 = time.perf_counter()
            try:
                result, warnings = call_with_retries(
                    args.provider,
                    batch,
                    args.include_context,
                    args.model,
                    args.timeout,
                    args.max_tokens,
                    args.schema_mode,
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
                f"provider={args.provider} batch={idx}/{len(batches)} items={len(batch)} "
                f"seconds={dt:.2f} warnings={len(warnings)}"
            )
            if args.sleep:
                time.sleep(args.sleep)
        warnings_fh.close()

    total = time.perf_counter() - started
    print(f"provider={args.provider} written={written} total_seconds={total:.2f} items_per_sec={written / max(total, 1e-9):.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
