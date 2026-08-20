#!/usr/bin/env python3
"""Judge substitution pairs with OpenAI-compatible providers.

Supported providers reuse the configuration from tag_entities_providers.py:
openrouter, groq, groq_fast, cerebras.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_small_substitution_probe import JUDGE_PROMPT, JudgmentBatch
from tag_entities_mistral import load_env_file
from tag_entities_providers import PROVIDERS, api_compatible_schema, provider_config


LABEL_TO_SCORE = {"changed": 0.0, "uncertain": 0.5, "preserved": 1.0}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


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
            pair_id = item.get("pair_id")
            if pair_id:
                ids.add(str(pair_id))
    return ids


def parse_json_content(content: str) -> JudgmentBatch:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start : end + 1]
    data = json.loads(content)
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        for item in data["items"]:
            if not isinstance(item, dict):
                continue
            item["rationale"] = str(item.get("rationale", ""))[:240]
            label = str(item.get("label", "uncertain"))
            if "score" not in item or item["score"] is None:
                item["score"] = LABEL_TO_SCORE.get(label, 0.5)
        return JudgmentBatch.model_validate(data)
    if isinstance(data, dict):
        items = []
        for pair_id, value in data.items():
            if isinstance(value, str):
                label = value
                rationale = "provider returned compact pair_id-to-label map"
                score = LABEL_TO_SCORE.get(label, 0.5)
            elif isinstance(value, dict):
                label = str(value.get("label", value.get("judge_label", "uncertain")))
                rationale = str(value.get("rationale", value.get("reason", "")))[:240]
                score = float(value.get("score", LABEL_TO_SCORE.get(label, 0.5)))
            else:
                label = "uncertain"
                rationale = "provider returned unsupported compact value"
                score = 0.5
            items.append(
                {
                    "pair_id": str(pair_id),
                    "label": label,
                    "score": score,
                    "rationale": rationale,
                }
            )
        return JudgmentBatch.model_validate({"items": items})
    return JudgmentBatch.model_validate(data)


def strict_objects_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {key: strict_objects_schema(value) for key, value in node.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(node, list):
        return [strict_objects_schema(value) for value in node]
    return node


def response_format(schema_mode: str) -> dict[str, Any]:
    if schema_mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "substitution_judgment_batch",
            "strict": True,
            "schema": strict_objects_schema(api_compatible_schema(JudgmentBatch.model_json_schema())),
        },
    }


def compact_pair(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": pair["pair_id"],
        "branch": pair.get("branch"),
        "candidate_kind": pair.get("candidate_kind"),
        "entity": pair.get("entity"),
        "candidate": pair.get("candidate"),
        "original_context": pair.get("original_context") or pair.get("left"),
        "candidate_context": pair.get("candidate_context") or pair.get("right"),
    }


def call_openai_compatible_judge(
    provider: str,
    batch: list[dict[str, Any]],
    model_override: str | None,
    timeout: int,
    max_tokens: int,
    schema_mode: str,
) -> tuple[JudgmentBatch, str]:
    api_key, base_url, model = provider_config(provider, model_override, schema_mode)
    payload = [compact_pair(pair) for pair in batch]
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": response_format(schema_mode),
        "messages": [
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    "Judge these substitutions. Keep pair_id unchanged. "
                    "Return JSON with this exact top-level shape: "
                    '{"items":[{"pair_id":"...","label":"preserved|changed|uncertain",'
                    '"score":0.0,"rationale":"short reason"}]}.\n'
                )
                + json.dumps(payload, ensure_ascii=False),
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "airi-mars-substitution-judge/0.1",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AIRI MARS substitution judge",
    }

    import httpx

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{base_url}/chat/completions", headers=headers, json=body)
    if resp.status_code >= 400:
        raise RuntimeError(f"{provider} HTTP {resp.status_code}: {resp.text[:1000]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return parse_json_content(content), model


def call_with_retries(
    provider: str,
    batch: list[dict[str, Any]],
    model_override: str | None,
    timeout: int,
    max_tokens: int,
    schema_mode: str,
    retries: int,
    retry_sleep: float,
    batch_idx: int,
) -> tuple[JudgmentBatch, str]:
    attempt = 0
    last_error: Exception | None = None
    while True:
        try:
            return call_openai_compatible_judge(
                provider,
                batch,
                model_override,
                timeout,
                max_tokens,
                schema_mode,
            )
        except Exception as exc:
            last_error = exc
            attempt += 1
            if attempt > retries:
                raise last_error
            delay = retry_sleep * min(2 ** (attempt - 1), 8)
            print(
                f"provider={provider} batch={batch_idx} retry={attempt}/{retries} "
                f"sleep={delay:.1f}s error={type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(delay)


def summarize(path: Path) -> dict[str, Any]:
    total = 0
    by_kind: dict[str, dict[str, int]] = {}
    expected_total: dict[str, int] = {}
    expected_ok: dict[str, int] = {}
    by_provider: dict[str, int] = {}
    if not path.exists():
        return {"total": 0, "by_candidate_kind": {}, "expected_agreement": {}}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            provider = str(item.get("judge_provider", "unknown"))
            by_provider[provider] = by_provider.get(provider, 0) + 1
            kind = item.get("candidate_kind", "unknown")
            label = item.get("judge_label", "missing")
            by_kind.setdefault(kind, {})
            by_kind[kind][label] = by_kind[kind].get(label, 0) + 1
            expected = item.get("expected_score")
            if expected is None:
                continue
            expected_label = "preserved" if float(expected) >= 0.9 else "changed" if float(expected) <= 0.1 else "uncertain"
            expected_total[kind] = expected_total.get(kind, 0) + 1
            if label == expected_label:
                expected_ok[kind] = expected_ok.get(kind, 0) + 1
    return {
        "total": total,
        "by_provider": by_provider,
        "by_candidate_kind": by_kind,
        "expected_agreement": {
            kind: {
                "ok": expected_ok.get(kind, 0),
                "total": count,
                "rate": round(expected_ok.get(kind, 0) / max(count, 1), 3),
            }
            for kind, count in sorted(expected_total.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=4500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--schema-mode", choices=["json_schema", "json_object"], default="json_schema")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warnings-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    load_env_file(Path("mvp/.env"))
    load_env_file(Path(".env"))
    pairs = load_jsonl(args.input_jsonl)
    existing = load_existing_ids(args.out) if args.resume else set()
    if existing:
        before = len(pairs)
        pairs = [pair for pair in pairs if str(pair.get("pair_id")) not in existing]
        print(
            f"resume=true existing={len(existing)} remaining={len(pairs)} "
            f"skipped_from_input={before - len(pairs)}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.warnings_out:
        args.warnings_out.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume and args.out.exists() else "w"
    warn_fh = (
        args.warnings_out.open(
            "a" if args.resume and args.warnings_out.exists() else "w",
            encoding="utf-8",
        )
        if args.warnings_out
        else None
    )

    written = 0
    started = time.time()
    provider_model = args.model or ""
    try:
        with args.out.open(mode, encoding="utf-8") as out_fh:
            batches = [pairs[i : i + args.batch_size] for i in range(0, len(pairs), args.batch_size)]
            for batch_idx, batch in enumerate(batches, start=1):
                batch_started = time.time()
                result, provider_model = call_with_retries(
                    args.provider,
                    batch,
                    args.model,
                    args.timeout,
                    args.max_tokens,
                    args.schema_mode,
                    args.retries,
                    args.retry_sleep,
                    batch_idx,
                )
                by_id = {item.pair_id: item for item in result.items}
                missing = 0
                for pair in batch:
                    judged = by_id.get(str(pair["pair_id"]))
                    item = dict(pair)
                    item["judge_provider"] = args.provider
                    item["judge_model"] = provider_model
                    if judged is None:
                        missing += 1
                        item["judge_label"] = "missing"
                        item["judge_score"] = None
                        item["judge_rationale"] = "missing pair_id in judge response"
                    else:
                        item["judge_label"] = judged.label
                        item["judge_score"] = judged.score
                        item["judge_rationale"] = judged.rationale
                    out_fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written += 1
                out_fh.flush()
                if missing and warn_fh:
                    warn_fh.write(
                        json.dumps(
                            {
                                "batch_idx": batch_idx,
                                "missing": missing,
                                "pair_ids": [p["pair_id"] for p in batch if str(p["pair_id"]) not in by_id],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    warn_fh.flush()
                print(
                    f"provider={args.provider} model={provider_model} batch={batch_idx}/{len(batches)} "
                    f"items={len(batch)} seconds={time.time() - batch_started:.2f} "
                    f"written={written} missing={missing}",
                    flush=True,
                )
                if args.sleep:
                    time.sleep(args.sleep)
    finally:
        if warn_fh:
            warn_fh.close()

    summary = summarize(args.out)
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written={written} total_seconds={time.time() - started:.2f}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
