#!/usr/bin/env python3
"""Judge generated substitution pairs with Mistral, with resume support."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from run_small_substitution_probe import JudgmentBatch, judge_batch, load_mistral_client
from tag_entities_mistral import load_env_file


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


def call_with_retries(
    client,
    model: str,
    batch: list[dict[str, Any]],
    max_tokens: int,
    retries: int,
    retry_sleep: float,
    batch_idx: int,
) -> JudgmentBatch:
    attempt = 0
    last_error: Exception | None = None
    while True:
        try:
            return judge_batch(client, model, batch, max_tokens)
        except Exception as exc:
            last_error = exc
            attempt += 1
            if attempt > retries:
                raise last_error
            delay = retry_sleep * min(2 ** (attempt - 1), 8)
            print(
                f"batch={batch_idx} retry={attempt}/{retries} sleep={delay:.1f}s "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(delay)


def summarize(path: Path) -> dict[str, Any]:
    total = 0
    by_kind: dict[str, dict[str, int]] = {}
    expected_total: dict[str, int] = {}
    expected_ok: dict[str, int] = {}
    if not path.exists():
        return {"total": 0, "by_candidate_kind": {}, "expected_agreement": {}}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
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
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4500)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warnings-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    load_env_file(Path("mvp/.env"))
    load_env_file(Path(".env"))
    model = args.model or os.environ.get("MISTRAL_MODEL") or "mistral-small-latest"

    pairs = load_jsonl(args.input_jsonl)
    if args.limit:
        pairs = pairs[: args.limit]
    existing = load_existing_ids(args.out) if args.resume else set()
    if existing:
        before = len(pairs)
        pairs = [pair for pair in pairs if str(pair.get("pair_id")) not in existing]
        print(f"resume=true existing={len(existing)} remaining={len(pairs)} skipped_from_input={before - len(pairs)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.warnings_out:
        args.warnings_out.parent.mkdir(parents=True, exist_ok=True)

    client = load_mistral_client()
    mode = "a" if args.resume and args.out.exists() else "w"
    warn_fh = args.warnings_out.open("a" if args.resume and args.warnings_out.exists() else "w", encoding="utf-8") if args.warnings_out else None

    written = 0
    started = time.time()
    try:
        with args.out.open(mode, encoding="utf-8") as out_fh:
            batches = [pairs[i : i + args.batch_size] for i in range(0, len(pairs), args.batch_size)]
            for batch_idx, batch in enumerate(batches, start=1):
                batch_started = time.time()
                result = call_with_retries(
                    client,
                    model,
                    batch,
                    args.max_tokens,
                    args.retries,
                    args.retry_sleep,
                    batch_idx,
                )
                by_id = {item.pair_id: item for item in result.items}
                missing = 0
                for pair in batch:
                    judged = by_id.get(str(pair["pair_id"]))
                    item = dict(pair)
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
                    f"batch={batch_idx}/{len(batches)} items={len(batch)} "
                    f"seconds={time.time() - batch_started:.2f} written={written} missing={missing}",
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
