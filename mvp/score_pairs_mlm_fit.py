#!/usr/bin/env python3
"""Score original and substituted spans with ModernBERT masked-LM likelihood."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def differing_spans(
    left: str,
    right: str,
    entity: str = "",
    candidate: str = "",
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if entity and candidate:
        for match in re.finditer(re.escape(entity), left, flags=re.IGNORECASE):
            rebuilt = left[: match.start()] + candidate + left[match.end() :]
            if rebuilt == right:
                return (match.start(), match.end()), (match.start(), match.start() + len(candidate))
    prefix = 0
    max_prefix = min(len(left), len(right))
    while prefix < max_prefix and left[prefix] == right[prefix]:
        prefix += 1
    left_remaining = len(left) - prefix
    right_remaining = len(right) - prefix
    suffix = 0
    while (
        suffix < left_remaining
        and suffix < right_remaining
        and left[len(left) - suffix - 1] == right[len(right) - suffix - 1]
    ):
        suffix += 1
    left_end = len(left) - suffix if suffix else len(left)
    right_end = len(right) - suffix if suffix else len(right)
    if prefix >= left_end or prefix >= right_end:
        return None
    return (prefix, left_end), (prefix, right_end)


def crop_span(text: str, span: tuple[int, int], chars: int) -> tuple[str, tuple[int, int]]:
    start, end = span
    lo = max(0, start - chars)
    hi = min(len(text), end + chars)
    return text[lo:hi], (start - lo, end - lo)


def make_job(tokenizer, text: str, span: tuple[int, int], max_length: int) -> dict[str, Any] | None:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoded.pop("offset_mapping")
    positions = [
        idx for idx, (start, end) in enumerate(offsets)
        if end > span[0] and start < span[1] and end > start
    ]
    if not positions:
        return None
    original_ids = [int(encoded["input_ids"][idx]) for idx in positions]
    masked_ids = list(encoded["input_ids"])
    for idx in positions:
        masked_ids[idx] = tokenizer.mask_token_id
    encoded["input_ids"] = masked_ids
    return {"encoded": encoded, "positions": positions, "target_ids": original_ids}


def score_jobs(model, tokenizer, jobs: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(jobs), batch_size):
            batch_jobs = jobs[start : start + batch_size]
            padded = tokenizer.pad(
                [job["encoded"] for job in batch_jobs], padding=True, return_tensors="pt"
            )
            logits = model(**{key: value.to(device) for key, value in padded.items()}).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            for batch_idx, job in enumerate(batch_jobs):
                values = [
                    float(log_probs[batch_idx, pos, token_id].item())
                    for pos, token_id in zip(job["positions"], job["target_ids"])
                ]
                results.append(
                    {
                        "avg_log_prob": sum(values) / len(values),
                        "sum_log_prob": sum(values),
                        "min_log_prob": min(values),
                        "token_count": len(values),
                    }
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--context-chars", type=int, default=700)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
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

    jobs: list[dict[str, Any]] = []
    job_refs: list[tuple[int, str]] = []
    skipped = 0
    output = [dict(row) for row in rows]
    for idx, row in enumerate(rows):
        left = str(row.get("left") or row.get("original_context") or "")
        right = str(row.get("right") or row.get("candidate_context") or "")
        spans = differing_spans(left, right, str(row.get("entity", "")), str(row.get("candidate", "")))
        if not spans:
            output[idx]["mlm_fit_error"] = "cannot_locate_single_changed_span"
            skipped += 1
            continue
        for side, text, span in (("original", left, spans[0]), ("candidate", right, spans[1])):
            cropped, local_span = crop_span(text, span, args.context_chars)
            job = make_job(tokenizer, cropped, local_span, args.max_length)
            if job is None:
                output[idx]["mlm_fit_error"] = f"cannot_align_{side}_span"
                skipped += 1
                break
            jobs.append(job)
            job_refs.append((idx, side))

    scores = score_jobs(model, tokenizer, jobs, device, args.batch_size)
    for (row_idx, side), score in zip(job_refs, scores):
        for key, value in score.items():
            output[row_idx][f"mlm_{side}_{key}"] = round(value, 6) if isinstance(value, float) else value
    complete = 0
    for row in output:
        original = row.get("mlm_original_avg_log_prob")
        candidate = row.get("mlm_candidate_avg_log_prob")
        if isinstance(original, float) and isinstance(candidate, float):
            row["mlm_candidate_delta_avg_log_prob"] = round(candidate - original, 6)
            row["mlm_fit_model"] = args.model
            complete += 1

    deltas = [row["mlm_candidate_delta_avg_log_prob"] for row in output if "mlm_candidate_delta_avg_log_prob" in row]
    summary = {
        "rows": len(rows),
        "complete": complete,
        "skipped": len(rows) - complete,
        "model": args.model,
        "delta_avg_log_prob": {
            "min": min(deltas, default=None),
            "max": max(deltas, default=None),
            "mean": round(sum(deltas) / len(deltas), 6) if deltas else None,
        },
    }
    write_jsonl(args.out, output)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
