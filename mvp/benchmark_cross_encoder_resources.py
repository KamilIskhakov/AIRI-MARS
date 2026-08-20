#!/usr/bin/env python3
"""Benchmark one/few cross-encoder training steps on local hardware."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_cross_encoder import (
    SPECIAL_TOKENS,
    PairDataset,
    amp_context,
    class_weights,
    collate,
    dedupe_rows,
    normalize_row,
    pick_device,
)


def load_rows(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = normalize_row(json.loads(line))
            if row is None:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return dedupe_rows(rows)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def max_rss_mb() -> float:
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return rss / 1024 / 1024
    return rss / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-mode", choices=["pair", "marked_pair", "entity_query"], default="marked_pair")
    parser.add_argument("--sample-rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    rows = load_rows(args.train_jsonl, args.sample_rows)
    if not rows:
        raise RuntimeError("No trainable rows found.")

    print(
        json.dumps(
            {
                "event": "load_start",
                "base_model": args.base_model,
                "device": str(device),
                "rows": len(rows),
                "batch_size": args.batch_size,
        "max_length": args.max_length,
        "input_mode": args.input_mode,
        "pad_to_max_length": args.pad_to_max_length,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    start_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    if added:
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    sync_device(device)
    load_seconds = time.perf_counter() - start_load

    dataset = PairDataset(rows, tokenizer, args.max_length, args.input_mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate(
            tokenizer,
            batch,
            args.pad_to_multiple_of,
            args.max_length if args.pad_to_max_length else None,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_weights = class_weights(rows, device)

    timings: list[float] = []
    losses: list[float] = []
    iterator = iter(loader)
    total_steps = args.warmup_steps + args.steps
    for step in range(total_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch.pop("meta_idx")
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        sync_device(device)
        t0 = time.perf_counter()
        with amp_context(device, args.precision):
            out = model(**{key: value for key, value in batch.items() if key != "labels"})
            loss = F.cross_entropy(out.logits, batch["labels"], weight=loss_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync_device(device)
        seconds = time.perf_counter() - t0
        if step >= args.warmup_steps:
            timings.append(seconds)
            losses.append(float(loss.detach().cpu()))
        print(
            json.dumps(
                {
                    "event": "step",
                    "step": step + 1,
                    "warmup": step < args.warmup_steps,
                    "seconds": round(seconds, 4),
                    "loss": round(float(loss.detach().cpu()), 5),
                    "rss_mb": round(max_rss_mb(), 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    avg_seconds = sum(timings) / max(len(timings), 1)
    train_rows = sum(1 for line in args.train_jsonl.open(encoding="utf-8") if line.strip())
    steps_per_epoch = (train_rows + args.batch_size - 1) // args.batch_size
    result = {
        "event": "result",
        "load_seconds": round(load_seconds, 2),
        "avg_train_step_seconds": round(avg_seconds, 4),
        "timed_steps": len(timings),
        "sample_losses": [round(x, 5) for x in losses],
        "train_rows": train_rows,
        "estimated_steps_per_epoch": steps_per_epoch,
        "estimated_epoch_seconds": round(avg_seconds * steps_per_epoch, 1),
        "estimated_epoch_hours": round(avg_seconds * steps_per_epoch / 3600, 3),
        "rss_mb": round(max_rss_mb(), 1),
        "device": str(device),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
