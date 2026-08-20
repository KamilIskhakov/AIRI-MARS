#!/usr/bin/env python3
"""Calibrate a trained cross-encoder decision threshold on saved splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_cross_encoder import (
    PairDataset,
    binary_metrics,
    collate,
    grouped_metrics,
    load_jsonl,
    pick_device,
    write_json,
)


@torch.no_grad()
def collect_scores(
    model,
    loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    model.eval()
    scores = []
    labels = []
    idxs: list[int] = []
    for batch in loader:
        meta_idx = batch.pop("meta_idx")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu()
        scores.append(probs)
        labels.append(batch["labels"].detach().cpu())
        idxs.extend(meta_idx.tolist())
    return torch.cat(scores), torch.cat(labels), idxs


def metrics_at_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    idxs: list[int],
    items: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    preds = (scores >= threshold).long()
    out: dict[str, Any] = binary_metrics(preds, labels)
    out["threshold"] = threshold
    out["count"] = len(labels)
    out["by_pair_kind"] = grouped_metrics(preds, labels, idxs, items, "pair_kind")
    out["by_entity_type"] = grouped_metrics(preds, labels, idxs, items, "entity_type")
    return out


def best_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    idxs: list[int],
    items: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for i in range(1, 100):
        threshold = i / 100
        current = metrics_at_threshold(scores, labels, idxs, items, threshold)
        if best is None or current[metric] > best[metric]:
            best = current
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--metric", default="macro_f1")
    args = parser.parse_args()

    metadata = json.loads((args.run_dir / "training_metadata.json").read_text(encoding="utf-8"))
    input_mode = metadata.get("input_mode", "marked_pair")
    max_length = int(metadata.get("max_length", 512))
    pad_to_multiple_of = int(metadata.get("pad_to_multiple_of", 8))
    pad_to_max_length = bool(metadata.get("pad_to_max_length", True))

    val_items = load_jsonl(args.run_dir / "val_split.jsonl")
    test_items = load_jsonl(args.run_dir / "test_split.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    device = pick_device(args.device)
    model.to(device)

    def make_loader(items: list[dict[str, Any]]) -> DataLoader:
        ds = PairDataset(items, tokenizer, max_length, input_mode)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda b: collate(
                tokenizer,
                b,
                pad_to_multiple_of,
                max_length if pad_to_max_length else None,
            ),
        )

    val_scores, val_labels, val_idxs = collect_scores(model, make_loader(val_items), device)
    test_scores, test_labels, test_idxs = collect_scores(model, make_loader(test_items), device)

    val_default = metrics_at_threshold(val_scores, val_labels, val_idxs, val_items, 0.5)
    val_best = best_threshold(val_scores, val_labels, val_idxs, val_items, args.metric)
    test_default = metrics_at_threshold(test_scores, test_labels, test_idxs, test_items, 0.5)
    test_at_val_best = metrics_at_threshold(
        test_scores,
        test_labels,
        test_idxs,
        test_items,
        float(val_best["threshold"]),
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "run_dir": str(args.run_dir),
        "metric": args.metric,
        "val_default": val_default,
        "val_best": val_best,
        "test_default": test_default,
        "test_at_val_best": test_at_val_best,
    }
    output_json = args.output_json or (args.run_dir / "threshold_calibration.json")
    write_json(output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
