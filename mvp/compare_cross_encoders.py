#!/usr/bin/env python3
"""Compare two cross-encoders on a shared entity-disjoint evaluation set."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_cross_encoder import PairDataset, binary_metrics, collate, load_jsonl, pick_device


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def entity_ids(path: Path) -> set[str]:
    return {str(row.get("entity_id")) for row in load_jsonl(path) if row.get("entity_id")}


def input_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")).strip().casefold() for key in ("left", "right", "entity", "candidate"))


def shared_unseen_rows(old_run: Path, new_run: Path) -> list[dict[str, Any]]:
    old_train_entities = entity_ids(old_run / "train_split.jsonl")
    new_train_entities = entity_ids(new_run / "train_split.jsonl")

    candidates: list[dict[str, Any]] = []
    for source_name, path, blocked in (
        ("new_test", new_run / "test_split.jsonl", old_train_entities),
        ("old_test", old_run / "test_split.jsonl", new_train_entities),
    ):
        for row in load_jsonl(path):
            entity_id = str(row.get("entity_id", ""))
            if not entity_id or entity_id in blocked:
                continue
            candidates.append({**row, "comparison_source": source_name})

    seen: set[tuple[str, ...]] = set()
    result: list[dict[str, Any]] = []
    for row in candidates:
        key = input_key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


@torch.no_grad()
def score_model(
    checkpoint: Path,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_length: int,
    input_mode: str,
) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()
    dataset = PairDataset(rows, tokenizer, max_length, input_mode)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate(tokenizer, batch, 8),
    )
    scores: list[float] = []
    for batch in loader:
        batch.pop("meta_idx")
        batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def rounded_metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    metrics = binary_metrics(torch.tensor(predictions), torch.tensor(labels))
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()}


def grouped_metrics(rows: list[dict[str, Any]], pred_key: str, field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "unknown")].append(row)
    result: dict[str, Any] = {}
    for name, items in sorted(groups.items()):
        if len(items) < 10:
            continue
        metrics = rounded_metrics([int(row[pred_key]) for row in items], [int(row["label"]) for row in items])
        result[name] = {"count": len(items), **metrics}
    return result


def bootstrap_delta(rows: list[dict[str, Any]], iterations: int, seed: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row.get("entity_id") or f"row-{idx}")].append(row)
    names = list(groups)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sample = [row for _ in names for row in groups[rng.choice(names)]]
        labels = [int(row["label"]) for row in sample]
        old_f1 = rounded_metrics([int(row["old_prediction"]) for row in sample], labels)["macro_f1"]
        new_f1 = rounded_metrics([int(row["new_prediction"]) for row in sample], labels)["macro_f1"]
        deltas.append(float(new_f1) - float(old_f1))
    deltas.sort()
    lo = deltas[int(iterations * 0.025)]
    hi = deltas[min(int(iterations * 0.975), iterations - 1)]
    return {
        "iterations": iterations,
        "groups": len(names),
        "mean_macro_f1_delta": round(sum(deltas) / len(deltas), 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "probability_new_better": round(sum(delta > 0 for delta in deltas) / len(deltas), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-checkpoint", required=True, type=Path)
    parser.add_argument("--new-checkpoint", required=True, type=Path)
    parser.add_argument("--old-run", required=True, type=Path)
    parser.add_argument("--new-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--old-input-mode", choices=["pair", "marked_pair", "entity_query", "masked_query"], default="marked_pair"
    )
    parser.add_argument(
        "--new-input-mode", choices=["pair", "marked_pair", "entity_query", "masked_query"], default="marked_pair"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    rows = shared_unseen_rows(args.old_run, args.new_run)
    if not rows:
        raise RuntimeError("No shared unseen evaluation rows were found.")
    device = pick_device(args.device)
    old_scores = score_model(args.old_checkpoint, rows, device, args.batch_size, args.max_length, args.old_input_mode)
    new_scores = score_model(args.new_checkpoint, rows, device, args.batch_size, args.max_length, args.new_input_mode)

    outcomes: Counter[str] = Counter()
    for row, old_score, new_score in zip(rows, old_scores, new_scores):
        label = int(row["label"])
        old_pred = int(old_score >= 0.5)
        new_pred = int(new_score >= 0.5)
        old_correct = old_pred == label
        new_correct = new_pred == label
        if new_correct and not old_correct:
            outcome = "new_fixed"
        elif old_correct and not new_correct:
            outcome = "new_regressed"
        elif new_correct:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"
        row.update(
            old_score=round(old_score, 6),
            new_score=round(new_score, 6),
            old_prediction=old_pred,
            new_prediction=new_pred,
            comparison_outcome=outcome,
        )
        outcomes[outcome] += 1

    labels = [int(row["label"]) for row in rows]
    old_metrics = rounded_metrics([int(row["old_prediction"]) for row in rows], labels)
    new_metrics = rounded_metrics([int(row["new_prediction"]) for row in rows], labels)
    summary = {
        "rows": len(rows),
        "old_input_mode": args.old_input_mode,
        "new_input_mode": args.new_input_mode,
        "entities": len({str(row.get("entity_id")) for row in rows}),
        "labels": dict(Counter(labels)),
        "sources": dict(Counter(str(row.get("comparison_source")) for row in rows)),
        "outcomes": dict(outcomes),
        "old": old_metrics,
        "new": new_metrics,
        "macro_f1_delta": round(new_metrics["macro_f1"] - old_metrics["macro_f1"], 6),
        "bootstrap": bootstrap_delta(rows, args.bootstrap_iterations, args.seed),
        "old_by_type": grouped_metrics(rows, "old_prediction", "entity_type"),
        "new_by_type": grouped_metrics(rows, "new_prediction", "entity_type"),
        "old_by_kind": grouped_metrics(rows, "old_prediction", "pair_kind"),
        "new_by_kind": grouped_metrics(rows, "new_prediction", "pair_kind"),
    }
    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
