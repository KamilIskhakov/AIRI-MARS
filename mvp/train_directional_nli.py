#!/usr/bin/env python3
"""Fine-tune ModernBERT on ordered entity-substitution NLI examples."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
except ImportError as exc:
    raise RuntimeError("scikit-learn is required for directional NLI metrics") from exc


LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") in LABELS and row.get("premise") and row.get("hypothesis"):
                rows.append(row)
    return rows


class NLIDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        item = self.tokenizer(
            row["premise"], row["hypothesis"], truncation=True, max_length=self.max_length, padding=False
        )
        item["labels"] = LABELS[row["label"]]
        return item


def make_loader(rows, tokenizer, max_length, batch_size, workers, shuffle):
    dataset = NLIDataset(rows, tokenizer, max_length)

    def collate(batch):
        labels = torch.tensor([item.pop("labels") for item in batch], dtype=torch.long)
        result = tokenizer.pad(batch, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        result["labels"] = labels
        return result

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
    )


def amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, Any]:
    model.eval()
    logits_all, labels_all = [], []
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**{key: value for key, value in batch.items() if key != "labels"}).logits
        total_loss += F.cross_entropy(logits, batch["labels"], reduction="sum").item()
        total += batch["labels"].shape[0]
        logits_all.append(logits.cpu())
        labels_all.append(batch["labels"].cpu())
    if not logits_all:
        return {"count": 0}
    predictions = torch.cat(logits_all).argmax(dim=-1).numpy()
    truth = torch.cat(labels_all).numpy()
    names = ["entailment", "neutral", "contradiction"]
    return {
        "count": int(total),
        "loss": total_loss / max(total, 1),
        "macro_f1": float(f1_score(truth, predictions, average="macro")),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted")),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            truth, predictions, labels=[0, 1, 2], target_names=names, output_dict=True, zero_division=0
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This training entry point expects a CUDA GPU")
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    train_rows, val_rows, test_rows = load_rows(args.train), load_rows(args.val), load_rows(args.test)
    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError("Train, validation, and test NLI files must all be non-empty")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label={value: key for key, value in LABELS.items()},
        label2id=LABELS,
        ignore_mismatched_sizes=True,
    ).to(device)
    train_loader = make_loader(train_rows, tokenizer, args.max_length, args.batch_size, args.num_workers, True)
    val_loader = make_loader(val_rows, tokenizer, args.max_length, args.batch_size, args.num_workers, False)
    test_loader = make_loader(test_rows, tokenizer, args.max_length, args.batch_size, args.num_workers, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(total_updates * args.warmup_ratio)), total_updates
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "fp16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "training_metadata.json",
        {
            "base_model": args.base_model,
            "labels": LABELS,
            "train": len(train_rows),
            "validation": len(val_rows),
            "test": len(test_rows),
            "train_distribution": dict(Counter(row["label"] for row in train_rows)),
            "max_length": args.max_length,
            "precision": args.precision,
        },
    )

    best = -1.0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"nli epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with amp_context(device, args.precision):
                logits = model(**{key: value for key, value in batch.items() if key != "labels"}).logits
                loss = F.cross_entropy(logits, batch["labels"]) / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}")
        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            model.save_pretrained(args.output_dir / "best")
            tokenizer.save_pretrained(args.output_dir / "best")

    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir / "best").to(device)
    write_json(args.output_dir / "test_metrics.json", evaluate(best_model, test_loader, device))
    print(f"NLI training complete: {args.output_dir}")


if __name__ == "__main__":
    main()
