#!/usr/bin/env python3
"""Score JSONL substitution pairs with a trained cross-encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PairDataset(Dataset):
    def __init__(self, items, tokenizer, max_length: int):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return self.tokenizer(
            item["left"],
            item["right"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    items = load_jsonl(args.input_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = pick_device(args.device)
    model.to(device)
    model.eval()

    dataset = PairDataset(items, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: tokenizer.pad(b, padding=True, return_tensors="pt"),
    )

    scores: list[float] = []
    preds: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="score"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
            preds.extend((probs >= 0.5).long().cpu().tolist())

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    labeled = 0
    with args.output_jsonl.open("w", encoding="utf-8") as fh:
        for item, score, pred in zip(items, scores, preds):
            item = dict(item)
            item["score"] = float(score)
            item["prediction"] = int(pred)
            if "label" in item:
                labeled += 1
                correct += int(int(item["label"]) == pred)
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    if labeled:
        print(f"accuracy={correct / labeled:.4f} labeled={labeled}")
    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
