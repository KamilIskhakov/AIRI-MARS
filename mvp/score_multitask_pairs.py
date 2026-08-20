#!/usr/bin/env python3
"""Score substitution pairs with an offline multi-head checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from train_cross_encoder import materialize_contexts, pick_device, text_fields
from train_multitask_cross_encoder import (
    NLI_NAMES,
    RELATION_NAMES,
    load_checkpoint,
    relation_from_directions,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def scoring_row(source: dict[str, Any]) -> dict[str, Any] | None:
    left = source.get("left") or source.get("original_context")
    right = source.get("right") or source.get("candidate_context")
    entity = str(source.get("entity", ""))
    candidate = str(source.get("candidate", ""))
    if not left or not right or not entity or not candidate:
        return None
    model_left, model_right, materialization = materialize_contexts(
        str(left), str(right), entity, candidate, source.get("mask_idx")
    )
    return {
        "left": str(left),
        "right": str(right),
        "model_left": model_left,
        "model_right": model_right,
        "context_materialization": materialization,
        "entity": entity,
        "candidate": candidate,
    }


class ScoringDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int, input_mode: str):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_mode = input_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        left, right = text_fields(self.rows[idx], self.input_mode)
        encoded = self.tokenizer(left, right, truncation=True, max_length=self.max_length, padding=False)
        encoded["row_idx"] = idx
        return encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Override checkpoint sequence length; 0 reuses its training value.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--changed-max", type=float, default=0.4)
    parser.add_argument("--preserved-min", type=float, default=0.6)
    args = parser.parse_args()
    if not 0 <= args.changed_max < args.preserved_min <= 1:
        raise ValueError("Expected 0 <= changed-max < preserved-min <= 1")

    metadata = json.loads((args.checkpoint / "multitask_config.json").read_text(encoding="utf-8"))
    input_mode = str(metadata.get("input_mode", "marked_pair"))
    max_length = args.max_length or int(metadata.get("max_length", 512))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    device = pick_device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    model.eval()

    sources = load_jsonl(args.input)
    normalized: list[dict[str, Any]] = []
    source_indexes: list[int] = []
    for idx, source in enumerate(sources):
        row = scoring_row(source)
        if row is not None:
            normalized.append(row)
            source_indexes.append(idx)
    dataset = ScoringDataset(normalized, tokenizer, max_length, input_mode)

    def collate(batch):
        indexes = torch.tensor([item.pop("row_idx") for item in batch], dtype=torch.long)
        encoded = tokenizer.pad(batch, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        encoded["row_idx"] = indexes
        return encoded

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    output = [dict(row) for row in sources]
    label_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    with torch.inference_mode():
        for batch in loader:
            row_indexes = batch.pop("row_idx").tolist()
            scores = model(**{key: value.to(device) for key, value in batch.items()})
            preserve_probs = torch.softmax(scores["preserve"], dim=-1).cpu()
            forward_probs = torch.softmax(scores["a_to_b"], dim=-1).cpu()
            backward_probs = torch.softmax(scores["b_to_a"], dim=-1).cpu()
            relation_probs = torch.softmax(scores["relation"], dim=-1).cpu()
            for batch_idx, normalized_idx in enumerate(row_indexes):
                source_idx = source_indexes[normalized_idx]
                preserve_probability = float(preserve_probs[batch_idx, 1])
                if preserve_probability >= args.preserved_min:
                    label = "preserved"
                elif preserve_probability <= args.changed_max:
                    label = "changed"
                else:
                    label = "uncertain"
                forward = NLI_NAMES[int(forward_probs[batch_idx].argmax())]
                backward = NLI_NAMES[int(backward_probs[batch_idx].argmax())]
                derived_relation = relation_from_directions(forward, backward)
                direct_relation = RELATION_NAMES[int(relation_probs[batch_idx].argmax())]
                output[source_idx].update(
                    multitask_preserved_probability=round(preserve_probability, 6),
                    multitask_label=label,
                    multitask_a_to_b=forward,
                    multitask_a_to_b_probabilities={
                        name: round(float(forward_probs[batch_idx, pos]), 6)
                        for pos, name in enumerate(NLI_NAMES)
                    },
                    multitask_b_to_a=backward,
                    multitask_b_to_a_probabilities={
                        name: round(float(backward_probs[batch_idx, pos]), 6)
                        for pos, name in enumerate(NLI_NAMES)
                    },
                    multitask_relation=derived_relation,
                    multitask_relation_head=direct_relation,
                    multitask_relation_agreement=derived_relation == direct_relation,
                    multitask_context_materialization=normalized[normalized_idx]["context_materialization"],
                    multitask_checkpoint=str(args.checkpoint),
                )
                label_counts[label] += 1
                relation_counts[derived_relation] += 1

    write_jsonl(args.out, output)
    summary = {
        "input_rows": len(sources),
        "scored_rows": len(normalized),
        "skipped_rows": len(sources) - len(normalized),
        "labels": dict(label_counts),
        "relations": dict(relation_counts),
        "thresholds": {"changed_max": args.changed_max, "preserved_min": args.preserved_min},
        "input_mode": input_mode,
        "max_length": max_length,
        "checkpoint": str(args.checkpoint),
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
