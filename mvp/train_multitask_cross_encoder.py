#!/usr/bin/env python3
"""Train one encoder for preservation, bidirectional NLI, relation, and ranking.

Directional losses are masked per row. The preservation head can therefore use
the full binary corpus while NLI heads only consume genuinely annotated rows.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train_cross_encoder import (
    SPECIAL_TOKENS,
    RankingDataset,
    binary_metrics,
    collate_ranking,
    dedupe_rows,
    load_jsonl,
    normalize_row,
    norm_key,
    pick_device,
    probability_metrics,
    encode_text_pair,
)


NLI_LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}
NLI_NAMES = ["entailment", "neutral", "contradiction"]
RELATION_LABELS = {
    "equivalence": 0,
    "generalization": 1,
    "specialization": 2,
    "substitution": 3,
    "contradiction": 4,
}
RELATION_NAMES = ["equivalence", "generalization", "specialization", "substitution", "contradiction"]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def relation_from_directions(forward: str, backward: str) -> str:
    if "contradiction" in {forward, backward}:
        return "contradiction"
    if forward == "entailment" and backward == "entailment":
        return "equivalence"
    if forward == "entailment":
        return "generalization"
    if backward == "entailment":
        return "specialization"
    return "substitution"


def raw_text_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    left = row.get("left") or row.get("original_context") or ""
    right = row.get("right") or row.get("candidate_context") or ""
    return (
        norm_key(left),
        norm_key(right),
        norm_key(row.get("entity", "")),
        norm_key(row.get("candidate", "")),
    )


def directional_payload(row: dict[str, Any], min_confidence: float) -> dict[str, Any] | None:
    confidence = float(row.get("directional_confidence", row.get("confidence", 0.0)) or 0.0)
    if confidence < min_confidence:
        return None
    forward = row.get("directional_a_to_b", row.get("a_to_b"))
    backward = row.get("directional_b_to_a", row.get("b_to_a"))
    if forward not in NLI_LABELS or backward not in NLI_LABELS:
        return None
    derived = relation_from_directions(str(forward), str(backward))
    return {
        "directional_a_to_b": forward,
        "directional_b_to_a": backward,
        "directional_relation": derived,
        "directional_relation_derived": derived,
        "directional_confidence": confidence,
    }


class DirectionalOverlay:
    def __init__(self, paths: list[Path], min_confidence: float):
        self.by_text: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.conflicted_texts: set[tuple[str, str, str, str]] = set()
        self.loaded = 0
        self.conflicts = 0
        for path in paths:
            for row in load_jsonl(path):
                payload = directional_payload(row, min_confidence)
                if payload is None:
                    continue
                self.loaded += 1
                self._insert(self.by_text, self.conflicted_texts, raw_text_key(row), payload)

    def _insert(self, target, conflicted: set[Any], key: Any, payload: dict[str, Any]) -> None:
        if key in conflicted:
            return
        previous = target.get(key)
        if previous and (
            previous["directional_a_to_b"], previous["directional_b_to_a"]
        ) != (payload["directional_a_to_b"], payload["directional_b_to_a"]):
            self.conflicts += 1
            target.pop(key, None)
            conflicted.add(key)
            return
        target[key] = payload

    def find(self, row: dict[str, Any]) -> dict[str, Any] | None:
        return self.by_text.get(raw_text_key(row))


def prepare_rows(path: Path, overlay: DirectionalOverlay, min_confidence: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in load_jsonl(path):
        payload = directional_payload(source, min_confidence) or overlay.find(source)
        row = normalize_row(source)
        if row is None:
            continue
        row["nli_a_to_b"] = NLI_LABELS[str(payload["directional_a_to_b"])] if payload else -100
        row["nli_b_to_a"] = NLI_LABELS[str(payload["directional_b_to_a"])] if payload else -100
        row["relation_label"] = RELATION_LABELS[str(payload["directional_relation"])] if payload else -100
        row["directional_confidence"] = float(payload["directional_confidence"]) if payload else 0.0
        row["supervision_quality"] = str(source.get("supervision_quality", "audited_or_consensus"))
        output.append(row)
    return dedupe_rows(output)


class MultiTaskDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int, input_mode: str, weak_weight: float):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_mode = input_mode
        self.weak_weight = weak_weight

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        encoded = encode_text_pair(self.tokenizer, row, self.input_mode, self.max_length)
        encoded["preserve_label"] = int(row["label"])
        encoded["nli_a_to_b"] = int(row["nli_a_to_b"])
        encoded["nli_b_to_a"] = int(row["nli_b_to_a"])
        encoded["relation_label"] = int(row["relation_label"])
        encoded["directional_confidence"] = float(row["directional_confidence"])
        encoded["sample_weight"] = (
            self.weak_weight if row.get("supervision_quality") == "weak_or_rule_based" else 1.0
        )
        return encoded


def collate_multitask(tokenizer, batch: list[dict[str, Any]], pad_to_multiple_of: int):
    scalar_keys = (
        "preserve_label",
        "nli_a_to_b",
        "nli_b_to_a",
        "relation_label",
        "directional_confidence",
        "sample_weight",
    )
    scalars = {key: [item.pop(key) for item in batch] for key in scalar_keys}
    result = tokenizer.pad(batch, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
    for key in scalar_keys[:4]:
        result[key] = torch.tensor(scalars[key], dtype=torch.long)
    result["directional_confidence"] = torch.tensor(scalars["directional_confidence"], dtype=torch.float32)
    result["sample_weight"] = torch.tensor(scalars["sample_weight"], dtype=torch.float32)
    return result


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(hidden))


class MultiHeadSubstitutionModel(nn.Module):
    def __init__(self, base_model: str, dropout: float = 0.1):
        super().__init__()
        local = Path(base_model).exists()
        config = AutoConfig.from_pretrained(base_model, local_files_only=local)
        self.encoder = AutoModel.from_pretrained(base_model, config=config, local_files_only=local)
        hidden = int(config.hidden_size)
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden, elementwise_affine=True, bias=False),
        )
        self.preserve_head = ProjectionHead(hidden, 2, dropout)
        self.a_to_b_head = ProjectionHead(hidden, 3, dropout)
        self.b_to_a_head = ProjectionHead(hidden, 3, dropout)
        self.relation_head = ProjectionHead(hidden, 5, dropout)

    def resize_token_embeddings(self, size: int) -> None:
        self.encoder.resize_token_embeddings(size)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        encoded = self.encoder(**inputs)
        hidden = self.shared(encoded.last_hidden_state[:, 0])
        return {
            "preserve": self.preserve_head(hidden),
            "a_to_b": self.a_to_b_head(hidden),
            "b_to_a": self.b_to_a_head(hidden),
            "relation": self.relation_head(hidden),
        }

    def head_state_dict(self) -> dict[str, Any]:
        return {
            "shared": self.shared.state_dict(),
            "preserve_head": self.preserve_head.state_dict(),
            "a_to_b_head": self.a_to_b_head.state_dict(),
            "b_to_a_head": self.b_to_a_head.state_dict(),
            "relation_head": self.relation_head.state_dict(),
        }

    def load_head_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("shared", "preserve_head", "a_to_b_head", "b_to_a_head", "relation_head"):
            getattr(self, name).load_state_dict(state[name])


def masked_cross_entropy(logits, labels, confidence: torch.Tensor | None = None) -> torch.Tensor:
    mask = labels != -100
    if not mask.any():
        return logits.sum() * 0.0
    losses = F.cross_entropy(logits[mask], labels[mask], reduction="none")
    if confidence is not None:
        weights = confidence[mask].clamp(min=0.1)
        return (losses * weights).sum() / weights.sum()
    return losses.mean()


def multitask_loss(outputs, batch, args, preserve_weights=None) -> tuple[torch.Tensor, dict[str, float]]:
    preserve_raw = F.cross_entropy(
        outputs["preserve"], batch["preserve_label"], weight=preserve_weights, reduction="none"
    )
    preserve = (preserve_raw * batch["sample_weight"]).sum() / batch["sample_weight"].sum().clamp(min=1e-6)
    forward = masked_cross_entropy(outputs["a_to_b"], batch["nli_a_to_b"], batch["directional_confidence"])
    backward = masked_cross_entropy(outputs["b_to_a"], batch["nli_b_to_a"], batch["directional_confidence"])
    relation = masked_cross_entropy(outputs["relation"], batch["relation_label"], batch["directional_confidence"])

    both = (batch["nli_a_to_b"] != -100) & (batch["nli_b_to_a"] != -100)
    if both.any():
        preserve_probability = torch.softmax(outputs["preserve"][both], dim=-1)[:, 1]
        mutual_entailment = (
            torch.softmax(outputs["a_to_b"][both], dim=-1)[:, NLI_LABELS["entailment"]]
            * torch.softmax(outputs["b_to_a"][both], dim=-1)[:, NLI_LABELS["entailment"]]
        )
        consistency = F.mse_loss(preserve_probability, mutual_entailment)
    else:
        consistency = outputs["preserve"].sum() * 0.0
    total = (
        preserve
        + args.nli_weight * (forward + backward)
        + args.relation_weight * relation
        + args.consistency_weight * consistency
    )
    values = {
        "preserve": float(preserve.detach()),
        "a_to_b": float(forward.detach()),
        "b_to_a": float(backward.detach()),
        "relation": float(relation.detach()),
        "consistency": float(consistency.detach()),
    }
    return total, values


def multiclass_metrics(predictions: torch.Tensor, labels: torch.Tensor, names: list[str]) -> dict[str, Any]:
    valid = labels != -100
    if not valid.any():
        return {"count": 0}
    predictions, labels = predictions[valid], labels[valid]
    matrix = torch.zeros((len(names), len(names)), dtype=torch.long)
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[truth, prediction] += 1
    per_class = {}
    f1_values = []
    for idx, name in enumerate(names):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum()) - tp
        fn = int(matrix[idx, :].sum()) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[idx].sum())}
    return {
        "count": int(valid.sum()),
        "accuracy": float((predictions == labels).float().mean()),
        "macro_f1": sum(f1_values) / len(f1_values),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(model, loader, device, min_directional_selection_count: int = 20) -> dict[str, Any]:
    model.eval()
    collected = {key: [] for key in ("preserve", "a_to_b", "b_to_a", "relation")}
    labels = {key: [] for key in ("preserve", "a_to_b", "b_to_a", "relation")}
    for batch in loader:
        model_inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key not in {
                "preserve_label", "nli_a_to_b", "nli_b_to_a", "relation_label",
                "directional_confidence", "sample_weight",
            }
        }
        outputs = model(**model_inputs)
        for key in collected:
            collected[key].append(outputs[key].cpu())
        labels["preserve"].append(batch["preserve_label"])
        labels["a_to_b"].append(batch["nli_a_to_b"])
        labels["b_to_a"].append(batch["nli_b_to_a"])
        labels["relation"].append(batch["relation_label"])
    if not collected["preserve"]:
        return {"count": 0}
    logits = {key: torch.cat(value) for key, value in collected.items()}
    truth = {key: torch.cat(value) for key, value in labels.items()}
    preserve_predictions = logits["preserve"].argmax(dim=-1)
    preserve = binary_metrics(preserve_predictions, truth["preserve"])
    preserve.update(probability_metrics(torch.softmax(logits["preserve"], dim=-1)[:, 1], truth["preserve"]))
    result = {
        "count": len(truth["preserve"]),
        "preserve": preserve,
        "a_to_b": multiclass_metrics(logits["a_to_b"].argmax(dim=-1), truth["a_to_b"], NLI_NAMES),
        "b_to_a": multiclass_metrics(logits["b_to_a"].argmax(dim=-1), truth["b_to_a"], NLI_NAMES),
        "relation": multiclass_metrics(logits["relation"].argmax(dim=-1), truth["relation"], RELATION_NAMES),
    }
    directional_scores = [
        result[key].get("macro_f1")
        for key in ("a_to_b", "b_to_a")
        if result[key].get("count", 0) >= min_directional_selection_count
    ]
    result["selection_score"] = preserve["macro_f1"] + 0.2 * (
        sum(directional_scores) / len(directional_scores) if directional_scores else 0.0
    )
    return result


def make_loader(rows, tokenizer, args, shuffle: bool):
    dataset = MultiTaskDataset(rows, tokenizer, args.max_length, args.input_mode, args.weak_weight)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_multitask(tokenizer, batch, args.pad_to_multiple_of),
    )


def save_checkpoint(model, tokenizer, path: Path, metadata: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(model.head_state_dict(), path / "multitask_heads.pt")
    write_json(path / "multitask_config.json", metadata)


def load_checkpoint(path: Path, device: torch.device) -> MultiHeadSubstitutionModel:
    metadata = json.loads((path / "multitask_config.json").read_text(encoding="utf-8"))
    model = MultiHeadSubstitutionModel(str(path), dropout=float(metadata["dropout"]))
    model.load_head_state_dict(torch.load(path / "multitask_heads.pt", map_location="cpu", weights_only=True))
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--directional-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--ranking-jsonl", type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-mode", choices=["pair", "marked_pair", "masked_query", "entity_query"], default="marked_pair")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ranking-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--nli-weight", type=float, default=0.3)
    parser.add_argument("--relation-weight", type=float, default=0.15)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=0.3)
    parser.add_argument("--weak-weight", type=float, default=0.4)
    parser.add_argument("--min-directional-confidence", type=float, default=0.8)
    parser.add_argument("--min-directional-selection-count", type=int, default=20)
    parser.add_argument("--no-weighted-loss", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    device = pick_device(args.device)

    overlay = DirectionalOverlay(args.directional_jsonl, args.min_directional_confidence)
    train_rows = prepare_rows(args.train, overlay, args.min_directional_confidence)
    val_rows = prepare_rows(args.val, overlay, args.min_directional_confidence)
    test_rows = prepare_rows(args.test, overlay, args.min_directional_confidence)
    if args.max_train_pairs:
        train_rows = train_rows[: args.max_train_pairs]
    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError("Train, val, and test files must contain binary substitution rows")

    local = Path(args.base_model).exists()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=local)
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model = MultiHeadSubstitutionModel(args.base_model, args.dropout)
    if added:
        model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    model.to(device)

    train_loader = make_loader(train_rows, tokenizer, args, True)
    val_loader = make_loader(val_rows, tokenizer, args, False)
    test_loader = make_loader(test_rows, tokenizer, args, False)

    ranking_loader = None
    ranking_items = []
    if args.ranking_jsonl and args.ranking_weight > 0:
        for source in load_jsonl(args.ranking_jsonl):
            positive = normalize_row(source.get("positive", {}))
            negative = normalize_row(source.get("negative", {}))
            if positive and negative and positive["label"] == 1 and negative["label"] == 0:
                ranking_items.append({"positive": positive, "negative": negative})
        if ranking_items:
            ranking_dataset = RankingDataset(ranking_items, tokenizer, args.max_length, args.input_mode)
            ranking_loader = DataLoader(
                ranking_dataset,
                batch_size=args.ranking_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=lambda batch: collate_ranking(tokenizer, batch, args.pad_to_multiple_of),
            )

    preserve_weights = None
    if not args.no_weighted_loss:
        counts = Counter(int(row["label"]) for row in train_rows)
        total = len(train_rows)
        preserve_weights = torch.tensor(
            [total / max(2 * counts.get(label, 1), 1) for label in (0, 1)], dtype=torch.float32, device=device
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(total_updates * args.warmup_ratio)), max(total_updates, 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_model": args.base_model,
        "dropout": args.dropout,
        "input_mode": args.input_mode,
        "truncation_strategy": "target_centered_marked_pair_v1",
        "max_length": args.max_length,
        "nli_labels": NLI_LABELS,
        "relation_labels": RELATION_LABELS,
        "weights": {
            "nli": args.nli_weight,
            "relation": args.relation_weight,
            "consistency": args.consistency_weight,
            "ranking": args.ranking_weight,
            "weak": args.weak_weight,
        },
        "rows": {
            "train": len(train_rows), "val": len(val_rows), "test": len(test_rows),
            "ranking": len(ranking_items), "directional_overlay": overlay.loaded,
        },
        "directional_conflicts": overlay.conflicts,
        "device": str(device),
        "precision": args.precision,
    }
    write_json(args.output_dir / "training_metadata.json", metadata)

    best_score = -1.0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        ranking_iterator = iter(ranking_loader) if ranking_loader else None
        progress = tqdm(train_loader, desc=f"multitask epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            labels = {
                key: batch.pop(key).to(device)
                for key in (
                    "preserve_label", "nli_a_to_b", "nli_b_to_a", "relation_label",
                    "directional_confidence", "sample_weight",
                )
            }
            model_inputs = {key: value.to(device) for key, value in batch.items()}
            autocast_enabled = device.type == "cuda" and args.precision != "fp32"
            autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = model(**model_inputs)
                loss, loss_values = multitask_loss(outputs, labels, args, preserve_weights)
                rank_loss = outputs["preserve"].sum() * 0.0
                if ranking_iterator is not None:
                    try:
                        ranking_batch = next(ranking_iterator)
                    except StopIteration:
                        ranking_iterator = iter(ranking_loader)
                        ranking_batch = next(ranking_iterator)
                    positive = {key: value.to(device) for key, value in ranking_batch["positive"].items()}
                    negative = {key: value.to(device) for key, value in ranking_batch["negative"].items()}
                    positive_logits = model(**positive)["preserve"]
                    negative_logits = model(**negative)["preserve"]
                    positive_score = positive_logits[:, 1] - positive_logits[:, 0]
                    negative_score = negative_logits[:, 1] - negative_logits[:, 0]
                    rank_loss = F.margin_ranking_loss(
                        positive_score, negative_score, torch.ones_like(positive_score), margin=args.ranking_margin
                    )
                    loss = loss + args.ranking_weight * rank_loss
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(
                preserve=f"{loss_values['preserve']:.3f}", nli=f"{loss_values['a_to_b'] + loss_values['b_to_a']:.3f}",
                rank=f"{float(rank_loss.detach()):.3f}",
            )

        val_metrics = evaluate(model, val_loader, device, args.min_directional_selection_count)
        val_metrics["epoch"] = epoch
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as target:
            target.write(json.dumps(val_metrics, ensure_ascii=False) + "\n")
        print(json.dumps(val_metrics, ensure_ascii=False, indent=2), flush=True)
        if val_metrics["selection_score"] > best_score:
            best_score = val_metrics["selection_score"]
            save_checkpoint(model, tokenizer, args.output_dir / "best", metadata)

    best_model = load_checkpoint(args.output_dir / "best", device)
    test_metrics = evaluate(best_model, test_loader, device, args.min_directional_selection_count)
    write_json(args.output_dir / "test_metrics.json", test_metrics)
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"Multi-head training complete: {args.output_dir}")


if __name__ == "__main__":
    main()
