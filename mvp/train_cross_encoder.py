#!/usr/bin/env python3
"""Fine-tune a HF encoder as a contextual entity-substitution cross-encoder.

Input rows may be either:
- train-ready rows: left, right, label, entity, candidate, entity_id, pair_kind;
- judged/consensus rows: original_context, candidate_context, consensus_label/judge_label.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
except ImportError:  # Optional for minimal smoke environments.
    average_precision_score = balanced_accuracy_score = roc_auc_score = None


SPECIAL_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]", "[TGT]"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_label(row: dict[str, Any]) -> int | None:
    if "label" in row and row["label"] in {0, 1, "0", "1"}:
        return int(row["label"])
    label = row.get("consensus_label", row.get("judge_label"))
    if label == "preserved":
        return 1
    if label == "changed":
        return 0
    return None


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    label = normalize_label(row)
    if label is None:
        return None
    left = row.get("left") or row.get("original_context")
    right = row.get("right") or row.get("candidate_context")
    if not left or not right:
        return None
    model_left, model_right, materialization = materialize_contexts(
        str(left), str(right), str(row.get("entity", "")), str(row.get("candidate", "")), row.get("mask_idx")
    )
    return {
        "left": str(left),
        "right": str(right),
        "model_left": model_left,
        "model_right": model_right,
        "context_materialization": materialization,
        "label": label,
        "entity": str(row.get("entity", "")),
        "candidate": str(row.get("candidate", "")),
        "entity_id": "" if row.get("entity_id") is None else str(row.get("entity_id", "")),
        "entity_type": str(row.get("entity_type", row.get("fine_type", ""))),
        "coarse_group": str(row.get("coarse_group", "")),
        "pair_kind": str(row.get("pair_kind", row.get("candidate_kind", ""))),
        "branch": str(row.get("branch", "")),
        "pair_id": str(row.get("pair_id", "")),
        "candidate_entity_id": "" if row.get("candidate_entity_id") is None else str(row.get("candidate_entity_id", "")),
        "source_id": "" if row.get("source_id") is None else str(row.get("source_id", "")),
        "text_id": "" if row.get("text_id") is None else str(row.get("text_id", "")),
        "mention_id": "" if row.get("mention_id") is None else str(row.get("mention_id", "")),
        "dataset": str(row.get("dataset", row.get("dataset_name", ""))),
        "domain": str(row.get("domain", "")),
        "annotation_source": str(row.get("annotation_source", "")),
    }


def replace_nth(text: str, needle: str, replacement: str, index: int) -> str | None:
    if not needle or index < 0:
        return None
    start = 0
    for _ in range(index + 1):
        pos = text.find(needle, start)
        if pos < 0:
            return None
        start = pos + len(needle)
    return text[: pos] + replacement + text[pos + len(needle) :]


def replace_first_case_insensitive(text: str, needle: str, replacement: str) -> str | None:
    if not needle:
        return None
    match = re.search(re.escape(needle), text, flags=re.IGNORECASE)
    if not match:
        return None
    return text[: match.start()] + replacement + text[match.end() :]


def replace_matching_occurrence(text: str, needle: str, replacement: str, target: str) -> str | None:
    if not needle:
        return None
    for match in re.finditer(re.escape(needle), text, flags=re.IGNORECASE):
        rebuilt = text[: match.start()] + replacement + text[match.end() :]
        if norm_key(rebuilt) == norm_key(target):
            return rebuilt
    return None


def materialize_contexts(
    left: str, right: str, entity: str, candidate: str, mask_idx: Any
) -> tuple[str, str, str]:
    """Restore the target mention without confusing another same-surface occurrence.

    Dataset contexts may contain several masked mentions. The candidate side is
    the authoritative location: replacing each source-side mask with the
    candidate must reproduce it exactly. This is safer than searching for the
    entity string, which may occur elsewhere in the document.
    """
    if "<mask>" in left and candidate:
        mask_count = left.count("<mask>")
        for idx in range(mask_count):
            candidate_left = replace_nth(left, "<mask>", candidate, idx)
            if candidate_left is not None and norm_key(candidate_left) == norm_key(right):
                original_left = replace_nth(left, "<mask>", entity, idx) or left
                return original_left, right, f"mask_diff:{idx}"
        try:
            idx = int(mask_idx)
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < mask_count:
            original_left = replace_nth(left, "<mask>", entity, idx) or left
            return original_left, right, f"mask_idx_fallback:{idx}"

    if entity and candidate:
        rebuilt_left = replace_matching_occurrence(right, candidate, entity, left)
        if rebuilt_left is not None:
            return rebuilt_left, right, "candidate_diff"
        rebuilt_right = replace_matching_occurrence(left, entity, candidate, right)
        if rebuilt_right is not None:
            return left, rebuilt_right, "entity_diff"
        rebuilt_left = replace_first_case_insensitive(right, candidate, entity)
        if rebuilt_left is not None:
            return rebuilt_left, right, "candidate_reverse_replace"
    return left, right, "unresolved"


def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove identical model inputs and drop contradictory labels.

    The same entity/candidate substitution is intentionally retained when it
    occurs in another context: contextual variation is part of the task.
    """
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            norm_key(row.get("left")),
            norm_key(row.get("right")),
            norm_key(row.get("entity")),
            norm_key(row.get("candidate")),
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for items in grouped.values():
        if len({int(item["label"]) for item in items}) != 1:
            continue
        out.append(items[0])
    return out


def mark_first(text: str, needle: str, start: str, end: str) -> str:
    if not needle:
        return text
    pos = text.find(needle)
    if pos < 0:
        pos = text.casefold().find(needle.casefold())
    if pos < 0:
        return text
    return text[:pos] + start + " " + text[pos : pos + len(needle)] + " " + end + text[pos + len(needle) :]


def mask_first(text: str, needle: str, token: str) -> str:
    if not needle:
        return text
    pos = text.find(needle)
    if pos < 0:
        pos = text.casefold().find(needle.casefold())
    if pos < 0:
        return text
    return text[:pos] + token + text[pos + len(needle) :]


def text_fields(item: dict[str, Any], mode: str) -> tuple[str, str]:
    left_source = item.get("model_left") or item["left"]
    right_source = item.get("model_right") or item["right"]
    if mode == "pair":
        return left_source, right_source
    if mode == "marked_pair":
        left = mark_first(left_source, item.get("entity", ""), "[E1]", "[/E1]")
        right = mark_first(right_source, item.get("candidate", ""), "[E2]", "[/E2]")
        return left, right
    if mode == "entity_query":
        query = (
            f"Original entity: {item.get('entity', '')}\n"
            f"Candidate entity: {item.get('candidate', '')}\n"
            "Question: does replacing the original entity with the candidate preserve the factual meaning?"
        )
        return left_source, query
    if mode == "masked_query":
        # M1: single shared context with the original mention masked, plus the
        # original/candidate entity strings as a second segment. Unlike
        # marked_pair, the model never sees two near-duplicate full texts and
        # does not have to rediscover where the substitution happened.
        masked_context = mask_first(left_source, item.get("entity", ""), "[TGT]")
        query = f"Original entity: {item.get('entity', '')}\nCandidate entity: {item.get('candidate', '')}"
        return masked_context, query
    raise ValueError(f"Unknown input mode: {mode}")


def crop_around_marked_span(
    token_ids: list[int], start_id: int, end_id: int, budget: int
) -> list[int] | None:
    """Keep a marked entity span and balanced context within a token budget."""
    try:
        span_start = token_ids.index(start_id)
        span_end = token_ids.index(end_id, span_start + 1)
    except ValueError:
        return None
    span_size = span_end - span_start + 1
    if span_size > budget:
        return None
    if len(token_ids) <= budget:
        return token_ids

    context_budget = budget - span_size
    left_context = context_budget // 2
    window_start = max(0, span_start - left_context)
    window_end = min(len(token_ids), window_start + budget)
    window_start = max(0, window_end - budget)
    if not (window_start <= span_start and span_end < window_end):
        window_start = max(0, span_end + 1 - budget)
        window_end = min(len(token_ids), window_start + budget)
    return token_ids[window_start:window_end]


def encode_text_pair(tokenizer, item: dict[str, Any], mode: str, max_length: int) -> dict[str, Any]:
    """Tokenize a pair while guaranteeing marked spans survive truncation."""
    left, right = text_fields(item, mode)
    if mode != "marked_pair":
        return tokenizer(left, right, truncation=True, max_length=max_length, padding=False)

    special_count = tokenizer.num_special_tokens_to_add(pair=True)
    content_budget = max_length - special_count
    if content_budget < 8:
        raise ValueError(f"max_length={max_length} is too small for a marked text pair")
    left_budget = content_budget // 2
    right_budget = content_budget - left_budget
    left_ids = tokenizer(left, add_special_tokens=False, truncation=False)["input_ids"]
    right_ids = tokenizer(right, add_special_tokens=False, truncation=False)["input_ids"]
    left_cropped = crop_around_marked_span(
        left_ids, tokenizer.convert_tokens_to_ids("[E1]"), tokenizer.convert_tokens_to_ids("[/E1]"), left_budget
    )
    right_cropped = crop_around_marked_span(
        right_ids, tokenizer.convert_tokens_to_ids("[E2]"), tokenizer.convert_tokens_to_ids("[/E2]"), right_budget
    )
    if left_cropped is None or right_cropped is None:
        return tokenizer(left, right, truncation=True, max_length=max_length, padding=False)
    left_window = tokenizer.decode(left_cropped, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    right_window = tokenizer.decode(right_cropped, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return tokenizer(
        left_window,
        right_window,
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=max_length,
    )


class PairDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], tokenizer, max_length: int, mode: str):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        enc = encode_text_pair(self.tokenizer, item, self.mode, self.max_length)
        enc["labels"] = int(item["label"])
        enc["meta_idx"] = idx
        return enc


class RankingDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], tokenizer, max_length: int, mode: str):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self) -> int:
        return len(self.items)

    def encode(self, item: dict[str, Any]) -> dict[str, Any]:
        return encode_text_pair(self.tokenizer, item, self.mode, self.max_length)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        return {"positive": self.encode(item["positive"]), "negative": self.encode(item["negative"])}


def collate(
    tokenizer,
    batch: list[dict[str, Any]],
    pad_to_multiple_of: int | None,
    pad_to_max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    labels = torch.tensor([x.pop("labels") for x in batch], dtype=torch.long)
    meta_idx = torch.tensor([x.pop("meta_idx") for x in batch], dtype=torch.long)
    padded = tokenizer.pad(
        batch,
        padding="max_length" if pad_to_max_length else True,
        max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )
    padded["labels"] = labels
    padded["meta_idx"] = meta_idx
    return padded


def collate_ranking(
    tokenizer,
    batch: list[dict[str, Any]],
    pad_to_multiple_of: int | None,
    pad_to_max_length: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    def pad(key: str) -> dict[str, torch.Tensor]:
        return tokenizer.pad(
            [item[key] for item in batch],
            padding="max_length" if pad_to_max_length else True,
            max_length=pad_to_max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    return {"positive": pad("positive"), "negative": pad("negative")}


def grouped_split(
    rows: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        group = row.get("entity_id") or f"row-{idx}"
        by_group[str(group)].append(row)
    groups = list(by_group)
    rng.shuffle(groups)
    n = len(groups)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    test_groups = set(groups[:test_n])
    val_groups = set(groups[test_n : test_n + val_n])
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for group, items in by_group.items():
        if group in test_groups:
            test.extend(items)
        elif group in val_groups:
            val.extend(items)
        else:
            train.extend(items)
    return train, val, test


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            if self.size[left_root] < self.size[right_root]:
                left_root, right_root = right_root, left_root
            self.parent[right_root] = left_root
            self.size[left_root] += self.size[right_root]


def component_split(
    rows: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep every source entity and document in exactly one split."""
    dsu = DisjointSet()
    for idx, row in enumerate(rows):
        row_key = f"row:{idx}"
        entity = row.get("entity_id") or norm_key(row.get("entity")) or f"missing:{idx}"
        dsu.union(row_key, f"entity:{entity}")
        document = row.get("source_id") or row.get("text_id")
        if document:
            dsu.union(row_key, f"document:{document}")

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        components[dsu.find(f"row:{idx}")].append(row)

    rng = random.Random(seed)
    groups = list(components.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    targets = {
        "train": len(rows) * (1.0 - val_ratio - test_ratio),
        "val": len(rows) * val_ratio,
        "test": len(rows) * test_ratio,
    }
    assigned: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for group in groups:
        choices = sorted(
            assigned,
            key=lambda name: (
                len(assigned[name]) / max(targets[name], 1.0),
                {"test": 0, "val": 1, "train": 2}[name],
            ),
        )
        assigned[choices[0]].extend(group)
    return assigned["train"], assigned["val"], assigned["test"]


def split_random(
    rows: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    items = list(rows)
    rng.shuffle(items)
    test_n = int(len(items) * test_ratio)
    val_n = int(len(items) * val_ratio)
    test = items[:test_n]
    val = items[test_n : test_n + val_n]
    train = items[test_n + val_n :]
    return train, val, test


def binary_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    correct = (preds == labels).sum().item()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    neg_precision = tn / max(tn + fn, 1)
    neg_recall = tn / max(tn + fp, 1)
    neg_f1 = 2 * neg_precision * neg_recall / max(neg_precision + neg_recall, 1e-12)
    return {
        "accuracy": correct / max(len(labels), 1),
        "precision_pos": precision,
        "recall_pos": recall,
        "f1_pos": f1,
        "precision_neg": neg_precision,
        "recall_neg": neg_recall,
        "f1_neg": neg_f1,
        "macro_f1": (f1 + neg_f1) / 2,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def probability_metrics(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = probabilities.detach().cpu().float()
    truth = labels.detach().cpu().long()
    predictions = (probs >= 0.5).long()
    result: dict[str, float] = {
        "brier_score": float(torch.mean((probs - truth.float()) ** 2).item()),
    }
    bins = torch.linspace(0.0, 1.0, 11)
    ece = 0.0
    for idx in range(10):
        include_right = idx == 9
        mask = (probs >= bins[idx]) & ((probs <= bins[idx + 1]) if include_right else (probs < bins[idx + 1]))
        if mask.any():
            confidence = probs[mask].mean().item()
            accuracy = truth[mask].float().mean().item()
            ece += mask.float().mean().item() * abs(confidence - accuracy)
    result["ece_10"] = float(ece)
    if balanced_accuracy_score is not None:
        y_true = truth.numpy()
        y_prob = probs.numpy()
        y_pred = predictions.numpy()
        result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        if len(set(y_true.tolist())) == 2:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            result["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return result


@torch.no_grad()
def evaluate(model, loader, items: list[dict[str, Any]], device, loss_weights: torch.Tensor | None) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    logits_all = []
    labels_all = []
    idx_all = []
    for batch in loader:
        meta_idx = batch.pop("meta_idx")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**{k: v for k, v in batch.items() if k != "labels"})
        loss = F.cross_entropy(out.logits, batch["labels"], weight=loss_weights)
        total_loss += loss.item() * batch["labels"].shape[0]
        total += batch["labels"].shape[0]
        logits_all.append(out.logits.cpu())
        labels_all.append(batch["labels"].cpu())
        idx_all.extend(meta_idx.tolist())
    if not logits_all:
        return {"loss": 0.0, "count": 0, "by_pair_kind": {}, "by_entity_type": {}}
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    preds = logits.argmax(dim=-1)
    result: dict[str, Any] = binary_metrics(preds, labels)
    result.update(probability_metrics(torch.softmax(logits, dim=-1)[:, 1], labels))
    result["loss"] = total_loss / max(total, 1)
    result["count"] = len(labels)
    probabilities = torch.softmax(logits, dim=-1)[:, 1]
    result["by_pair_kind"] = grouped_metrics(preds, probabilities, labels, idx_all, items, "pair_kind")
    result["by_entity_type"] = grouped_metrics(preds, probabilities, labels, idx_all, items, "entity_type")
    result["by_coarse_group"] = grouped_metrics(preds, probabilities, labels, idx_all, items, "coarse_group")
    result["by_domain"] = grouped_metrics(preds, probabilities, labels, idx_all, items, "domain")
    return result


def grouped_metrics(
    preds: torch.Tensor,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    idxs: list[int],
    items: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for pos, idx in enumerate(idxs):
        buckets[str(items[idx].get(field, "unknown") or "unknown")].append(pos)
    out: dict[str, dict[str, float]] = {}
    for key, positions in buckets.items():
        if len(positions) < 5:
            continue
        pos_tensor = torch.tensor(positions, dtype=torch.long)
        metrics = binary_metrics(preds[pos_tensor], labels[pos_tensor])
        metrics.update(probability_metrics(probabilities[pos_tensor], labels[pos_tensor]))
        out[key] = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
        out[key]["count"] = len(positions)
    return out


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def class_weights(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    counts = Counter(int(row["label"]) for row in rows)
    total = max(sum(counts.values()), 1)
    weights = [
        total / max(2 * counts.get(0, 1), 1),
        total / max(2 * counts.get(1, 1), 1),
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def summarize_rows(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "rows": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "pair_kind": dict(Counter(row.get("pair_kind", "unknown") for row in rows).most_common(20)),
        "entity_type": dict(Counter(row.get("entity_type", "unknown") for row in rows).most_common(20)),
        "entities": len({row.get("entity_id") for row in rows}),
    }


def save_split(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--input-mode", choices=["pair", "marked_pair", "entity_query", "masked_query"], default="marked_pair"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split", choices=["entity_document", "entity", "random"], default="entity_document")
    parser.add_argument("--val-jsonl", type=Path)
    parser.add_argument("--test-jsonl", type=Path)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--weighted-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-splits", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--ranking-jsonl", type=Path)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-batch-size", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    raw_rows = load_jsonl(args.train_jsonl)
    rows = [row for source in raw_rows if (row := normalize_row(source)) is not None]
    rows = dedupe_rows(rows)
    if args.max_train_pairs > 0:
        rows = rows[: args.max_train_pairs]
    if not rows:
        raise RuntimeError("No trainable rows found.")

    if bool(args.val_jsonl) != bool(args.test_jsonl):
        raise ValueError("Use --val-jsonl and --test-jsonl together")
    if args.val_jsonl and args.test_jsonl:
        train_items = rows
        val_items = dedupe_rows(
            [row for source in load_jsonl(args.val_jsonl) if (row := normalize_row(source)) is not None]
        )
        test_items = dedupe_rows(
            [row for source in load_jsonl(args.test_jsonl) if (row := normalize_row(source)) is not None]
        )
    elif args.split == "entity_document":
        train_items, val_items, test_items = component_split(rows, args.val_ratio, args.test_ratio, args.seed)
    elif args.split == "entity":
        train_items, val_items, test_items = grouped_split(rows, args.val_ratio, args.test_ratio, args.seed)
    else:
        train_items, val_items, test_items = split_random(rows, args.val_ratio, args.test_ratio, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_splits:
        save_split(args.output_dir / "train_split.jsonl", train_items)
        save_split(args.output_dir / "val_split.jsonl", val_items)
        save_split(args.output_dir / "test_split.jsonl", test_items)

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

    train_ds = PairDataset(train_items, tokenizer, args.max_length, args.input_mode)
    val_ds = PairDataset(val_items, tokenizer, args.max_length, args.input_mode)
    test_ds = PairDataset(test_items, tokenizer, args.max_length, args.input_mode)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate(
            tokenizer,
            b,
            args.pad_to_multiple_of,
            args.max_length if args.pad_to_max_length else None,
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate(
            tokenizer,
            b,
            args.pad_to_multiple_of,
            args.max_length if args.pad_to_max_length else None,
        ),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate(
            tokenizer,
            b,
            args.pad_to_multiple_of,
            args.max_length if args.pad_to_max_length else None,
        ),
    )

    ranking_items: list[dict[str, Any]] = []
    ranking_loader = None
    if args.ranking_jsonl and args.ranking_weight > 0:
        for source in load_jsonl(args.ranking_jsonl):
            positive = normalize_row(source.get("positive", {}))
            negative = normalize_row(source.get("negative", {}))
            if positive is None or negative is None or positive["label"] != 1 or negative["label"] != 0:
                continue
            ranking_items.append({"positive": positive, "negative": negative})
        if not ranking_items:
            raise RuntimeError("--ranking-jsonl did not contain valid positive-vs-negative pairs")
        ranking_ds = RankingDataset(ranking_items, tokenizer, args.max_length, args.input_mode)
        ranking_loader = DataLoader(
            ranking_ds,
            batch_size=args.ranking_batch_size or args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_ranking(
                tokenizer,
                b,
                args.pad_to_multiple_of,
                args.max_length if args.pad_to_max_length else None,
            ),
        )

    device = pick_device(args.device)
    model.to(device)
    loss_weights = class_weights(train_items, device) if args.weighted_loss else None
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.precision == "fp16")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    update_steps_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
    total_steps = max(1, update_steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )

    metadata = {
        "base_model": args.base_model,
        "train_jsonl": str(args.train_jsonl),
        "input_mode": args.input_mode,
        "max_length": args.max_length,
        "split": args.split,
        "fixed_validation": str(args.val_jsonl) if args.val_jsonl else None,
        "fixed_test": str(args.test_jsonl) if args.test_jsonl else None,
        "precision": args.precision,
        "weighted_loss": args.weighted_loss,
        "gradient_checkpointing": args.gradient_checkpointing,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "pad_to_max_length": args.pad_to_max_length,
        "loss_weights": loss_weights.detach().cpu().tolist() if loss_weights is not None else None,
        "ranking_jsonl": str(args.ranking_jsonl) if args.ranking_jsonl else None,
        "ranking_weight": args.ranking_weight,
        "ranking_pairs": len(ranking_items),
        "data": [
            summarize_rows("train", train_items),
            summarize_rows("val", val_items),
            summarize_rows("test", test_items),
        ],
    }
    write_json(args.output_dir / "training_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)

    metrics_path = args.output_dir / "metrics.jsonl"
    best_macro_f1 = -1.0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        running_cls = 0.0
        running_rank = 0.0
        ranking_iterator = iter(ranking_loader) if ranking_loader is not None else None
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch.pop("meta_idx")
            batch = {k: v.to(device) for k, v in batch.items()}
            with amp_context(device, args.precision):
                out = model(**{k: v for k, v in batch.items() if k != "labels"})
                cls_loss = F.cross_entropy(out.logits, batch["labels"], weight=loss_weights)
                rank_loss = torch.zeros((), device=device)
                if ranking_iterator is not None:
                    try:
                        ranking_batch = next(ranking_iterator)
                    except StopIteration:
                        ranking_iterator = iter(ranking_loader)
                        ranking_batch = next(ranking_iterator)
                    positive = {k: v.to(device) for k, v in ranking_batch["positive"].items()}
                    negative = {k: v.to(device) for k, v in ranking_batch["negative"].items()}
                    positive_logits = model(**positive).logits[:, 1]
                    negative_logits = model(**negative).logits[:, 1]
                    rank_loss = F.softplus(negative_logits - positive_logits).mean()
                loss = cls_loss + args.ranking_weight * rank_loss
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum
            running_cls += cls_loss.item()
            running_rank += rank_loss.item()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.eval_steps and global_step % args.eval_steps == 0:
                    result = evaluate(model, val_loader, val_items, device, loss_weights)
                    record = {"phase": "val", "epoch": epoch + 1, "step": global_step, **result}
                    with metrics_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print("eval", json.dumps(record, ensure_ascii=False, indent=2), flush=True)
            progress.set_postfix(
                loss=f"{running / max(step, 1):.4f}",
                cls=f"{running_cls / max(step, 1):.4f}",
                rank=f"{running_rank / max(step, 1):.4f}",
            )

        val_result = evaluate(model, val_loader, val_items, device, loss_weights)
        record = {"phase": "val", "epoch": epoch + 1, "step": global_step, **val_result}
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("eval", json.dumps(record, ensure_ascii=False, indent=2), flush=True)

        if val_result["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_result["macro_f1"])
            best_dir = args.output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            write_json(best_dir / "best_metrics.json", record)
            print(f"saved best checkpoint {best_dir}", flush=True)

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    test_result = evaluate(model, test_loader, test_items, device, loss_weights) if test_items else {}
    write_json(args.output_dir / "test_metrics.json", test_result)
    print("test", json.dumps(test_result, ensure_ascii=False, indent=2), flush=True)
    print(f"saved final checkpoint {final_dir}", flush=True)


if __name__ == "__main__":
    main()
