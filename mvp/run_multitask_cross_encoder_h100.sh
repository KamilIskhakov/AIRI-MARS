#!/usr/bin/env bash
set -euo pipefail

# Fully local H100 training for preservation + bidirectional NLI + ranking.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CORPUS_DIR="${CORPUS_DIR:-$ROOT/data/semantic_corpus_v1_20260819}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000}"
RANKING_JSONL="${RANKING_JSONL:-$CORPUS_DIR/ranking_train.jsonl}"
DIRECTIONAL_JSONL="${DIRECTIONAL_JSONL:-$ROOT/data/semantic_quality_rejudge_20260819/directional_disagreements_v2.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/models/semantic_multitask_$(date +%Y%m%d_%H%M%S)}"

for required in "$CORPUS_DIR/train.jsonl" "$CORPUS_DIR/val.jsonl" "$CORPUS_DIR/test.jsonl" "$BASE_MODEL"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

optional_args=()
[[ -s "$RANKING_JSONL" ]] && optional_args+=(--ranking-jsonl "$RANKING_JSONL")
[[ -s "$DIRECTIONAL_JSONL" ]] && optional_args+=(--directional-jsonl "$DIRECTIONAL_JSONL")
[[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]] && optional_args+=(--gradient-checkpointing)
[[ "${MAX_TRAIN_PAIRS:-0}" != "0" ]] && optional_args+=(--max-train-pairs "$MAX_TRAIN_PAIRS")

"$PYTHON_BIN" -u mvp/train_multitask_cross_encoder.py \
  --train "$CORPUS_DIR/train.jsonl" \
  --val "$CORPUS_DIR/val.jsonl" \
  --test "$CORPUS_DIR/test.jsonl" \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --input-mode "${INPUT_MODE:-marked_pair}" \
  --epochs "${EPOCHS:-3}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --ranking-batch-size "${RANKING_BATCH_SIZE:-8}" \
  --grad-accum "${GRAD_ACCUM:-1}" \
  --lr "${LR:-2e-5}" \
  --max-length "${MAX_LENGTH:-512}" \
  --precision "${PRECISION:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --nli-weight "${NLI_WEIGHT:-0.3}" \
  --relation-weight "${RELATION_WEIGHT:-0.15}" \
  --consistency-weight "${CONSISTENCY_WEIGHT:-0.05}" \
  --ranking-weight "${RANKING_WEIGHT:-0.25}" \
  --weak-weight "${WEAK_WEIGHT:-0.4}" \
  "${optional_args[@]}"

echo "Multi-head H100 run complete: $OUTPUT_DIR"
