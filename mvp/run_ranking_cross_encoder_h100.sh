#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CORPUS_DIR="${CORPUS_DIR:-$ROOT/data/semantic_corpus_v1_20260819}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000}"
INPUT_MODE="${INPUT_MODE:-marked_pair}"
RANKING_WEIGHT="${RANKING_WEIGHT:-0.25}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/models/semantic_ranking_$(date +%Y%m%d_%H%M%S)}"

for required in \
  "$CORPUS_DIR/train.jsonl" \
  "$CORPUS_DIR/val.jsonl" \
  "$CORPUS_DIR/test.jsonl" \
  "$CORPUS_DIR/ranking_train.jsonl" \
  "$BASE_MODEL"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required artifact: $required" >&2
    exit 2
  fi
done

"$PYTHON_BIN" -u "$ROOT/mvp/train_cross_encoder.py" \
  --train-jsonl "$CORPUS_DIR/train.jsonl" \
  --val-jsonl "$CORPUS_DIR/val.jsonl" \
  --test-jsonl "$CORPUS_DIR/test.jsonl" \
  --ranking-jsonl "$CORPUS_DIR/ranking_train.jsonl" \
  --ranking-weight "$RANKING_WEIGHT" \
  --ranking-batch-size "${RANKING_BATCH_SIZE:-8}" \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --input-mode "$INPUT_MODE" \
  --epochs "${EPOCHS:-4}" \
  --batch-size "${BATCH_SIZE:-24}" \
  --grad-accum "${GRAD_ACCUM:-1}" \
  --lr "${LR:-2e-5}" \
  --max-length "${MAX_LENGTH:-512}" \
  --precision "${PRECISION:-bf16}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --no-weighted-loss \
  --save-splits

echo "Ranking run complete: $OUTPUT_DIR"
