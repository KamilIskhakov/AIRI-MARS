#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_DIR="${CORPUS_DIR:-$ROOT/data/semantic_corpus_v1_20260819}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/semantic_h100}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODES="${MODES:-pair marked_pair masked_query}"

for required in train.jsonl val.jsonl test.jsonl manifest.json; do
  if [[ ! -s "$CORPUS_DIR/$required" ]]; then
    echo "Missing corpus artifact: $CORPUS_DIR/$required" >&2
    exit 2
  fi
done
if [[ ! -d "$BASE_MODEL" ]]; then
  echo "Missing base model: $BASE_MODEL" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
cp "$CORPUS_DIR/manifest.json" "$RUN_ROOT/corpus_manifest.json"

for mode in $MODES; do
  run_name="modernbert_${mode}_$(date +%Y%m%d_%H%M%S)"
  echo "Starting H100 ablation: $mode -> $run_name"
  env \
    PYTHON_BIN="$PYTHON_BIN" \
    TRAIN_JSONL="$CORPUS_DIR/train.jsonl" \
    VAL_JSONL="$CORPUS_DIR/val.jsonl" \
    TEST_JSONL="$CORPUS_DIR/test.jsonl" \
    BASE_MODEL="$BASE_MODEL" \
    RUN_ROOT="$RUN_ROOT" \
    RUN_NAME="$run_name" \
    INPUT_MODE="$mode" \
    EPOCHS="${EPOCHS:-4}" \
    BATCH_SIZE="${BATCH_SIZE:-32}" \
    GRAD_ACCUM="${GRAD_ACCUM:-1}" \
    LR="${LR:-2e-5}" \
    MAX_LENGTH="${MAX_LENGTH:-512}" \
    PRECISION="${PRECISION:-bf16}" \
    DEVICE="${DEVICE:-cuda}" \
    NUM_WORKERS="${NUM_WORKERS:-8}" \
    WEIGHTED_LOSS="${WEIGHTED_LOSS:-0}" \
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}" \
    bash "$ROOT/mvp/run_cross_encoder_4090.sh"
done

echo "All ablations finished under $RUN_ROOT"
