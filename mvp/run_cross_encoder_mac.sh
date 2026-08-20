#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PY="/Users/kamiliskhakov/VS Code Project/SKOLTECH sum 2026/smoVLA drone/.conda-ml/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PY}"

TRAIN_JSONL="${TRAIN_JSONL:-$ROOT/data/proper_name_agent_annotation_strict_12k/train_consensus.jsonl}"
LOCAL_BASE="$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000"
BASE_MODEL="${BASE_MODEL:-$LOCAL_BASE}"

RUN_ROOT="${RUN_ROOT:-$ROOT/runs/cross_encoder}"
RUN_NAME="${RUN_NAME:-local_mps_marked_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/$RUN_NAME}"

EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DEVICE="${DEVICE:-mps}"
PRECISION="${PRECISION:-fp32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -s "$TRAIN_JSONL" ]]; then
  echo "Train file is missing or empty: $TRAIN_JSONL" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
echo "python=$PYTHON_BIN"
echo "train_jsonl=$TRAIN_JSONL"
echo "base_model=$BASE_MODEL"
echo "out_dir=$OUT_DIR"
echo "device=$DEVICE batch_size=$BATCH_SIZE max_length=$MAX_LENGTH epochs=$EPOCHS"

EXTRA_ARGS=()
if [[ "$MAX_TRAIN_PAIRS" != "0" ]]; then
  EXTRA_ARGS+=(--max-train-pairs "$MAX_TRAIN_PAIRS")
fi

env PYTHONPATH="$ROOT/mvp:${PYTHONPATH:-}" "$PYTHON_BIN" -u "$ROOT/mvp/train_cross_encoder.py" \
  --train-jsonl "$TRAIN_JSONL" \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUT_DIR" \
  --input-mode marked_pair \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --max-length "$MAX_LENGTH" \
  --device "$DEVICE" \
  --precision "$PRECISION" \
  --num-workers "$NUM_WORKERS" \
  --pad-to-max-length \
  --save-splits \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$OUT_DIR/train.log"
