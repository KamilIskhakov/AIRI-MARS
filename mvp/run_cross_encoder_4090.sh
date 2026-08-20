#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DEFAULT_TRAIN="$ROOT/data/proper_name_agent_annotation_full_20260710_mpe2/train_all_reliable.jsonl"
if [[ ! -s "$DEFAULT_TRAIN" ]]; then
  DEFAULT_TRAIN="$ROOT/data/proper_name_agent_annotation_strict_12k/train_consensus_clean.jsonl"
fi
if [[ ! -s "$DEFAULT_TRAIN" ]]; then
  DEFAULT_TRAIN="$ROOT/data/proper_name_agent_annotation_strict_12k/train_consensus.jsonl"
fi
TRAIN_JSONL="${TRAIN_JSONL:-$DEFAULT_TRAIN}"
LOCAL_BASE="$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000"
if [[ -z "${BASE_MODEL:-}" ]]; then
  if [[ -d "$LOCAL_BASE" ]]; then
    BASE_MODEL="$LOCAL_BASE"
  else
    BASE_MODEL="microsoft/deberta-v3-base"
  fi
fi

RUN_ROOT="${RUN_ROOT:-$ROOT/runs/cross_encoder}"
RUN_NAME="${RUN_NAME:-proper_full_reliable_marked_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/$RUN_NAME}"

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-24}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
PRECISION="${PRECISION:-bf16}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_STEPS="${EVAL_STEPS:-0}"
INPUT_MODE="${INPUT_MODE:-marked_pair}"
WEIGHTED_LOSS="${WEIGHTED_LOSS:-1}"
MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
VAL_JSONL="${VAL_JSONL:-}"
TEST_JSONL="${TEST_JSONL:-}"

if [[ ! -s "$TRAIN_JSONL" ]]; then
  echo "Train file is missing or empty: $TRAIN_JSONL" >&2
  echo "Wait until the annotation pipeline writes train_consensus.jsonl." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
echo "train_jsonl=$TRAIN_JSONL"
echo "base_model=$BASE_MODEL"
echo "out_dir=$OUT_DIR"
echo "input_mode=$INPUT_MODE"
echo "weighted_loss=$WEIGHTED_LOSS"
echo "max_train_pairs=$MAX_TRAIN_PAIRS"

LOSS_ARGS=()
if [[ "$WEIGHTED_LOSS" == "0" ]]; then
  LOSS_ARGS+=(--no-weighted-loss)
fi
LIMIT_ARGS=()
if [[ "$MAX_TRAIN_PAIRS" != "0" ]]; then
  LIMIT_ARGS+=(--max-train-pairs "$MAX_TRAIN_PAIRS")
fi
CHECKPOINT_ARGS=()
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  CHECKPOINT_ARGS+=(--gradient-checkpointing)
fi
FIXED_SPLIT_ARGS=()
if [[ -n "$VAL_JSONL" || -n "$TEST_JSONL" ]]; then
  if [[ ! -s "$VAL_JSONL" || ! -s "$TEST_JSONL" ]]; then
    echo "VAL_JSONL and TEST_JSONL must both point to non-empty files" >&2
    exit 2
  fi
  FIXED_SPLIT_ARGS+=(--val-jsonl "$VAL_JSONL" --test-jsonl "$TEST_JSONL")
fi

env PYTHONPATH="$ROOT/mvp:${PYTHONPATH:-}" "$PYTHON_BIN" -u "$ROOT/mvp/train_cross_encoder.py" \
  --train-jsonl "$TRAIN_JSONL" \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUT_DIR" \
  --input-mode "$INPUT_MODE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --max-length "$MAX_LENGTH" \
  --precision "$PRECISION" \
  --device "$DEVICE" \
  --num-workers "$NUM_WORKERS" \
  --eval-steps "$EVAL_STEPS" \
  --save-splits \
  "${FIXED_SPLIT_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "${LOSS_ARGS[@]}" 2>&1 | tee "$OUT_DIR/train.log"

env PYTHONPATH="$ROOT/mvp:${PYTHONPATH:-}" "$PYTHON_BIN" -u "$ROOT/mvp/evaluate_cross_encoder_thresholds.py" \
  --checkpoint "$OUT_DIR/best" \
  --run-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" 2>&1 | tee "$OUT_DIR/threshold_calibration.log"

if [[ "${RUN_ENTITY_QUERY:-0}" == "1" ]]; then
  QUERY_OUT_DIR="${QUERY_OUT_DIR:-${OUT_DIR}_entity_query}"
  mkdir -p "$QUERY_OUT_DIR"
  env PYTHONPATH="$ROOT/mvp:${PYTHONPATH:-}" "$PYTHON_BIN" -u "$ROOT/mvp/train_cross_encoder.py" \
    --train-jsonl "$TRAIN_JSONL" \
    --base-model "$BASE_MODEL" \
    --output-dir "$QUERY_OUT_DIR" \
    --input-mode entity_query \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --max-length "$MAX_LENGTH" \
    --precision "$PRECISION" \
    --device "$DEVICE" \
    --num-workers "$NUM_WORKERS" \
    --eval-steps "$EVAL_STEPS" \
    --save-splits \
    "${FIXED_SPLIT_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${LIMIT_ARGS[@]}" \
    "${LOSS_ARGS[@]}" 2>&1 | tee "$QUERY_OUT_DIR/train.log"

  env PYTHONPATH="$ROOT/mvp:${PYTHONPATH:-}" "$PYTHON_BIN" -u "$ROOT/mvp/evaluate_cross_encoder_thresholds.py" \
    --checkpoint "$QUERY_OUT_DIR/best" \
    --run-dir "$QUERY_OUT_DIR" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" 2>&1 | tee "$QUERY_OUT_DIR/threshold_calibration.log"
fi
