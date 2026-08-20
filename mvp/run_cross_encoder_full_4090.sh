#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export TRAIN_JSONL="${TRAIN_JSONL:-$ROOT/data/proper_name_agent_annotation_full_20260710_mpe2/train_all_reliable.jsonl}"
export BASE_MODEL="${BASE_MODEL:-$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000}"
export RUN_NAME="${RUN_NAME:-proper_full_reliable_modernbert_$(date +%Y%m%d_%H%M%S)}"
export EPOCHS="${EPOCHS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export LR="${LR:-2e-5}"
export MAX_LENGTH="${MAX_LENGTH:-512}"
export PRECISION="${PRECISION:-bf16}"
export DEVICE="${DEVICE:-cuda}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export INPUT_MODE="${INPUT_MODE:-marked_pair}"
export WEIGHTED_LOSS="${WEIGHTED_LOSS:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export RUN_ENTITY_QUERY="${RUN_ENTITY_QUERY:-0}"

exec bash "$ROOT/mvp/run_cross_encoder_4090.sh"
