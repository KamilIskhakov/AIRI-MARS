#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/tmp/airi_mars_torch_venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/data/proper_name_agent_annotation_longrun}"
DB_PATH="${DB_PATH:-$ROOT/data/entity_inventory.sqlite}"
START_OFFSET="${START_OFFSET:-0}"
WINDOWS="${WINDOWS:-1}"
LIMIT_PER_SHARD="${LIMIT_PER_SHARD:-250}"
MENTIONS_PER_ENTITY="${MENTIONS_PER_ENTITY:-2}"
SEED="${SEED:-101}"
RUN_GROQ="${RUN_GROQ:-0}"
VALIDATE_EXISTING_PAIRS="${VALIDATE_EXISTING_PAIRS:-0}"

mkdir -p "$OUT_ROOT"
echo "[$(date)] start windows out_root=$OUT_ROOT start_offset=$START_OFFSET windows=$WINDOWS limit_per_shard=$LIMIT_PER_SHARD mentions_per_entity=$MENTIONS_PER_ENTITY db_path=$DB_PATH" | tee -a "$OUT_ROOT/windows.log"

for ((window_idx = 0; window_idx < WINDOWS; window_idx++)); do
  base_offset=$((START_OFFSET + window_idx * LIMIT_PER_SHARD * 2))
  window_name="$(printf 'window_%03d_offset_%06d' "$window_idx" "$base_offset")"
  out_dir="$OUT_ROOT/$window_name"
  done_marker="$out_dir/.done"

  if [[ -f "$done_marker" ]]; then
    echo "[$(date)] skip $window_name done_marker_exists=true" | tee -a "$OUT_ROOT/windows.log"
    continue
  fi

  if [[ "$VALIDATE_EXISTING_PAIRS" == "1" ]] && [[ ! -s "$out_dir/proper_agent_pairs.jsonl" ]]; then
    echo "[$(date)] skip $window_name validate_existing_pairs=true pairs_missing=true" | tee -a "$OUT_ROOT/windows.log"
    continue
  fi

  echo "[$(date)] run $window_name" | tee -a "$OUT_ROOT/windows.log"
  env \
    OUT_DIR="$out_dir" \
    DB_PATH="$DB_PATH" \
    LIMIT_PER_SHARD="$LIMIT_PER_SHARD" \
    BASE_OFFSET="$base_offset" \
    MENTIONS_PER_ENTITY="$MENTIONS_PER_ENTITY" \
    SEED="$SEED" \
    RUN_GROQ="$RUN_GROQ" \
    VALIDATE_EXISTING_PAIRS="$VALIDATE_EXISTING_PAIRS" \
    PYTHON="$PYTHON_BIN" \
    bash "$ROOT/mvp/run_proper_agent_annotation_parallel.sh"
  touch "$done_marker"
  echo "[$(date)] done $window_name" | tee -a "$OUT_ROOT/windows.log"
done

echo "[$(date)] all requested windows done" | tee -a "$OUT_ROOT/windows.log"
