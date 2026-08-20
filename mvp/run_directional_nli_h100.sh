#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NLI_DIR="${NLI_DIR:-$ROOT/data/directional_relations}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/models/directional_nli_$(date +%Y%m%d_%H%M%S)}"

for required in "$NLI_DIR/train_nli.jsonl" "$NLI_DIR/val_nli.jsonl" "$NLI_DIR/test_nli.jsonl" "$BASE_MODEL"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required artifact: $required" >&2
    exit 2
  fi
done

"$PYTHON_BIN" -u "$ROOT/mvp/train_directional_nli.py" \
  --train "$NLI_DIR/train_nli.jsonl" \
  --val "$NLI_DIR/val_nli.jsonl" \
  --test "$NLI_DIR/test_nli.jsonl" \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "${EPOCHS:-3}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --grad-accum "${GRAD_ACCUM:-1}" \
  --lr "${LR:-2e-5}" \
  --max-length "${MAX_LENGTH:-512}" \
  --precision "${PRECISION:-bf16}" \
  --num-workers "${NUM_WORKERS:-8}"

echo "Directional NLI run complete: $OUTPUT_DIR"
