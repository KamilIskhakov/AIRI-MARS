#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CORPUS_DIR="${CORPUS_DIR:-$ROOT/data/semantic_corpus_v1_20260819}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000}"
DIRECTIONAL_JSONL="${DIRECTIONAL_JSONL:-$ROOT/data/semantic_quality_rejudge_20260819/directional_disagreements_v2.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/multitask_smoke_$(date +%Y%m%d_%H%M%S)}"
FIXTURE_DIR="$OUTPUT_DIR/fixture"
MODEL_DIR="$OUTPUT_DIR/model"

for required in \
  "$CORPUS_DIR/train.jsonl" \
  "$CORPUS_DIR/val.jsonl" \
  "$CORPUS_DIR/test.jsonl" \
  "$CORPUS_DIR/ranking_train.jsonl" \
  "$DIRECTIONAL_JSONL" \
  "$BASE_MODEL"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

"$PYTHON_BIN" - <<PY
import torch
from transformers import AutoModel, AutoTokenizer

model_path = r"$BASE_MODEL"
AutoTokenizer.from_pretrained(model_path, local_files_only=True)
AutoModel.from_pretrained(model_path, local_files_only=True)
requested = "${DEVICE:-auto}"
if requested.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit("DEVICE=cuda requested, but CUDA is unavailable")
print(f"preflight: torch={torch.__version__} cuda={torch.cuda.is_available()} model=ok tokenizer=ok")
PY

"$PYTHON_BIN" mvp/build_multitask_smoke_fixture.py \
  --corpus-dir "$CORPUS_DIR" \
  --directional-jsonl "$DIRECTIONAL_JSONL" \
  --output-dir "$FIXTURE_DIR" \
  --rows-per-split 8 \
  --ranking-pairs 1

"$PYTHON_BIN" -u mvp/train_multitask_cross_encoder.py \
  --train "$FIXTURE_DIR/train.jsonl" \
  --val "$FIXTURE_DIR/val.jsonl" \
  --test "$FIXTURE_DIR/test.jsonl" \
  --ranking-jsonl "$FIXTURE_DIR/ranking_train.jsonl" \
  --directional-jsonl "$DIRECTIONAL_JSONL" \
  --base-model "$BASE_MODEL" \
  --output-dir "$MODEL_DIR" \
  --input-mode marked_pair \
  --epochs 1 \
  --batch-size 8 \
  --ranking-batch-size 1 \
  --grad-accum 1 \
  --max-length "${MAX_LENGTH:-128}" \
  --precision "${PRECISION:-fp32}" \
  --device "${DEVICE:-auto}" \
  --num-workers 0 \
  --nli-weight 0.3 \
  --relation-weight 0.15 \
  --consistency-weight 0.05 \
  --ranking-weight 0.25 \
  --weak-weight 0.4

for required in \
  "$MODEL_DIR/best/multitask_heads.pt" \
  "$MODEL_DIR/best/multitask_config.json" \
  "$MODEL_DIR/metrics.jsonl" \
  "$MODEL_DIR/test_metrics.json"; do
  [[ -s "$required" ]] || { echo "Smoke output missing or empty: $required" >&2; exit 3; }
done

echo "SMOKE TEST PASSED: exactly one train batch completed"
echo "Artifacts: $OUTPUT_DIR"
