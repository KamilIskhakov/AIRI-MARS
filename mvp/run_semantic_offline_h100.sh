#!/usr/bin/env bash
set -euo pipefail

# Fully offline data preparation and optional H100 training.
# No Mistral, OpenRouter, HTTP client, or API key is read here.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DB="${DB:-$ROOT/data/entity_inventory.sqlite}"
MODEL="${MODEL:-$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000}"
RUN_ROOT="${RUN_ROOT:-$ROOT/data/semantic_pipeline_offline_h100_$(date +%Y%m%d_%H%M%S)}"
PROPER_TRAIN="${PROPER_TRAIN:-$ROOT/data/proper_name_agent_annotation_full_20260710_mpe2/train_all_reliable.jsonl}"
COMMON_EXISTING="${COMMON_EXISTING:-$ROOT/data/common_noun_embedding_final_20260714/train_audited.jsonl}"
COMMON_TAGS="${COMMON_TAGS:-$ROOT/data/common_noun_retyping_20260714/tags_mistral_medium.jsonl}"
INCLUDE_COMMON_WEAK="${INCLUDE_COMMON_WEAK:-1}"
RUN_TRAINING="${RUN_TRAINING:-0}"
TRAIN_MODE="${TRAIN_MODE:-multitask}"

TAG_FILES=(
  "${TAG_FILE_00:-$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk00.cerebras.jsonl}"
  "${TAG_FILE_01:-$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk01.mistral.jsonl}"
  "${TAG_FILE_02:-$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk02.groq_gptoss.jsonl}"
  "${TAG_FILE_03:-$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk03.openrouter.jsonl}"
)

for required in "$DB" "$MODEL" "$PROPER_TRAIN" "$COMMON_EXISTING" "$COMMON_TAGS" "${TAG_FILES[@]}"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

echo "[0/9] Check offline runtime resources"
MODEL_PATH="$MODEL" DEVICE_CHECK="$DEVICE" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import numpy
import torch
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer

model = Path(os.environ["MODEL_PATH"])
if not model.is_dir():
    raise SystemExit(f"Local model directory is missing: {model}")
AutoTokenizer.from_pretrained(model, local_files_only=True)
if not wn.synsets("entity", pos=wn.NOUN):
    raise SystemExit("NLTK WordNet data is missing; copy it to the H100 before running offline")
device = os.environ["DEVICE_CHECK"]
if device.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
print(f"offline resources: model=ok wordnet=ok numpy=ok torch={torch.__version__} device={device}")
PY

mkdir -p "$RUN_ROOT" "$RUN_ROOT/audit" "$RUN_ROOT/common" "$RUN_ROOT/numeric"
printf 'offline=true\nremote_judgment_used=false\nmodel=%s\ndevice=%s\ntrain_mode=%s\n' \
  "$MODEL" "$DEVICE" "$TRAIN_MODE" > "$RUN_ROOT/run_config.txt"

echo "[1/9] Audit existing local sources"
"$PYTHON_BIN" -u mvp/audit_semantic_dataset.py \
  --input "proper=$PROPER_TRAIN" \
  --input "common_existing=$COMMON_EXISTING" \
  --out "$RUN_ROOT/audit/input_sources.json" \
  --sample-out "$RUN_ROOT/audit/input_sources.samples.jsonl"

echo "[2/9] Generate common candidates locally"
OUT_ROOT="$RUN_ROOT/common" DB="$DB" MODEL="$MODEL" PYTHON_BIN="$PYTHON_BIN" \
  MAX_WORDNET_ITEMS="${MAX_WORDNET_ITEMS:-20000}" \
  bash mvp/run_common_weak_generation.sh

echo "[3/9] Generate numeric candidates from the short local context window"
"$PYTHON_BIN" -u mvp/generate_numeric_context_pairs.py \
  --tag-files "${TAG_FILES[@]}" --db "$DB" \
  --out "$RUN_ROOT/numeric/pairs.jsonl" \
  --summary-out "$RUN_ROOT/numeric/generation_summary.json" \
  --mentions-per-entity "${NUMERIC_MENTIONS:-1}"

echo "[4/9] Score contextual fit with the local MLM"
for branch in common numeric; do
  input_path="$RUN_ROOT/$branch/train_weak.jsonl"
  [[ "$branch" == "numeric" ]] && input_path="$RUN_ROOT/numeric/pairs.jsonl"
  "$PYTHON_BIN" -u mvp/score_pairs_mlm_fit.py \
    --input "$input_path" \
    --out "$RUN_ROOT/$branch/pairs_mlm_scored.jsonl" \
    --summary-out "$RUN_ROOT/$branch/mlm_fit_summary.json" \
    --model "$MODEL" --batch-size "${MLM_BATCH_SIZE:-32}" --device "$DEVICE"
done

echo "[5/9] Materialize local weak/rule labels"
label_inputs=(--input "numeric=$RUN_ROOT/numeric/pairs.jsonl")
if [[ "$INCLUDE_COMMON_WEAK" == "1" ]]; then
  label_inputs+=(--input "common_weak=$RUN_ROOT/common/train_weak.jsonl")
fi
"$PYTHON_BIN" mvp/materialize_offline_labels.py \
  "${label_inputs[@]}" \
  --out "$RUN_ROOT/offline_train_candidates.jsonl" \
  --summary-out "$RUN_ROOT/offline_train_candidates.summary.json"

echo "[6/9] Audit candidate pool"
"$PYTHON_BIN" -u mvp/audit_semantic_dataset.py \
  --input "offline_candidates=$RUN_ROOT/offline_train_candidates.jsonl" \
  --out "$RUN_ROOT/audit/offline_candidates.json" \
  --sample-out "$RUN_ROOT/audit/offline_candidates.samples.jsonl" \
  --require-both-labels-by coarse_group --min-group-rows 5 --fail-on-critical

echo "[7/9] Build entity/document-disjoint offline corpus"
"$PYTHON_BIN" mvp/build_semantic_corpus.py \
  --input "proper=$PROPER_TRAIN" \
  --input "common_existing=$COMMON_EXISTING" \
  --input "offline_candidates=$RUN_ROOT/offline_train_candidates.jsonl" \
  --output-dir "$RUN_ROOT/corpus" --seed "${SPLIT_SEED:-41}"

"$PYTHON_BIN" -u mvp/audit_semantic_dataset.py \
  --input "corpus=$RUN_ROOT/corpus/all.jsonl" \
  --out "$RUN_ROOT/audit/final_corpus.json" \
  --sample-out "$RUN_ROOT/audit/final_corpus.samples.jsonl" \
  --require-both-labels-by coarse_group --min-group-rows 20 --fail-on-critical

echo "[8/9] Build leakage-safe ranking pairs from train only"
"$PYTHON_BIN" mvp/prepare_ranking_pairs.py \
  --input "$RUN_ROOT/corpus/train.jsonl" \
  --out "$RUN_ROOT/corpus/ranking_train.jsonl" \
  --summary-out "$RUN_ROOT/corpus/ranking_summary.json" \
  --max-negatives-per-positive "${MAX_NEGATIVES_PER_POSITIVE:-4}" \
  --seed "${SPLIT_SEED:-41}"

if [[ "$RUN_TRAINING" == "1" ]]; then
  echo "[9/9] Train the $TRAIN_MODE cross-encoder on H100"
  if [[ "$TRAIN_MODE" == "multitask" ]]; then
    CORPUS_DIR="$RUN_ROOT/corpus" BASE_MODEL="$MODEL" \
    RANKING_JSONL="$RUN_ROOT/corpus/ranking_train.jsonl" \
    OUTPUT_DIR="$RUN_ROOT/training/modernbert_offline_multitask" \
    DEVICE="$DEVICE" PYTHON_BIN="$PYTHON_BIN" \
    INPUT_MODE="${INPUT_MODE:-marked_pair}" EPOCHS="${EPOCHS:-3}" \
    BATCH_SIZE="${BATCH_SIZE:-32}" GRAD_ACCUM="${GRAD_ACCUM:-1}" \
    LR="${LR:-2e-5}" MAX_LENGTH="${MAX_LENGTH:-512}" \
    PRECISION="${PRECISION:-bf16}" NUM_WORKERS="${NUM_WORKERS:-8}" \
    WEAK_WEIGHT="${WEAK_WEIGHT:-0.4}" \
    bash mvp/run_multitask_cross_encoder_h100.sh
  elif [[ "$TRAIN_MODE" == "binary" ]]; then
    TRAIN_JSONL="$RUN_ROOT/corpus/train.jsonl" \
    VAL_JSONL="$RUN_ROOT/corpus/val.jsonl" \
    TEST_JSONL="$RUN_ROOT/corpus/test.jsonl" \
    BASE_MODEL="$MODEL" DEVICE="$DEVICE" PYTHON_BIN="$PYTHON_BIN" \
    RUN_ROOT="$RUN_ROOT/training" RUN_NAME="modernbert_offline_binary" \
    INPUT_MODE="${INPUT_MODE:-marked_pair}" EPOCHS="${EPOCHS:-3}" \
    BATCH_SIZE="${BATCH_SIZE:-32}" GRAD_ACCUM="${GRAD_ACCUM:-1}" \
    LR="${LR:-2e-5}" MAX_LENGTH="${MAX_LENGTH:-512}" \
    PRECISION="${PRECISION:-bf16}" NUM_WORKERS="${NUM_WORKERS:-8}" \
    WEIGHTED_LOSS="${WEIGHTED_LOSS:-1}" GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}" \
    bash mvp/run_cross_encoder_4090.sh
  else
    echo "Unknown TRAIN_MODE=$TRAIN_MODE; expected multitask or binary" >&2
    exit 2
  fi
else
  echo "[9/9] Training not started. Set RUN_TRAINING=1 for the H100 run."
fi

echo "Offline pipeline complete: $RUN_ROOT"
