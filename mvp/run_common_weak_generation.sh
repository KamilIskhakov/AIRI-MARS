#!/usr/bin/env bash
set -euo pipefail

# Generate common-entity candidates locally. Mistral is not called here.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DB="${DB:-$ROOT/data/entity_inventory.sqlite}"
MODEL="${MODEL:-$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000}"
OUT_ROOT="${OUT_ROOT:-$ROOT/data/common_weak_encoder_$(date +%Y%m%d_%H%M%S)}"
COMMON_TAGS="${COMMON_TAGS:-$ROOT/data/common_noun_retyping_20260714/tags_mistral_medium.jsonl}"

mkdir -p "$OUT_ROOT"
for required in "$DB" "$MODEL" "$COMMON_TAGS"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

echo "[1/3] Retrieve from existing common inventory"
"$PYTHON_BIN" -u mvp/generate_common_embedding_pairs.py \
  --source inventory --db "$DB" --model "$MODEL" \
  --out "$OUT_ROOT/inventory_pairs.jsonl" \
  --neighbors-out "$OUT_ROOT/inventory_neighbors.jsonl" \
  --summary-out "$OUT_ROOT/inventory_summary.json" \
  --very-high-min "${VERY_HIGH_MIN:-0.97}" \
  --hard-min "${HARD_MIN:-0.90}" --hard-max "${HARD_MAX:-0.97}" \
  --neighbors "${NEIGHBORS:-80}" --pairs-per-bucket "${PAIRS_PER_BUCKET:-8}" \
  --min-mentions "${MIN_MENTIONS:-2}" --require-wordnet-noun \
  --batch-size "${EMBED_BATCH_SIZE:-256}"

echo "[2/4] Expand with WordNet nouns using the same local encoder"
"$PYTHON_BIN" -u mvp/generate_common_wordnet_embedding_pairs.py \
  --tag-files "$COMMON_TAGS" --db "$DB" --model "$MODEL" \
  --out "$OUT_ROOT/wordnet_pairs.jsonl" \
  --neighbors-out "$OUT_ROOT/wordnet_neighbors.jsonl" \
  --summary-out "$OUT_ROOT/wordnet_summary.json" \
  --very-high-min "${VERY_HIGH_MIN:-0.97}" \
  --hard-min "${HARD_MIN:-0.90}" --hard-max "${HARD_MAX:-0.97}" \
  --neighbors "${WORDNET_NEIGHBORS:-80}" \
  --pairs-per-bucket "${WORDNET_PAIRS_PER_BUCKET:-8}" \
  --max-wordnet-items "${MAX_WORDNET_ITEMS:-20000}" \
  --batch-size "${EMBED_BATCH_SIZE:-256}"

echo "[3/4] Add direct WordNet synonym positives without an LLM"
"$PYTHON_BIN" -u mvp/generate_common_wordnet_synonym_pairs.py \
  --source inventory --db "$DB" \
  --out "$OUT_ROOT/wordnet_synonym_pairs.jsonl" \
  --summary-out "$OUT_ROOT/wordnet_synonym_summary.json" \
  --synonyms-per-source "${SYNONYMS_PER_SOURCE:-3}" \
  --mentions-per-source "${SYNONYM_MENTIONS:-1}" \
  --min-mentions "${MIN_MENTIONS:-2}"

echo "[4/4] Assign conservative weak labels without an LLM"
"$PYTHON_BIN" mvp/label_common_embedding_pairs.py \
  --input "$OUT_ROOT/inventory_pairs.jsonl" "$OUT_ROOT/wordnet_pairs.jsonl" "$OUT_ROOT/wordnet_synonym_pairs.jsonl" \
  --out "$OUT_ROOT/train_weak.jsonl" \
  --summary-out "$OUT_ROOT/weak_summary.json" \
  --positive-min "${POSITIVE_MIN:-0.90}" \
  --hard-min "${HARD_MIN:-0.90}" --hard-max "${HARD_MAX:-0.97}" \
  ${INCLUDE_MEDIUM:+--include-medium}

echo "Weak common generation complete: $OUT_ROOT"
