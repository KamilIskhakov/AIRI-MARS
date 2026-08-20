#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DB="${DB:-$ROOT/data/entity_inventory.sqlite}"
MODEL="${MODEL:-$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000}"
OUT_ROOT="${OUT_ROOT:-$ROOT/data/nonproper_annotation_$(date +%Y%m%d)}"
COMMON_TAGS="${COMMON_TAGS:-$ROOT/data/common_noun_retyping_20260714/tags_mistral_medium.jsonl}"
MISTRAL_MODEL="${MISTRAL_STRONG_MODEL:-mistral-medium-latest}"
SECOND_JUDGE_PROVIDER="${SECOND_JUDGE_PROVIDER:-groq}"
SECOND_JUDGE_MODEL="${SECOND_JUDGE_MODEL:-}"
SECOND_JUDGE_BATCH_SIZE="${SECOND_JUDGE_BATCH_SIZE:-2}"

TAG_FILES=(
  "$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk00.cerebras.jsonl"
  "$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk01.mistral.jsonl"
  "$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk02.groq_gptoss.jsonl"
  "$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk03.openrouter.jsonl"
)

mkdir -p "$OUT_ROOT/common" "$OUT_ROOT/numeric" "$OUT_ROOT/corpus"
for required in "$DB" "$MODEL" "$COMMON_TAGS" "${TAG_FILES[@]}"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required artifact: $required" >&2
    exit 2
  fi
done

echo "[1/6] Generate common MLM candidates"
"$PYTHON_BIN" -u "$ROOT/mvp/generate_common_mlm_candidates.py" \
  --tag-files "$COMMON_TAGS" \
  --db "$DB" \
  --model "$MODEL" \
  --out "$OUT_ROOT/common/pairs_mlm_candidates.jsonl" \
  --summary-out "$OUT_ROOT/common/generation_summary.json" \
  --top-k "${COMMON_TOP_K:-200}" \
  --candidates-per-mention "${COMMON_CANDIDATES:-5}" \
  --mentions-per-entity "${COMMON_MENTIONS:-2}" \
  --device "${DEVICE:-auto}"

echo "[1b/6] Generate explicit common synonyms and hard negatives with Mistral"
"$PYTHON_BIN" -u "$ROOT/mvp/generate_common_candidates_mistral.py" \
  --input "$OUT_ROOT/common/pairs_mlm_candidates.jsonl" \
  --out "$OUT_ROOT/common/pairs_agent_candidates.jsonl" \
  --raw-out "$OUT_ROOT/common/pairs_agent_candidates.raw.jsonl" \
  --model "$MISTRAL_MODEL" \
  --batch-size "${COMMON_AGENT_BATCH_SIZE:-5}"

"$PYTHON_BIN" "$ROOT/mvp/merge_pair_files.py" \
  --input "$OUT_ROOT/common/pairs_mlm_candidates.jsonl" "$OUT_ROOT/common/pairs_agent_candidates.jsonl" \
  --out "$OUT_ROOT/common/pairs.jsonl"

echo "[2/6] Generate numeric candidates"
"$PYTHON_BIN" -u "$ROOT/mvp/generate_numeric_context_pairs.py" \
  --tag-files "${TAG_FILES[@]}" \
  --db "$DB" \
  --out "$OUT_ROOT/numeric/pairs.jsonl" \
  --summary-out "$OUT_ROOT/numeric/generation_summary.json" \
  --mentions-per-entity "${NUMERIC_MENTIONS:-1}"

echo "[3/6] Score contextual fit with ModernBERT MLM"
for branch in common numeric; do
  "$PYTHON_BIN" -u "$ROOT/mvp/score_pairs_mlm_fit.py" \
    --input "$OUT_ROOT/$branch/pairs.jsonl" \
    --out "$OUT_ROOT/$branch/pairs_mlm_scored.jsonl" \
    --summary-out "$OUT_ROOT/$branch/mlm_fit_summary.json" \
    --model "$MODEL" \
    --batch-size "${MLM_BATCH_SIZE:-8}" \
    --device "${DEVICE:-auto}"
done

set -a
source "$ROOT/mvp/.env"
set +a
if [[ -n "${MISTRAL_API_KEY_2:-}" ]]; then
  export MISTRAL_API_KEY="$MISTRAL_API_KEY_2"
fi

echo "[4/6] Independent Mistral and ${SECOND_JUDGE_PROVIDER} judgments"
for branch in common numeric; do
  "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_mistral.py" \
    --input-jsonl "$OUT_ROOT/$branch/pairs_mlm_scored.jsonl" \
    --out "$OUT_ROOT/$branch/judge_mistral.jsonl" \
    --summary-out "$OUT_ROOT/$branch/judge_mistral_summary.json" \
    --warnings-out "$OUT_ROOT/$branch/judge_mistral_warnings.jsonl" \
    --model "$MISTRAL_MODEL" \
    --batch-size "${JUDGE_BATCH_SIZE:-8}" \
    --resume &
  mistral_pid=$!

  second_model_args=()
  if [[ -n "$SECOND_JUDGE_MODEL" ]]; then
    second_model_args=(--model "$SECOND_JUDGE_MODEL")
  fi
  "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_provider.py" \
    --provider "$SECOND_JUDGE_PROVIDER" \
    --input-jsonl "$OUT_ROOT/$branch/pairs_mlm_scored.jsonl" \
    --out "$OUT_ROOT/$branch/judge_second.jsonl" \
    --summary-out "$OUT_ROOT/$branch/judge_second_summary.json" \
    --warnings-out "$OUT_ROOT/$branch/judge_second_warnings.jsonl" \
    --batch-size "$SECOND_JUDGE_BATCH_SIZE" \
    --max-tokens "${SECOND_JUDGE_MAX_TOKENS:-1800}" \
    --schema-mode json_schema \
    "${second_model_args[@]}" \
    --resume &
  second_pid=$!

  wait "$mistral_pid"
  wait "$second_pid"
done

echo "[5/6] Conservative two-judge consensus"
for branch in common numeric; do
  "$PYTHON_BIN" "$ROOT/mvp/merge_judges_consensus.py" \
    --judged "$OUT_ROOT/$branch/judge_mistral.jsonl" "$OUT_ROOT/$branch/judge_second.jsonl" \
    --out-final "$OUT_ROOT/$branch/consensus.jsonl" \
    --out-train "$OUT_ROOT/$branch/train_consensus.jsonl" \
    --summary-out "$OUT_ROOT/$branch/consensus_summary.json" \
    --positive-votes 2 \
    --negative-votes 2
done

echo "[6/6] Build fixed entity/document-disjoint corpus"
"$PYTHON_BIN" "$ROOT/mvp/build_semantic_corpus.py" \
  --input "proper=$ROOT/data/proper_name_agent_annotation_full_20260710_mpe2/train_all_reliable.jsonl" \
  --input "common_existing=$ROOT/data/common_noun_embedding_final_20260714/train_audited.jsonl" \
  --input "common_mlm=$OUT_ROOT/common/train_consensus.jsonl" \
  --input "numeric=$OUT_ROOT/numeric/train_consensus.jsonl" \
  --output-dir "$OUT_ROOT/corpus" \
  --seed "${SPLIT_SEED:-41}"

echo "Pipeline complete: $OUT_ROOT"
