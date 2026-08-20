#!/usr/bin/env bash
set -euo pipefail

# Reproducible candidate generation -> independent judging -> consensus -> corpus.
# Each run gets a fresh directory; existing data is never removed or overwritten.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH="$ROOT/mvp${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

RUN_ROOT="${RUN_ROOT:-$ROOT/data/semantic_pipeline_$(date +%Y%m%d_%H%M%S)}"
DB="${DB:-$ROOT/data/entity_inventory.sqlite}"
MODEL="${MODEL:-$ROOT/models/h1/merged/cnn_dailymail+xsum+multi_news+samsum/merged/checkpoint-40000}"
MISTRAL_MODEL="${MISTRAL_MODEL:-${MISTRAL_STRONG_MODEL:-mistral-medium-latest}}"
SECOND_JUDGE_PROVIDER="${SECOND_JUDGE_PROVIDER:-openrouter}"
SECOND_JUDGE_MODEL="${SECOND_JUDGE_MODEL:-}"
PROPER_TRAIN="${PROPER_TRAIN:-$ROOT/data/proper_name_agent_annotation_full_20260710_mpe2/train_all_reliable.jsonl}"
COMMON_EXISTING="${COMMON_EXISTING:-$ROOT/data/common_noun_embedding_final_20260714/train_audited.jsonl}"
COMMON_TAGS="${COMMON_TAGS:-$ROOT/data/common_noun_retyping_20260714/tags_mistral_medium.jsonl}"

TAG_FILES=(
  "${TAG_FILE_00:-$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk00.cerebras.jsonl}"
  "${TAG_FILE_01:-$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk01.mistral.jsonl}"
  "${TAG_FILE_02:-$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk02.groq_gptoss.jsonl}"
  "${TAG_FILE_03:-$ROOT/data/substitution_annotation_current_all/tag_snapshot/entity_tags_top50k_chunk03.openrouter.jsonl}"
)

for required in "$DB" "$MODEL" "$PROPER_TRAIN" "$COMMON_EXISTING" "$COMMON_TAGS" "${TAG_FILES[@]}"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 2; }
done

mkdir -p "$RUN_ROOT/common" "$RUN_ROOT/numeric" "$RUN_ROOT/audit"

echo "[0/7] Audit existing sources"
"$PYTHON_BIN" -u mvp/audit_semantic_dataset.py \
  --input "proper=$PROPER_TRAIN" \
  --input "common_existing=$COMMON_EXISTING" \
  --out "$RUN_ROOT/audit/input_sources.json" \
  --sample-out "$RUN_ROOT/audit/input_sources.samples.jsonl"

echo "[1/7] Generate common candidates with ModernBERT MLM"
"$PYTHON_BIN" -u mvp/generate_common_mlm_candidates.py \
  --tag-files "$COMMON_TAGS" --db "$DB" --model "$MODEL" \
  --out "$RUN_ROOT/common/pairs_mlm_candidates.jsonl" \
  --summary-out "$RUN_ROOT/common/generation_summary.json" \
  --top-k "${COMMON_TOP_K:-200}" \
  --candidates-per-mention "${COMMON_CANDIDATES:-8}" \
  --mentions-per-entity "${COMMON_MENTIONS:-2}" \
  --device "${DEVICE:-auto}"

echo "[2/7] Ask Mistral for contextual positives and hard negatives"
"$PYTHON_BIN" -u mvp/generate_common_candidates_mistral.py \
  --input "$RUN_ROOT/common/pairs_mlm_candidates.jsonl" \
  --out "$RUN_ROOT/common/pairs_agent_candidates.jsonl" \
  --raw-out "$RUN_ROOT/common/pairs_agent_candidates.raw.jsonl" \
  --model "$MISTRAL_MODEL" \
  --batch-size "${COMMON_AGENT_BATCH_SIZE:-5}" \
  --retries "${LLM_RETRIES:-5}"
"$PYTHON_BIN" mvp/merge_pair_files.py \
  --input "$RUN_ROOT/common/pairs_mlm_candidates.jsonl" "$RUN_ROOT/common/pairs_agent_candidates.jsonl" \
  --out "$RUN_ROOT/common/pairs.jsonl"

echo "[3/7] Generate numeric pairs with the local short-window rules"
"$PYTHON_BIN" -u mvp/generate_numeric_context_pairs.py \
  --tag-files "${TAG_FILES[@]}" --db "$DB" \
  --out "$RUN_ROOT/numeric/pairs.jsonl" \
  --summary-out "$RUN_ROOT/numeric/generation_summary.json" \
  --mentions-per-entity "${NUMERIC_MENTIONS:-1}"

echo "[4/7] Compute MLM fit for both branches"
for branch in common numeric; do
  "$PYTHON_BIN" -u mvp/score_pairs_mlm_fit.py \
    --input "$RUN_ROOT/$branch/pairs.jsonl" \
    --out "$RUN_ROOT/$branch/pairs_mlm_scored.jsonl" \
    --summary-out "$RUN_ROOT/$branch/mlm_fit_summary.json" \
    --model "$MODEL" --batch-size "${MLM_BATCH_SIZE:-32}" --device "${DEVICE:-auto}"
done

echo "[5/7] Judge in parallel with Mistral and ${SECOND_JUDGE_PROVIDER}"
set -a
[[ -f "$ROOT/mvp/.env" ]] && source "$ROOT/mvp/.env"
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"
set +a
if [[ -n "${MISTRAL_API_KEY_2:-}" ]]; then export MISTRAL_API_KEY="$MISTRAL_API_KEY_2"; fi

judge_pids=()
for branch in common numeric; do
  "$PYTHON_BIN" -u mvp/judge_substitution_pairs_mistral.py \
    --input-jsonl "$RUN_ROOT/$branch/pairs_mlm_scored.jsonl" \
    --out "$RUN_ROOT/$branch/judge_mistral.jsonl" \
    --summary-out "$RUN_ROOT/$branch/judge_mistral_summary.json" \
    --warnings-out "$RUN_ROOT/$branch/judge_mistral_warnings.jsonl" \
    --model "$MISTRAL_MODEL" --batch-size "${JUDGE_BATCH_SIZE:-8}" --resume \
    --retries "${LLM_RETRIES:-5}" &
  judge_pids+=("$!")

  provider_args=(--provider "$SECOND_JUDGE_PROVIDER")
  [[ -n "$SECOND_JUDGE_MODEL" ]] && provider_args+=(--model "$SECOND_JUDGE_MODEL")
  "$PYTHON_BIN" -u mvp/judge_substitution_pairs_provider.py \
    "${provider_args[@]}" \
    --input-jsonl "$RUN_ROOT/$branch/pairs_mlm_scored.jsonl" \
    --out "$RUN_ROOT/$branch/judge_second.jsonl" \
    --summary-out "$RUN_ROOT/$branch/judge_second_summary.json" \
    --warnings-out "$RUN_ROOT/$branch/judge_second_warnings.jsonl" \
    --batch-size "${SECOND_JUDGE_BATCH_SIZE:-8}" \
    --max-tokens "${SECOND_JUDGE_MAX_TOKENS:-1800}" \
    --schema-mode json_schema --resume \
    --retries "${LLM_RETRIES:-5}" &
  judge_pids+=("$!")
done
for pid in "${judge_pids[@]}"; do
  wait "$pid"
done

echo "[6/7] Keep only two-judge consensus for training"
for branch in common numeric; do
  "$PYTHON_BIN" mvp/merge_judges_consensus.py \
    --judged "$RUN_ROOT/$branch/judge_mistral.jsonl" "$RUN_ROOT/$branch/judge_second.jsonl" \
    --out-final "$RUN_ROOT/$branch/consensus.jsonl" \
    --out-train "$RUN_ROOT/$branch/train_consensus.jsonl" \
    --summary-out "$RUN_ROOT/$branch/consensus_summary.json" \
    --positive-votes 2 --negative-votes 2
  "$PYTHON_BIN" mvp/audit_semantic_dataset.py \
    --input "${branch}_train=$RUN_ROOT/$branch/train_consensus.jsonl" \
    --out "$RUN_ROOT/audit/${branch}_train.json" \
    --sample-out "$RUN_ROOT/audit/${branch}_train.samples.jsonl" \
    --require-both-labels-by candidate_kind --min-group-rows 5 --fail-on-critical
done

echo "[7/7] Build an entity/document-disjoint corpus"
"$PYTHON_BIN" mvp/build_semantic_corpus.py \
  --input "proper=$PROPER_TRAIN" \
  --input "common_existing=$COMMON_EXISTING" \
  --input "common_mlm=$RUN_ROOT/common/train_consensus.jsonl" \
  --input "numeric=$RUN_ROOT/numeric/train_consensus.jsonl" \
  --output-dir "$RUN_ROOT/corpus" --seed "${SPLIT_SEED:-41}"

"$PYTHON_BIN" -u mvp/audit_semantic_dataset.py \
  --input "corpus=$RUN_ROOT/corpus/all.jsonl" \
  --out "$RUN_ROOT/audit/final_corpus.json" \
  --sample-out "$RUN_ROOT/audit/final_corpus.samples.jsonl" \
  --require-both-labels-by coarse_group --min-group-rows 20 --fail-on-critical

echo "Pipeline complete: $RUN_ROOT"
