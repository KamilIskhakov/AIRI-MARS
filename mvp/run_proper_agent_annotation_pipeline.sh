#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/tmp/airi_mars_torch_venv/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/data/proper_name_agent_annotation_current}"
LIMIT="${LIMIT:-500}"
SEED="${SEED:-71}"

TAG_FILES=(
  "$ROOT/data/entity_tags_top50k_chunk01.mistral.jsonl"
  "$ROOT/data/entity_tags_top50k_chunk03.openrouter.jsonl"
)

cd "$ROOT"
mkdir -p "$OUT_DIR"

echo "[$(date)] generate proper-name candidates limit=$LIMIT"
env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/generate_proper_alias_pairs_mistral.py" \
  --tag-files "${TAG_FILES[@]}" \
  --out "$OUT_DIR/proper_agent_pairs.jsonl" \
  --raw-out "$OUT_DIR/proper_agent_raw.jsonl" \
  --limit "$LIMIT" \
  --batch-size 3 \
  --max-context-chars 6000 \
  --pair-context-chars 1800 \
  --sleep 0.8 \
  --retries 5 \
  --retry-sleep 15 \
  --resume \
  --seed "$SEED"

echo "[$(date)] judge with Mistral"
env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_mistral.py" \
  --input-jsonl "$OUT_DIR/proper_agent_pairs.jsonl" \
  --out "$OUT_DIR/judge_mistral.jsonl" \
  --batch-size 4 \
  --sleep 1.0 \
  --max-tokens 3500 \
  --resume \
  --retries 5 \
  --retry-sleep 20 \
  --warnings-out "$OUT_DIR/judge_mistral_warnings.jsonl" \
  --summary-out "$OUT_DIR/judge_mistral_summary.json"

echo "[$(date)] judge with OpenRouter"
env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_provider.py" \
  --provider openrouter \
  --input-jsonl "$OUT_DIR/proper_agent_pairs.jsonl" \
  --out "$OUT_DIR/judge_openrouter.jsonl" \
  --batch-size 4 \
  --sleep 1.0 \
  --max-tokens 3500 \
  --schema-mode json_schema \
  --resume \
  --retries 5 \
  --retry-sleep 15 \
  --warnings-out "$OUT_DIR/judge_openrouter_warnings.jsonl" \
  --summary-out "$OUT_DIR/judge_openrouter_summary.json"

echo "[$(date)] merge two-judge consensus"
env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/merge_judges_consensus.py" \
  --judged "$OUT_DIR/judge_mistral.jsonl" "$OUT_DIR/judge_openrouter.jsonl" \
  --out-final "$OUT_DIR/consensus_mistral_openrouter.jsonl" \
  --out-train "$OUT_DIR/train_mistral_openrouter.jsonl" \
  --summary-out "$OUT_DIR/consensus_mistral_openrouter_summary.json" \
  --positive-votes 2 \
  --negative-votes 2

echo "[$(date)] extract uncertain cases for Groq tie-break"
PIPELINE_OUT_DIR="$OUT_DIR" env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
out_dir = Path(os.environ["PIPELINE_OUT_DIR"])
rows = [json.loads(line) for line in (out_dir / "consensus_mistral_openrouter.jsonl").open(encoding="utf-8") if line.strip()]
uncertain_path = out_dir / "consensus_uncertain_for_groq.jsonl"
with uncertain_path.open("w", encoding="utf-8") as out:
    for row in rows:
        if row.get("consensus_label") == "uncertain":
            clean = {key: value for key, value in row.items() if key not in {"judge_votes"} and not key.startswith("consensus_")}
            out.write(json.dumps(clean, ensure_ascii=False) + "\n")
print(f"uncertain_for_groq={sum(1 for row in rows if row.get('consensus_label') == 'uncertain')}")
PY

if [[ -s "$OUT_DIR/consensus_uncertain_for_groq.jsonl" ]]; then
  echo "[$(date)] judge uncertain cases with Groq"
  env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_provider.py" \
    --provider groq \
    --input-jsonl "$OUT_DIR/consensus_uncertain_for_groq.jsonl" \
    --out "$OUT_DIR/judge_groq_uncertain.jsonl" \
    --batch-size 2 \
    --sleep 20 \
    --max-tokens 2800 \
    --schema-mode json_schema \
    --resume \
    --retries 6 \
    --retry-sleep 20 \
    --warnings-out "$OUT_DIR/judge_groq_uncertain_warnings.jsonl" \
    --summary-out "$OUT_DIR/judge_groq_uncertain_summary.json"

  echo "[$(date)] merge three-judge consensus"
  env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/merge_judges_consensus.py" \
    --judged "$OUT_DIR/judge_mistral.jsonl" "$OUT_DIR/judge_openrouter.jsonl" "$OUT_DIR/judge_groq_uncertain.jsonl" \
    --out-final "$OUT_DIR/consensus_3judge.jsonl" \
    --out-train "$OUT_DIR/train_3judge.jsonl" \
    --summary-out "$OUT_DIR/consensus_3judge_summary.json" \
    --positive-votes 2 \
    --negative-votes 2
else
  cp "$OUT_DIR/consensus_mistral_openrouter.jsonl" "$OUT_DIR/consensus_3judge.jsonl"
  cp "$OUT_DIR/train_mistral_openrouter.jsonl" "$OUT_DIR/train_3judge.jsonl"
  cp "$OUT_DIR/consensus_mistral_openrouter_summary.json" "$OUT_DIR/consensus_3judge_summary.json"
fi

echo "[$(date)] done"
