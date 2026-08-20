#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/tmp/airi_mars_torch_venv/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/data/proper_name_agent_annotation_strict_parallel}"
DB_PATH="${DB_PATH:-$ROOT/data/entity_inventory.sqlite}"
LIMIT_PER_SHARD="${LIMIT_PER_SHARD:-250}"
BASE_OFFSET="${BASE_OFFSET:-0}"
MENTIONS_PER_ENTITY="${MENTIONS_PER_ENTITY:-1}"
SEED="${SEED:-101}"
RUN_GROQ="${RUN_GROQ:-0}"
RUN_OPENROUTER="${RUN_OPENROUTER:-1}"
MISTRAL_KEY_1_NAME="${MISTRAL_KEY_1_NAME:-MISTRAL_API_KEY}"
MISTRAL_KEY_2_NAME="${MISTRAL_KEY_2_NAME:-MISTRAL_API_KEY_2}"
MISTRAL_REPAIR_KEY_NAME="${MISTRAL_REPAIR_KEY_NAME:-$MISTRAL_KEY_2_NAME}"
CONSENSUS_POSITIVE_VOTES="${CONSENSUS_POSITIVE_VOTES:-2}"
CONSENSUS_NEGATIVE_VOTES="${CONSENSUS_NEGATIVE_VOTES:-2}"
VALIDATE_EXISTING_PAIRS="${VALIDATE_EXISTING_PAIRS:-0}"

TAG_FILES=(
  "$ROOT/data/entity_tags_top50k_chunk01.mistral.jsonl"
  "$ROOT/data/entity_tags_top50k_chunk03.openrouter.jsonl"
)

cd "$ROOT"
mkdir -p "$OUT_DIR/shards"

load_env_var() {
  local key="$1"
  "$PYTHON_BIN" - "$key" <<'PY'
import sys
from pathlib import Path

key = sys.argv[1]
for path in [Path("mvp/.env"), Path(".env")]:
    if not path.exists():
        continue
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            print(v.strip().strip('"').strip("'"))
            raise SystemExit
PY
}

generate_shard() {
  local shard="$1"
  local offset="$2"
  local key_name="$3"
  local shard_dir="$OUT_DIR/shards/$shard"
  local api_key
  mkdir -p "$shard_dir"
  api_key="$(load_env_var "$key_name" || true)"
  if [[ -z "$api_key" ]]; then
    echo "[$(date)] shard=$shard skipped: $key_name is missing" | tee -a "$OUT_DIR/pipeline.log"
    return 0
  fi
  echo "[$(date)] shard=$shard generate offset=$offset limit=$LIMIT_PER_SHARD key=$key_name" | tee -a "$OUT_DIR/pipeline.log"
  env MISTRAL_API_KEY="$api_key" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/generate_proper_alias_pairs_mistral.py" \
    --tag-files "${TAG_FILES[@]}" \
    --db "$DB_PATH" \
    --out "$shard_dir/proper_agent_pairs.jsonl" \
    --raw-out "$shard_dir/proper_agent_raw.jsonl" \
    --limit "$LIMIT_PER_SHARD" \
    --offset "$offset" \
    --batch-size 3 \
    --max-context-chars 6000 \
    --pair-context-chars 1800 \
    --pool-size 12 \
    --mentions-per-entity "$MENTIONS_PER_ENTITY" \
    --sleep 0.8 \
    --retries 5 \
    --retry-sleep 15 \
    --resume \
    --seed "$SEED" \
    > "$shard_dir/generate.log" 2>&1
  echo "[$(date)] shard=$shard generate done" | tee -a "$OUT_DIR/pipeline.log"
}

merge_pairs() {
  echo "[$(date)] merge generated shards" | tee -a "$OUT_DIR/pipeline.log"
  env OUT_DIR="$OUT_DIR" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
pair_out = out_dir / "proper_agent_pairs.jsonl"
raw_out = out_dir / "proper_agent_raw.jsonl"
pair_id = 0
with pair_out.open("w", encoding="utf-8") as pair_fh:
    for path in sorted((out_dir / "shards").glob("*/proper_agent_pairs.jsonl")):
        shard = path.parent.name
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id += 1
            row["source_pair_id"] = row.get("pair_id")
            row["source_shard"] = shard
            row["pair_id"] = f"pa{pair_id:07d}"
            pair_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
with raw_out.open("w", encoding="utf-8") as raw_fh:
    for path in sorted((out_dir / "shards").glob("*/proper_agent_raw.jsonl")):
        shard = path.parent.name
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            data = row if isinstance(row, dict) else {"raw": row}
            data["source_shard"] = shard
            raw_fh.write(json.dumps(data, ensure_ascii=False) + "\n")
print(f"merged_pairs={pair_id}")
PY
}

judge_provider() {
  local provider="$1"
  local out="$2"
  local sleep_s="$3"
  echo "[$(date)] judge provider=$provider" | tee -a "$OUT_DIR/pipeline.log"
  env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_provider.py" \
    --provider "$provider" \
    --input-jsonl "$OUT_DIR/proper_agent_pairs.jsonl" \
    --out "$OUT_DIR/$out.jsonl" \
    --batch-size 4 \
    --sleep "$sleep_s" \
    --max-tokens 3500 \
    --schema-mode json_schema \
    --resume \
    --retries 5 \
    --retry-sleep 15 \
    --warnings-out "$OUT_DIR/${out}_warnings.jsonl" \
    --summary-out "$OUT_DIR/${out}_summary.json" \
    > "$OUT_DIR/${out}.log" 2>&1
  echo "[$(date)] judge provider=$provider done" | tee -a "$OUT_DIR/pipeline.log"
}

judge_mistral() {
  local key_name="$1"
  local out="$2"
  local api_key
  api_key="$(load_env_var "$key_name" || true)"
  if [[ -z "$api_key" ]]; then
    echo "[$(date)] Mistral judge skipped: $key_name is missing" | tee -a "$OUT_DIR/pipeline.log"
    return 0
  fi
  echo "[$(date)] judge Mistral key=$key_name" | tee -a "$OUT_DIR/pipeline.log"
  env MISTRAL_API_KEY="$api_key" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_mistral.py" \
    --input-jsonl "$OUT_DIR/proper_agent_pairs.jsonl" \
    --out "$OUT_DIR/$out.jsonl" \
    --batch-size 4 \
    --sleep 1.0 \
    --max-tokens 3500 \
    --resume \
    --retries 5 \
    --retry-sleep 20 \
    --warnings-out "$OUT_DIR/${out}_warnings.jsonl" \
    --summary-out "$OUT_DIR/${out}_summary.json" \
    > "$OUT_DIR/${out}.log" 2>&1
  echo "[$(date)] judge Mistral key=$key_name done" | tee -a "$OUT_DIR/pipeline.log"
}

split_mistral_judge_inputs() {
  echo "[$(date)] split Mistral judge inputs" | tee -a "$OUT_DIR/pipeline.log"
  env OUT_DIR="$OUT_DIR" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
judge_dir = out_dir / "judge_inputs"
judge_dir.mkdir(parents=True, exist_ok=True)
paths = [judge_dir / "mistral_key1.jsonl", judge_dir / "mistral_key2.jsonl"]
handles = [path.open("w", encoding="utf-8") for path in paths]
try:
    for idx, line in enumerate((out_dir / "proper_agent_pairs.jsonl").open(encoding="utf-8")):
        if not line.strip():
            continue
        handles[idx % 2].write(line)
finally:
    for handle in handles:
        handle.close()
print("mistral_split_counts", [sum(1 for line in path.open(encoding="utf-8") if line.strip()) for path in paths])
PY
}

judge_mistral_file() {
  local key_name="$1"
  local input="$2"
  local out="$3"
  local api_key
  api_key="$(load_env_var "$key_name" || true)"
  if [[ -z "$api_key" ]]; then
    echo "[$(date)] Mistral judge skipped: $key_name is missing" | tee -a "$OUT_DIR/pipeline.log"
    return 0
  fi
  echo "[$(date)] judge Mistral key=$key_name input=$(basename "$input")" | tee -a "$OUT_DIR/pipeline.log"
  env MISTRAL_API_KEY="$api_key" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_mistral.py" \
    --input-jsonl "$input" \
    --out "$OUT_DIR/$out.jsonl" \
    --batch-size 4 \
    --sleep 1.0 \
    --max-tokens 3500 \
    --resume \
    --retries 5 \
    --retry-sleep 20 \
    --warnings-out "$OUT_DIR/${out}_warnings.jsonl" \
    --summary-out "$OUT_DIR/${out}_summary.json" \
    > "$OUT_DIR/${out}.log" 2>&1
  echo "[$(date)] judge Mistral key=$key_name input=$(basename "$input") done" | tee -a "$OUT_DIR/pipeline.log"
}

merge_mistral_judges() {
  echo "[$(date)] merge Mistral judge shards" | tee -a "$OUT_DIR/pipeline.log"
  : > "$OUT_DIR/judge_mistral.jsonl"
  for path in "$OUT_DIR/judge_mistral_key1.jsonl" "$OUT_DIR/judge_mistral_key2.jsonl"; do
    if [[ -s "$path" ]]; then
      cat "$path" >> "$OUT_DIR/judge_mistral.jsonl"
    fi
  done
  env OUT_DIR="$OUT_DIR" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from judge_substitution_pairs_mistral import summarize

out_dir = Path(os.environ["OUT_DIR"])
summary = summarize(out_dir / "judge_mistral.jsonl")
(out_dir / "judge_mistral_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

repair_mistral_missing() {
  local api_key
  api_key="$(load_env_var "$MISTRAL_REPAIR_KEY_NAME" || true)"
  if [[ -z "$api_key" ]]; then
    echo "[$(date)] Mistral repair skipped: $MISTRAL_REPAIR_KEY_NAME is missing" | tee -a "$OUT_DIR/pipeline.log"
    return 0
  fi
  echo "[$(date)] extract Mistral missing for repair" | tee -a "$OUT_DIR/pipeline.log"
  env OUT_DIR="$OUT_DIR" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
missing_path = out_dir / "judge_mistral_missing_for_repair.jsonl"
count = 0
with (out_dir / "judge_mistral.jsonl").open(encoding="utf-8") as src, missing_path.open("w", encoding="utf-8") as out:
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("judge_label") != "missing":
            continue
        clean = {key: value for key, value in row.items() if not key.startswith("judge_")}
        out.write(json.dumps(clean, ensure_ascii=False) + "\n")
        count += 1
print(f"mistral_missing_for_repair={count}")
PY
  if [[ ! -s "$OUT_DIR/judge_mistral_missing_for_repair.jsonl" ]]; then
    echo "[$(date)] Mistral repair skipped: no missing rows" | tee -a "$OUT_DIR/pipeline.log"
    return 0
  fi
  echo "[$(date)] repair Mistral missing rows" | tee -a "$OUT_DIR/pipeline.log"
  env MISTRAL_API_KEY="$api_key" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_mistral.py" \
    --input-jsonl "$OUT_DIR/judge_mistral_missing_for_repair.jsonl" \
    --out "$OUT_DIR/judge_mistral_repair.jsonl" \
    --batch-size 2 \
    --sleep 1.0 \
    --max-tokens 2500 \
    --resume \
    --retries 6 \
    --retry-sleep 20 \
    --warnings-out "$OUT_DIR/judge_mistral_repair_warnings.jsonl" \
    --summary-out "$OUT_DIR/judge_mistral_repair_summary.json" \
    > "$OUT_DIR/judge_mistral_repair.log" 2>&1
  echo "[$(date)] merge Mistral repair rows" | tee -a "$OUT_DIR/pipeline.log"
  env OUT_DIR="$OUT_DIR" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from judge_substitution_pairs_mistral import summarize

out_dir = Path(os.environ["OUT_DIR"])
repair = {}
repair_path = out_dir / "judge_mistral_repair.jsonl"
if repair_path.exists():
    with repair_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("judge_label") != "missing":
                repair[str(row.get("pair_id"))] = row

merged = []
replaced = 0
with (out_dir / "judge_mistral.jsonl").open(encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        row = json.loads(line)
        pair_id = str(row.get("pair_id"))
        if row.get("judge_label") == "missing" and pair_id in repair:
            merged.append(repair[pair_id])
            replaced += 1
        else:
            merged.append(row)

with (out_dir / "judge_mistral.jsonl").open("w", encoding="utf-8") as out:
    for row in merged:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
summary = summarize(out_dir / "judge_mistral.jsonl")
summary["repair_replaced_missing"] = replaced
(out_dir / "judge_mistral_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

echo "[$(date)] start strict parallel pipeline limit_per_shard=$LIMIT_PER_SHARD base_offset=$BASE_OFFSET mentions_per_entity=$MENTIONS_PER_ENTITY validate_existing_pairs=$VALIDATE_EXISTING_PAIRS" > "$OUT_DIR/pipeline.log"

if [[ "$VALIDATE_EXISTING_PAIRS" == "1" ]]; then
  if [[ ! -s "$OUT_DIR/proper_agent_pairs.jsonl" ]]; then
    echo "[$(date)] validate_existing_pairs requested but proper_agent_pairs.jsonl is missing" | tee -a "$OUT_DIR/pipeline.log"
    exit 0
  fi
  echo "[$(date)] validate existing generated pairs; skip generation" | tee -a "$OUT_DIR/pipeline.log"
else
  generate_shard shard00 "$BASE_OFFSET" "$MISTRAL_KEY_1_NAME" &
  pid0=$!
  generate_shard shard01 "$((BASE_OFFSET + LIMIT_PER_SHARD))" "$MISTRAL_KEY_2_NAME" &
  pid1=$!
  wait "$pid0" "$pid1"

  merge_pairs
fi

pair_count="$(wc -l < "$OUT_DIR/proper_agent_pairs.jsonl" | tr -d ' ')"
if [[ "$pair_count" == "0" ]]; then
  echo "[$(date)] no generated pairs; writing empty outputs" | tee -a "$OUT_DIR/pipeline.log"
  : > "$OUT_DIR/judge_mistral.jsonl"
  : > "$OUT_DIR/judge_openrouter.jsonl"
  : > "$OUT_DIR/consensus.jsonl"
  : > "$OUT_DIR/train_consensus.jsonl"
  : > "$OUT_DIR/train_consensus_clean.jsonl"
  printf '{"total_pairs":0,"train_pairs":0,"consensus_labels":{},"train_labels":{},"by_candidate_kind":{}}\n' > "$OUT_DIR/consensus_summary.json"
  printf '{"input":0,"output":0}\n' > "$OUT_DIR/train_consensus_clean.summary.json"
  env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/audit_annotation_quality.py" \
    --pairs "$OUT_DIR/proper_agent_pairs.jsonl" \
    --consensus "$OUT_DIR/consensus.jsonl" \
    --train "$OUT_DIR/train_consensus_clean.jsonl" \
    --out-json "$OUT_DIR/annotation_quality_summary.json" \
    --sample-out "$OUT_DIR/annotation_quality_samples.jsonl" \
    > "$OUT_DIR/annotation_quality.log" 2>&1
  echo "[$(date)] done" | tee -a "$OUT_DIR/pipeline.log"
  exit 0
fi

split_mistral_judge_inputs
judge_mistral_file "$MISTRAL_KEY_1_NAME" "$OUT_DIR/judge_inputs/mistral_key1.jsonl" judge_mistral_key1 &
pid_mistral1=$!
judge_mistral_file "$MISTRAL_KEY_2_NAME" "$OUT_DIR/judge_inputs/mistral_key2.jsonl" judge_mistral_key2 &
pid_mistral2=$!
if [[ "$RUN_OPENROUTER" == "1" ]]; then
  judge_provider openrouter judge_openrouter 1.0 &
  pid_openrouter=$!
else
  echo "[$(date)] judge provider=openrouter skipped RUN_OPENROUTER=$RUN_OPENROUTER" | tee -a "$OUT_DIR/pipeline.log"
  : > "$OUT_DIR/judge_openrouter.jsonl"
  pid_openrouter=""
fi
if [[ "$RUN_GROQ" == "1" ]]; then
  judge_provider groq judge_groq 20 &
  pid_groq=$!
  wait "$pid_mistral1" "$pid_mistral2"
  merge_mistral_judges
  repair_mistral_missing
  if [[ -n "$pid_openrouter" ]]; then
    wait "$pid_openrouter" "$pid_groq"
  else
    wait "$pid_groq"
  fi
else
  echo "[$(date)] judge provider=groq skipped RUN_GROQ=$RUN_GROQ" | tee -a "$OUT_DIR/pipeline.log"
  wait "$pid_mistral1" "$pid_mistral2"
  merge_mistral_judges
  repair_mistral_missing
  if [[ -n "$pid_openrouter" ]]; then
    wait "$pid_openrouter"
  fi
fi

echo "[$(date)] merge consensus" | tee -a "$OUT_DIR/pipeline.log"
judged_files=()
for candidate in "$OUT_DIR/judge_mistral.jsonl" "$OUT_DIR/judge_openrouter.jsonl"; do
  if [[ -s "$candidate" ]]; then
    judged_files+=("$candidate")
  fi
done

env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/merge_judges_consensus.py" \
  --judged "${judged_files[@]}" \
  --out-final "$OUT_DIR/consensus.jsonl" \
  --out-train "$OUT_DIR/train_consensus.jsonl" \
  --summary-out "$OUT_DIR/consensus_summary.json" \
  --positive-votes "$CONSENSUS_POSITIVE_VOTES" \
  --negative-votes "$CONSENSUS_NEGATIVE_VOTES" \
  > "$OUT_DIR/consensus.log" 2>&1

echo "[$(date)] clean train pairs" | tee -a "$OUT_DIR/pipeline.log"
env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/clean_consensus_train.py" \
  --input "$OUT_DIR/train_consensus.jsonl" \
  --out "$OUT_DIR/train_consensus_clean.jsonl" \
  --summary-out "$OUT_DIR/train_consensus_clean.summary.json" \
  > "$OUT_DIR/train_consensus_clean.log" 2>&1

echo "[$(date)] audit annotation quality" | tee -a "$OUT_DIR/pipeline.log"
env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/audit_annotation_quality.py" \
  --pairs "$OUT_DIR/proper_agent_pairs.jsonl" \
  --consensus "$OUT_DIR/consensus.jsonl" \
  --train "$OUT_DIR/train_consensus_clean.jsonl" \
  --out-json "$OUT_DIR/annotation_quality_summary.json" \
  --sample-out "$OUT_DIR/annotation_quality_samples.jsonl" \
  > "$OUT_DIR/annotation_quality.log" 2>&1

echo "[$(date)] done" | tee -a "$OUT_DIR/pipeline.log"
