#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/tmp/airi_mars_torch_venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/data/proper_name_agent_annotation_longrun}"
DB_PATH="${DB_PATH:-$ROOT/data/entity_inventory.sqlite}"
START_OFFSET="${START_OFFSET:-1000}"
WINDOWS="${WINDOWS:-1}"
LIMIT_PER_SHARD="${LIMIT_PER_SHARD:-250}"
MENTIONS_PER_ENTITY="${MENTIONS_PER_ENTITY:-2}"
SEED="${SEED:-101}"
MISTRAL_KEY_1_NAME="${MISTRAL_KEY_1_NAME:-MISTRAL_API_KEY}"
MISTRAL_KEY_2_NAME="${MISTRAL_KEY_2_NAME:-MISTRAL_API_KEY_2}"
MISTRAL_REPAIR_KEY_NAME="${MISTRAL_REPAIR_KEY_NAME:-$MISTRAL_KEY_2_NAME}"

TAG_FILES=(
  "$ROOT/data/entity_tags_top50k_chunk01.mistral.jsonl"
  "$ROOT/data/entity_tags_top50k_chunk03.openrouter.jsonl"
)

cd "$ROOT"
mkdir -p "$OUT_ROOT"

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
  local out_dir="$1"
  local shard="$2"
  local offset="$3"
  local key_name="$4"
  local shard_dir="$out_dir/shards/$shard"
  local api_key
  mkdir -p "$shard_dir"
  api_key="$(load_env_var "$key_name" || true)"
  if [[ -z "$api_key" ]]; then
    echo "[$(date)] shard=$shard skipped: $key_name is missing" | tee -a "$out_dir/prefill.log"
    return 0
  fi
  echo "[$(date)] shard=$shard generate offset=$offset limit=$LIMIT_PER_SHARD key=$key_name" | tee -a "$out_dir/prefill.log"
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
    > "$shard_dir/generate.prefill.log" 2>&1
  echo "[$(date)] shard=$shard generate done" | tee -a "$out_dir/prefill.log"
}

merge_pairs() {
  local out_dir="$1"
  echo "[$(date)] merge generated shards" | tee -a "$out_dir/prefill.log"
  env OUT_DIR="$out_dir" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
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

split_mistral_judge_inputs() {
  local out_dir="$1"
  echo "[$(date)] split Mistral judge inputs" | tee -a "$out_dir/prefill.log"
  env OUT_DIR="$out_dir" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
judge_dir = out_dir / "judge_inputs"
judge_dir.mkdir(parents=True, exist_ok=True)
paths = [judge_dir / "mistral_key1.jsonl", judge_dir / "mistral_key2.jsonl"]
handles = [path.open("w", encoding="utf-8") for path in paths]
try:
    pair_path = out_dir / "proper_agent_pairs.jsonl"
    for idx, line in enumerate(pair_path.open(encoding="utf-8")):
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
  local out_dir="$1"
  local key_name="$2"
  local input="$3"
  local out="$4"
  local api_key
  api_key="$(load_env_var "$key_name" || true)"
  if [[ -z "$api_key" ]]; then
    echo "[$(date)] Mistral judge skipped: $key_name is missing" | tee -a "$out_dir/prefill.log"
    return 0
  fi
  echo "[$(date)] judge Mistral key=$key_name input=$(basename "$input")" | tee -a "$out_dir/prefill.log"
  env MISTRAL_API_KEY="$api_key" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_mistral.py" \
    --input-jsonl "$input" \
    --out "$out_dir/$out.jsonl" \
    --batch-size 4 \
    --sleep 1.0 \
    --max-tokens 3500 \
    --resume \
    --retries 5 \
    --retry-sleep 20 \
    --warnings-out "$out_dir/${out}_warnings.jsonl" \
    --summary-out "$out_dir/${out}_summary.json" \
    > "$out_dir/${out}.prefill.log" 2>&1
  echo "[$(date)] judge Mistral key=$key_name input=$(basename "$input") done" | tee -a "$out_dir/prefill.log"
}

merge_mistral_judges() {
  local out_dir="$1"
  echo "[$(date)] merge Mistral judge shards" | tee -a "$out_dir/prefill.log"
  : > "$out_dir/judge_mistral.jsonl"
  for path in "$out_dir/judge_mistral_key1.jsonl" "$out_dir/judge_mistral_key2.jsonl"; do
    if [[ -s "$path" ]]; then
      cat "$path" >> "$out_dir/judge_mistral.jsonl"
    fi
  done
  env OUT_DIR="$out_dir" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
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
  local out_dir="$1"
  local api_key
  api_key="$(load_env_var "$MISTRAL_REPAIR_KEY_NAME" || true)"
  if [[ -z "$api_key" ]]; then
    echo "[$(date)] Mistral repair skipped: $MISTRAL_REPAIR_KEY_NAME is missing" | tee -a "$out_dir/prefill.log"
    return 0
  fi
  echo "[$(date)] extract Mistral missing for repair" | tee -a "$out_dir/prefill.log"
  env OUT_DIR="$out_dir" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
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
  if [[ ! -s "$out_dir/judge_mistral_missing_for_repair.jsonl" ]]; then
    echo "[$(date)] Mistral repair skipped: no missing rows" | tee -a "$out_dir/prefill.log"
    return 0
  fi
  echo "[$(date)] repair Mistral missing rows" | tee -a "$out_dir/prefill.log"
  env MISTRAL_API_KEY="$api_key" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_mistral.py" \
    --input-jsonl "$out_dir/judge_mistral_missing_for_repair.jsonl" \
    --out "$out_dir/judge_mistral_repair.jsonl" \
    --batch-size 2 \
    --sleep 1.0 \
    --max-tokens 2500 \
    --resume \
    --retries 6 \
    --retry-sleep 20 \
    --warnings-out "$out_dir/judge_mistral_repair_warnings.jsonl" \
    --summary-out "$out_dir/judge_mistral_repair_summary.json" \
    > "$out_dir/judge_mistral_repair.prefill.log" 2>&1
  echo "[$(date)] merge Mistral repair rows" | tee -a "$out_dir/prefill.log"
  env OUT_DIR="$out_dir" PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
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

echo "[$(date)] start Mistral prefill out_root=$OUT_ROOT start_offset=$START_OFFSET windows=$WINDOWS limit_per_shard=$LIMIT_PER_SHARD mentions_per_entity=$MENTIONS_PER_ENTITY db_path=$DB_PATH" | tee -a "$OUT_ROOT/mistral_prefill.log"

for ((window_idx = 0; window_idx < WINDOWS; window_idx++)); do
  base_offset=$((START_OFFSET + window_idx * LIMIT_PER_SHARD * 2))
  window_name="$(printf 'window_%03d_offset_%06d' "$((base_offset / (LIMIT_PER_SHARD * 2)))" "$base_offset")"
  out_dir="$OUT_ROOT/$window_name"
  mkdir -p "$out_dir/shards"
  if [[ -f "$out_dir/.done" ]]; then
    echo "[$(date)] skip $window_name done_marker_exists=true" | tee -a "$OUT_ROOT/mistral_prefill.log"
    continue
  fi
  if [[ -s "$out_dir/judge_mistral.jsonl" ]] && [[ -s "$out_dir/proper_agent_pairs.jsonl" ]]; then
    pair_count="$(wc -l < "$out_dir/proper_agent_pairs.jsonl" | tr -d ' ')"
    judge_count="$(wc -l < "$out_dir/judge_mistral.jsonl" | tr -d ' ')"
    if (( judge_count >= pair_count )); then
      echo "[$(date)] skip $window_name mistral_prefill_exists=true pairs=$pair_count judged=$judge_count" | tee -a "$OUT_ROOT/mistral_prefill.log"
      continue
    fi
    echo "[$(date)] resume incomplete $window_name pairs=$pair_count judged=$judge_count" | tee -a "$OUT_ROOT/mistral_prefill.log"
  fi
  echo "[$(date)] prefill $window_name" | tee -a "$OUT_ROOT/mistral_prefill.log"
  echo "[$(date)] start Mistral prefill window=$window_name base_offset=$base_offset" > "$out_dir/prefill.log"
  generate_shard "$out_dir" shard00 "$base_offset" "$MISTRAL_KEY_1_NAME" &
  pid0=$!
  generate_shard "$out_dir" shard01 "$((base_offset + LIMIT_PER_SHARD))" "$MISTRAL_KEY_2_NAME" &
  pid1=$!
  wait "$pid0" "$pid1"
  merge_pairs "$out_dir"
  if [[ ! -s "$out_dir/proper_agent_pairs.jsonl" ]]; then
    echo "[$(date)] no generated pairs for $window_name" | tee -a "$OUT_ROOT/mistral_prefill.log" "$out_dir/prefill.log"
    continue
  fi
  split_mistral_judge_inputs "$out_dir"
  judge_mistral_file "$out_dir" "$MISTRAL_KEY_1_NAME" "$out_dir/judge_inputs/mistral_key1.jsonl" judge_mistral_key1 &
  pid_m1=$!
  judge_mistral_file "$out_dir" "$MISTRAL_KEY_2_NAME" "$out_dir/judge_inputs/mistral_key2.jsonl" judge_mistral_key2 &
  pid_m2=$!
  wait "$pid_m1" "$pid_m2"
  merge_mistral_judges "$out_dir"
  repair_mistral_missing "$out_dir"
  echo "[$(date)] prefill done $window_name" | tee -a "$OUT_ROOT/mistral_prefill.log" "$out_dir/prefill.log"
done

echo "[$(date)] all requested Mistral prefill windows done" | tee -a "$OUT_ROOT/mistral_prefill.log"
