#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/tmp/airi_mars_torch_venv/bin/python}"
LOG_DIR="$ROOT/data/tagger_logs"

mkdir -p "$LOG_DIR"

launch() {
  local name="$1"
  shift
  local session="airi_${name//./_}"
  local log="$LOG_DIR/${name}.log"
  local session_file="$LOG_DIR/${name}.screen"
  local job_script="$LOG_DIR/${name}.cmd.sh"
  local out_path=""
  local previous_arg=""

  for arg in "$@"; do
    if [[ "$previous_arg" == "--out" ]]; then
      out_path="$arg"
      break
    fi
    previous_arg="$arg"
  done

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$ROOT"
    printf 'exec env PYTHONPATH=%q %q -u' "$ROOT/mvp" "$PYTHON_BIN"
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  } >"$job_script"
  chmod +x "$job_script"

  screen -S "$session" -X quit >/dev/null 2>&1 || true
  if [[ -n "$out_path" ]]; then
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      kill "$pid" >/dev/null 2>&1 || true
    done < <(pgrep -f -- "$out_path" || true)
    sleep 1
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      kill -9 "$pid" >/dev/null 2>&1 || true
    done < <(pgrep -f -- "$out_path" || true)
  fi
  : >"$log"
  screen -dmS "$session" bash -lc "$(printf 'exec %q >>%q 2>&1' "$job_script" "$log")"
  echo "$session" >"$session_file"
  printf '%s screen=%s log=%s\n' "$name" "$session" "$log"
}

launch "chunk00.cerebras" \
  "$ROOT/mvp/tag_entities_providers.py" \
  --provider cerebras \
  --inventory "$ROOT/data/chunks/entity_inventory_top50k_chunk00_000001_012500.jsonl" \
  --out "$ROOT/data/entity_tags_top50k_chunk00.cerebras.jsonl" \
  --limit 12500 \
  --batch-size 10 \
  --sleep 10 \
  --max-tokens 2500 \
  --timeout 120 \
  --schema-mode json_schema \
  --resume \
  --retries 4 \
  --retry-sleep 15 \
  --warnings-out "$ROOT/data/entity_tags_top50k_chunk00.cerebras.warnings.jsonl"

launch "chunk01.mistral" \
  "$ROOT/mvp/tag_entities_mistral.py" \
  --inventory "$ROOT/data/chunks/entity_inventory_top50k_chunk01_012501_025000.jsonl" \
  --out "$ROOT/data/entity_tags_top50k_chunk01.mistral.jsonl" \
  --limit 12500 \
  --batch-size 10 \
  --sleep 1 \
  --max-tokens 2500 \
  --resume \
  --retries 4 \
  --retry-sleep 5 \
  --warnings-out "$ROOT/data/entity_tags_top50k_chunk01.mistral.warnings.jsonl"

launch "chunk02.groq_gptoss" \
  "$ROOT/mvp/tag_entities_providers.py" \
  --provider groq \
  --inventory "$ROOT/data/chunks/entity_inventory_top50k_chunk02_025001_037500.jsonl" \
  --out "$ROOT/data/entity_tags_top50k_chunk02.groq_gptoss.jsonl" \
  --limit 12500 \
  --batch-size 5 \
  --sleep 15 \
  --max-tokens 1800 \
  --timeout 120 \
  --schema-mode json_schema \
  --resume \
  --retries 4 \
  --retry-sleep 15 \
  --warnings-out "$ROOT/data/entity_tags_top50k_chunk02.groq_gptoss.warnings.jsonl"

launch "chunk03.openrouter" \
  "$ROOT/mvp/tag_entities_providers.py" \
  --provider openrouter \
  --inventory "$ROOT/data/chunks/entity_inventory_top50k_chunk03_037501_050000.jsonl" \
  --out "$ROOT/data/entity_tags_top50k_chunk03.openrouter.jsonl" \
  --limit 12500 \
  --batch-size 10 \
  --sleep 2 \
  --max-tokens 2500 \
  --timeout 120 \
  --schema-mode json_schema \
  --resume \
  --retries 4 \
  --retry-sleep 5 \
  --warnings-out "$ROOT/data/entity_tags_top50k_chunk03.openrouter.warnings.jsonl"
