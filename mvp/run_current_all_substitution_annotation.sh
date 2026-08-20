#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/tmp/airi_mars_torch_venv/bin/python}"
OUT_DIR="$ROOT/data/substitution_annotation_current_all"

cd "$ROOT"

env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_mistral.py" \
  --input-jsonl "$OUT_DIR/probe_pairs.jsonl" \
  --out "$OUT_DIR/judge_mistral.jsonl" \
  --batch-size 8 \
  --sleep 0.7 \
  --max-tokens 4500 \
  --resume \
  --retries 5 \
  --retry-sleep 15 \
  --warnings-out "$OUT_DIR/judge_warnings.jsonl" \
  --summary-out "$OUT_DIR/judge_mistral_summary.json"

env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

out_dir = Path("data/substitution_annotation_current_all")
rows = [json.loads(line) for line in (out_dir / "judge_mistral.jsonl").open(encoding="utf-8") if line.strip()]
missing = [row for row in rows if row.get("judge_label") == "missing"]
with (out_dir / "missing_pairs_for_repair.jsonl").open("w", encoding="utf-8") as out:
    for row in missing:
        clean = {key: value for key, value in row.items() if not key.startswith("judge_")}
        out.write(json.dumps(clean, ensure_ascii=False) + "\n")
print(f"missing_for_repair={len(missing)}")
PY

if [[ -s "$OUT_DIR/missing_pairs_for_repair.jsonl" ]]; then
  rm -f "$OUT_DIR/judge_mistral_repair.jsonl" "$OUT_DIR/judge_repair_summary.json" "$OUT_DIR/judge_repair_warnings.jsonl"
  env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/judge_substitution_pairs_mistral.py" \
    --input-jsonl "$OUT_DIR/missing_pairs_for_repair.jsonl" \
    --out "$OUT_DIR/judge_mistral_repair.jsonl" \
    --batch-size 4 \
    --sleep 0.7 \
    --max-tokens 3000 \
    --retries 5 \
    --retry-sleep 15 \
    --warnings-out "$OUT_DIR/judge_repair_warnings.jsonl" \
    --summary-out "$OUT_DIR/judge_repair_summary.json"
else
  : > "$OUT_DIR/judge_mistral_repair.jsonl"
fi

env PYTHONPATH="$ROOT/mvp" "$PYTHON_BIN" -u "$ROOT/mvp/finalize_substitution_judgments.py" \
  --raw "$OUT_DIR/judge_mistral.jsonl" \
  --repair "$OUT_DIR/judge_mistral_repair.jsonl" \
  --out-final "$OUT_DIR/judge_mistral_final.jsonl" \
  --out-train "$OUT_DIR/train_pairs_mistral_judged.jsonl" \
  --summary-out "$OUT_DIR/judge_mistral_final_summary.json" \
  --report-out "$OUT_DIR/annotation_report.md"
