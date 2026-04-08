#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs"
PY="/home/student/arthas/EasyEdit3/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

BASE_MODEL="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="$ROOT/final_reports/distill_accuracy_${TS}.json"
OUT_MD="$ROOT/final_reports/distill_accuracy_${TS}.md"
LATEST_JSON="$ROOT/final_reports/distill_accuracy_latest.json"
LATEST_MD="$ROOT/final_reports/distill_accuracy_latest.md"
LOG="$ROOT/final_reports/wait_and_report_${TS}.log"

mkdir -p "$ROOT/final_reports"

{
  echo "[INFO] waiting for running distillation jobs to finish"
  while true; do
    if ps -ef | grep -E "run_all_teacher_distill.sh|train_dental_lora7.py|generate_teacher_labels.py" | grep -v grep >/dev/null; then
      sleep 20
      continue
    fi
    break
  done

  echo "[INFO] all distillation jobs completed, generating final report"
  "$PY" "$ROOT/shared/report_student_accuracy.py" \
    --base_model "$BASE_MODEL" \
    --test_data "$TEST_DATA" \
    --distill_root "$ROOT" \
    --output_json "$OUT_JSON" \
    --output_md "$OUT_MD"

  cp "$OUT_JSON" "$LATEST_JSON"
  cp "$OUT_MD" "$LATEST_MD"

  echo "[DONE] report_json=$OUT_JSON"
  echo "[DONE] report_md=$OUT_MD"
  echo "[DONE] latest_json=$LATEST_JSON"
  echo "[DONE] latest_md=$LATEST_MD"
} 2>&1 | tee "$LOG"
