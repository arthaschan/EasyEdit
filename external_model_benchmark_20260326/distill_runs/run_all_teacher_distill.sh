#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs"
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$ROOT/run_all_${TS}.log"

FAILED=0

run_one() {
  local name="$1"
  local script="$2"
  echo "[START] teacher=$name"
  if bash "$script"; then
    echo "[DONE] teacher=$name"
  else
    echo "[FAIL] teacher=$name"
    FAILED=1
  fi
}

{
  run_one "DeepSeek-V3" "$ROOT/deepseek_v3/run_distill_train.sh"
  run_one "moonshot-v1-32k" "$ROOT/moonshot_v1_32k/run_distill_train.sh"
  run_one "doubao" "$ROOT/doubao/run_distill_train.sh"

  if [[ "$FAILED" -eq 0 ]]; then
    echo "[DONE] all teacher distillation runs finished"
  else
    echo "[DONE_WITH_FAILURE] some teacher runs failed"
    exit 1
  fi
} 2>&1 | tee "$MASTER_LOG"
