#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
PY="${EASYEDIT_PY:-/home/student/arthas/EasyEdit3/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_doubao_round1"
RUN_ROOT="$ROOT_DIR/runs/$RUN_ID"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/artifacts" "$RUN_ROOT/outputs"

TRAIN_DATA_GT="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
TEACHER_RAW="$ROOT_DIR/artifacts/teacher_train.jsonl"
TEACHER_SELECTIVE="$RUN_ROOT/artifacts/teacher_train_selective.jsonl"
TEACHER_CLEAN="$RUN_ROOT/artifacts/teacher_train_clean.jsonl"
TEACHER_MISMATCH="$RUN_ROOT/artifacts/teacher_train_mismatch.jsonl"
PARAMS_JSON="$ROOT_DIR/grid_params_two_stage_round1.json"
MASTER_LOG="$RUN_ROOT/logs/two_stage_master.log"

{
  echo "[INFO] RUN_ID=$RUN_ID"

  if [[ ! -f "$TEACHER_RAW" ]]; then
    echo "[ERROR] teacher data missing: $TEACHER_RAW"
    exit 2
  fi

  echo "[STEP] build selective dataset from doubao teacher labels"
  "$PY" "/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective/shared/prepare_selective_dataset.py" \
    --teacher_data "$TEACHER_RAW" \
    --output_train "$TEACHER_SELECTIVE" \
    --output_clean "$TEACHER_CLEAN" \
    --output_mismatch "$TEACHER_MISMATCH"

  echo "[STEP] run two-stage grid"
  "$PY" - "$PARAMS_JSON" "$RUN_ROOT" "$PROJECT_ROOT" "$BASE_MODEL" "$TRAIN_DATA_GT" "$TEACHER_SELECTIVE" "$VAL_DATA" "$TEST_DATA" "$PY" <<'PY2'
import json
import subprocess
import sys
from pathlib import Path

params_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
project_root = Path(sys.argv[3])
base_model = sys.argv[4]
train_gt = sys.argv[5]
train_selective = sys.argv[6]
val_data = sys.argv[7]
test_data = sys.argv[8]
py = sys.argv[9]

params = json.loads(params_path.read_text(encoding='utf-8'))
for p in params:
    name = p['name']
    out_dir = run_root / 'outputs' / name
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_log = run_root / 'logs' / f'stage1_{name}.log'
    stage2_log = run_root / 'logs' / f'stage2_{name}.log'

    stage1 = [
        py, str(project_root / 'train_dental_lora7_doubao.py'),
        '--model_name', base_model,
        '--data_path', train_gt,
        '--val_path', val_data,
        '--test_path', test_data,
        '--output_dir', str(out_dir),
        '--num_epochs', str(p['num_epochs_stage1']),
        '--batch_size', str(p['batch_size']),
        '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
        '--learning_rate', str(p['learning_rate']),
        '--rank', str(p['rank']),
        '--lora_alpha', str(p['lora_alpha']),
        '--temperature', '1.5',
        '--alpha', '0.0',
        '--default_distill_mask', '0',
        '--resume',
    ]

    stage2 = [
        py, str(project_root / 'train_dental_lora7_doubao.py'),
        '--model_name', base_model,
        '--data_path', train_selective,
        '--val_path', val_data,
        '--test_path', test_data,
        '--output_dir', str(out_dir),
        '--num_epochs', str(p['num_epochs_stage2']),
        '--batch_size', str(p['batch_size']),
        '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
        '--learning_rate', str(p['learning_rate']),
        '--rank', str(p['rank']),
        '--lora_alpha', str(p['lora_alpha']),
        '--temperature', str(p['temperature']),
        '--alpha', str(p['alpha']),
        '--default_distill_mask', '1',
        '--resume',
    ]

    print(f"[RUN][stage1] {name}", flush=True)
    with stage1_log.open('w', encoding='utf-8') as lf:
        rc1 = subprocess.run(stage1, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc1 != 0:
        print(f"[FAIL][stage1] {name} rc={rc1}", flush=True)
        continue

    print(f"[RUN][stage2] {name}", flush=True)
    with stage2_log.open('w', encoding='utf-8') as lf:
        rc2 = subprocess.run(stage2, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc2 != 0:
        print(f"[FAIL][stage2] {name} rc={rc2}", flush=True)
    else:
        print(f"[DONE] {name}", flush=True)
PY2

  "$PY" "/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective/shared/summarize_two_stage_results.py" \
    --run_root "$RUN_ROOT" \
    --params "$PARAMS_JSON" \
    --baseline 77.11

  echo "[DONE] doubao two-stage round1 finished"
  echo "[OUT] run_root=$RUN_ROOT"
} 2>&1 | tee "$MASTER_LOG"
