#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
PY="${EASYEDIT_PY:-/home/student/arthas/EasyEdit3/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

RUN_BASE_ID="$(date +%Y%m%d_%H%M%S)"
RUN_REPRO_ROOT="$ROOT_DIR/runs/${RUN_BASE_ID}_repro3"
RUN_ALPHA_ROOT="$ROOT_DIR/runs/${RUN_BASE_ID}_alpha_low"
mkdir -p "$RUN_REPRO_ROOT/logs" "$RUN_REPRO_ROOT/outputs" "$RUN_ALPHA_ROOT/logs" "$RUN_ALPHA_ROOT/outputs"

TRAIN_DATA_GT="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
FIXED_SELECTIVE="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective/runs/20260407_094814/artifacts/teacher_train_selective.jsonl"

if [[ ! -f "$FIXED_SELECTIVE" ]]; then
  echo "[ERROR] fixed selective labels not found: $FIXED_SELECTIVE"
  exit 2
fi

run_grid() {
  local params_json="$1"
  local run_root="$2"
  local tag="$3"

  echo "[INFO] start grid: $tag"
  "$PY" - "$params_json" "$run_root" "$PROJECT_ROOT" "$BASE_MODEL" "$TRAIN_DATA_GT" "$FIXED_SELECTIVE" "$VAL_DATA" "$TEST_DATA" "$PY" "$tag" <<'PY2'
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
tag = sys.argv[10]

params = json.loads(params_path.read_text(encoding='utf-8'))
for p in params:
    name = p['name']
    out_dir = run_root / 'outputs' / name
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_log = run_root / 'logs' / f'stage1_{name}.log'
    stage2_log = run_root / 'logs' / f'stage2_{name}.log'

    stage1 = [
        py,
        str(project_root / 'train_dental_lora7_deepseek.py'),
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
        py,
        str(project_root / 'train_dental_lora7_deepseek.py'),
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

    print(f"[RUN][{tag}][stage1] {name}", flush=True)
    with stage1_log.open('w', encoding='utf-8') as lf:
        r1 = subprocess.run(stage1, stdout=lf, stderr=subprocess.STDOUT)
    if r1.returncode != 0:
        print(f"[FAIL][{tag}][stage1] {name} rc={r1.returncode}", flush=True)
        continue

    print(f"[RUN][{tag}][stage2] {name}", flush=True)
    with stage2_log.open('w', encoding='utf-8') as lf:
        r2 = subprocess.run(stage2, stdout=lf, stderr=subprocess.STDOUT)
    if r2.returncode != 0:
        print(f"[FAIL][{tag}][stage2] {name} rc={r2.returncode}", flush=True)
    else:
        print(f"[DONE][{tag}] {name}", flush=True)
PY2
}

MASTER_LOG="$ROOT_DIR/runs/${RUN_BASE_ID}_followup_master.log"
{
  echo "[INFO] RUN_BASE_ID=$RUN_BASE_ID"
  echo "[INFO] fixed_teacher_labels=$FIXED_SELECTIVE"

  run_grid "$ROOT_DIR/grid_params_round5_repro3.json" "$RUN_REPRO_ROOT" "repro3"
  "$PY" "$ROOT_DIR/shared/summarize_two_stage_results.py" \
    --run_root "$RUN_REPRO_ROOT" \
    --params "$ROOT_DIR/grid_params_round5_repro3.json" \
    --baseline 77.11

  run_grid "$ROOT_DIR/grid_params_round5_alpha_low.json" "$RUN_ALPHA_ROOT" "alpha_low"
  "$PY" "$ROOT_DIR/shared/summarize_two_stage_results.py" \
    --run_root "$RUN_ALPHA_ROOT" \
    --params "$ROOT_DIR/grid_params_round5_alpha_low.json" \
    --baseline 77.11

  echo "[DONE] followup training finished"
  echo "[OUT] repro_root=$RUN_REPRO_ROOT"
  echo "[OUT] alpha_root=$RUN_ALPHA_ROOT"
} 2>&1 | tee "$MASTER_LOG"

echo "[OUT] master_log=$MASTER_LOG"
