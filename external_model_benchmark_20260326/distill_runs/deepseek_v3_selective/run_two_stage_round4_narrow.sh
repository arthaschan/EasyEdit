#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILL_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
PY="${EASYEDIT_PY:-/home/student/arthas/EasyEdit3/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT_DIR/runs/$RUN_ID"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/artifacts" "$RUN_ROOT/outputs"

TRAIN_DATA_GT="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
SYSTEM_PROMPT="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/prompt_system.txt"
BASE_MODEL="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
TEACHER_CANDIDATE="$DISTILL_ROOT/deepseek_v3/teacher_candidate.json"
TEACHER_DATA_RAW="$RUN_ROOT/artifacts/teacher_train_raw.jsonl"
TEACHER_DATA_SELECTIVE="$RUN_ROOT/artifacts/teacher_train_selective.jsonl"
TEACHER_DATA_CLEAN="$RUN_ROOT/artifacts/teacher_train_clean.jsonl"
TEACHER_DATA_MISMATCH="$RUN_ROOT/artifacts/teacher_train_mismatch.jsonl"

PARAMS_JSON="$ROOT_DIR/grid_params_round4_narrow.json"
MASTER_LOG="$RUN_ROOT/logs/two_stage_master.log"

{
  echo "[INFO] RUN_ID=$RUN_ID"
  echo "[INFO] MODE=round4_narrow"

  echo "[STEP] generate teacher labels"
  "$PY" "$DISTILL_ROOT/shared/generate_teacher_labels.py" \
    --dataset "$TRAIN_DATA_GT" \
    --candidate "$TEACHER_CANDIDATE" \
    --system_prompt "$SYSTEM_PROMPT" \
    --output "$TEACHER_DATA_RAW" \
    --sample_size 0 \
    --seed 42 \
    --timeout_sec 120 \
    --max_tokens 16 \
    --max_retries 2 \
    --request_interval_sec 0.8 \
    --resume

  echo "[STEP] build selective dataset"
  "$PY" "$ROOT_DIR/shared/prepare_selective_dataset.py" \
    --teacher_data "$TEACHER_DATA_RAW" \
    --output_train "$TEACHER_DATA_SELECTIVE" \
    --output_clean "$TEACHER_DATA_CLEAN" \
    --output_mismatch "$TEACHER_DATA_MISMATCH"

  echo "[STEP] run two-stage narrow grid"
  "$PY" - "$PARAMS_JSON" "$RUN_ROOT" "$PROJECT_ROOT" "$BASE_MODEL" "$TRAIN_DATA_GT" "$TEACHER_DATA_SELECTIVE" "$VAL_DATA" "$TEST_DATA" "$PY" <<'PY2'
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

    print(f"[RUN][stage1] {name}", flush=True)
    with stage1_log.open('w', encoding='utf-8') as lf:
        r1 = subprocess.run(stage1, stdout=lf, stderr=subprocess.STDOUT)
    if r1.returncode != 0:
        print(f"[FAIL][stage1] {name} rc={r1.returncode}", flush=True)
        continue

    print(f"[RUN][stage2] {name} alpha={p['alpha']} temp={p['temperature']} lr={p['learning_rate']}", flush=True)
    with stage2_log.open('w', encoding='utf-8') as lf:
        r2 = subprocess.run(stage2, stdout=lf, stderr=subprocess.STDOUT)
    if r2.returncode != 0:
        print(f"[FAIL][stage2] {name} rc={r2.returncode}", flush=True)
    else:
        print(f"[DONE] {name}", flush=True)
PY2

  "$PY" "$ROOT_DIR/shared/summarize_two_stage_results.py" \
    --run_root "$RUN_ROOT" \
    --params "$PARAMS_JSON" \
    --baseline 77.11

  echo "[DONE] two-stage round4 narrow finished"
} 2>&1 | tee "$MASTER_LOG"

echo "[OUT] run_root=$RUN_ROOT"
