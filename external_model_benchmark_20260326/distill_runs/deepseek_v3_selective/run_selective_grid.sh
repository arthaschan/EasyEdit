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

TRAIN_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
SYSTEM_PROMPT="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/prompt_system.txt"
BASE_MODEL="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
TEACHER_CANDIDATE="$DISTILL_ROOT/deepseek_v3/teacher_candidate.json"
TEACHER_DATA_RAW="$RUN_ROOT/artifacts/teacher_train_raw.jsonl"
TEACHER_DATA_SELECTIVE="$RUN_ROOT/artifacts/teacher_train_selective.jsonl"
TEACHER_DATA_CLEAN="$RUN_ROOT/artifacts/teacher_train_clean.jsonl"
TEACHER_DATA_MISMATCH="$RUN_ROOT/artifacts/teacher_train_mismatch.jsonl"

MASTER_LOG="$RUN_ROOT/logs/selective_master.log"

{
  echo "[INFO] RUN_ID=$RUN_ID"
  echo "[STEP] generate teacher labels (DeepSeek)"
  "$PY" "$DISTILL_ROOT/shared/generate_teacher_labels.py" \
    --dataset "$TRAIN_DATA" \
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

  echo "[STEP] prepare selective dataset"
  "$PY" "$ROOT_DIR/shared/prepare_selective_dataset.py" \
    --teacher_data "$TEACHER_DATA_RAW" \
    --output_train "$TEACHER_DATA_SELECTIVE" \
    --output_clean "$TEACHER_DATA_CLEAN" \
    --output_mismatch "$TEACHER_DATA_MISMATCH"

  LINES=$(wc -l < "$TEACHER_DATA_SELECTIVE")
  echo "[INFO] selective_train_lines=$LINES"
  if [[ "$LINES" -le 0 ]]; then
    echo "[FATAL] selective data empty"
    exit 2
  fi

  echo "[STEP] train grid combos"
  "$PY" - "$ROOT_DIR/grid_params.json" "$RUN_ROOT" "$PROJECT_ROOT" "$BASE_MODEL" "$TEACHER_DATA_SELECTIVE" "$VAL_DATA" "$TEST_DATA" "$PY" <<'PY2'
import json
import subprocess
import sys
from pathlib import Path

params_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
project_root = Path(sys.argv[3])
base_model = sys.argv[4]
train_data = sys.argv[5]
val_data = sys.argv[6]
test_data = sys.argv[7]
py = sys.argv[8]

params = json.loads(params_path.read_text(encoding='utf-8'))
for p in params:
    name = p['name']
    out_dir = run_root / 'outputs' / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_root / 'logs' / f'train_{name}.log'
    cmd = [
        py,
        str(project_root / 'train_dental_lora7_deepseek.py'),
        '--model_name', base_model,
        '--data_path', train_data,
        '--val_path', val_data,
        '--test_path', test_data,
        '--output_dir', str(out_dir),
        '--num_epochs', str(p['num_epochs']),
        '--batch_size', str(p['batch_size']),
        '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
        '--learning_rate', str(p['learning_rate']),
        '--rank', str(p['rank']),
        '--lora_alpha', str(p['lora_alpha']),
        '--temperature', str(p['temperature']),
        '--alpha', str(p['alpha']),
        '--resume',
    ]
    print(f"[RUN] {name} alpha={p['alpha']} temp={p['temperature']} lr={p['learning_rate']}", flush=True)
    with log_path.open('w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[FAIL] {name} rc={proc.returncode}", flush=True)
    else:
        print(f"[DONE] {name}", flush=True)
PY2

  "$PY" "$ROOT_DIR/shared/summarize_results.py" \
    --run_root "$RUN_ROOT" \
    --params "$ROOT_DIR/grid_params.json" \
    --baseline 77.11

  echo "[DONE] selective grid finished"
} 2>&1 | tee "$MASTER_LOG"

echo "[OUT] run_root=$RUN_ROOT"
