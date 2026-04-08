#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$ROOT_DIR/../.." && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
MODEL_PATH="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
PY="/home/student/arthas/EasyEdit3/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT_DIR/logs/distill_train_${TS}.log"
TEACHER_DATA="$ROOT_DIR/artifacts/teacher_train.jsonl"
STUDENT_OUT="$ROOT_DIR/outputs/student_qwen25_7b_from_doubao"

TRAIN_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
SYSTEM_PROMPT="$BENCH_DIR/prompt_system.txt"

{
  echo "[STEP] Generate teacher-labeled dataset (doubao)"
  "$PY" "$BENCH_DIR/distill_runs/shared/generate_teacher_labels.py" \
    --dataset "$TRAIN_DATA" \
    --candidate "$ROOT_DIR/teacher_candidate.json" \
    --system_prompt "$SYSTEM_PROMPT" \
    --output "$TEACHER_DATA" \
    --sample_size 0 \
    --seed 42 \
    --timeout_sec 120 \
    --max_tokens 16 \
    --max_retries 2 \
    --request_interval_sec 0.8 \
    --resume

  LINES=$(wc -l < "$TEACHER_DATA")
  echo "[INFO] teacher_data_lines=$LINES"
  if [[ "$LINES" -le 0 ]]; then
    echo "[FATAL] teacher data is empty, abort training"
    exit 2
  fi

  echo "[STEP] Train student Qwen2.5-7B-Instruct from teacher labels"
  "$PY" "$PROJECT_ROOT/train_dental_lora7_doubao.py" \
    --model_name "$MODEL_PATH" \
    --data_path "$TEACHER_DATA" \
    --val_path "$VAL_DATA" \
    --test_path "$TEST_DATA" \
    --output_dir "$STUDENT_OUT" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --rank 16 \
    --lora_alpha 32 \
    --temperature 2.0 \
    --alpha 0.5 \
    --resume

  echo "[DONE] teacher_data=$TEACHER_DATA"
} 2>&1 | tee "$LOG"
