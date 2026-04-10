#!/usr/bin/env bash
# Generate teacher labels for 490 new samples using both DeepSeek and Doubao in parallel
set -euo pipefail

cd /home/student/arthas/EasyEdit3

RUN_ROOT="external_model_benchmark_20260326/distill_runs/augmented_distill/runs/20260409_214033_augmented_distill"
PY=".venv/bin/python"
SHARED="external_model_benchmark_20260326/distill_runs/shared"
SYSTEM_PROMPT="external_model_benchmark_20260326/prompt_system.txt"
NEW_SAMPLES="$RUN_ROOT/artifacts/new_samples_490.jsonl"

DS_OUTPUT="$RUN_ROOT/artifacts/deepseek_teacher_new490.jsonl"
DB_OUTPUT="$RUN_ROOT/artifacts/doubao_teacher_new490.jsonl"
DS_CANDIDATE="external_model_benchmark_20260326/distill_runs/deepseek_v3/teacher_candidate.json"
DB_CANDIDATE="external_model_benchmark_20260326/distill_runs/doubao/teacher_candidate.json"

echo "[$(date)] Starting parallel teacher label generation..."

# DeepSeek in background
$PY $SHARED/generate_teacher_labels.py \
  --dataset "$NEW_SAMPLES" \
  --candidate "$DS_CANDIDATE" \
  --system_prompt "$SYSTEM_PROMPT" \
  --output "$DS_OUTPUT" \
  --max_tokens 16 \
  --max_retries 3 \
  --request_interval_sec 0.3 \
  --resume \
  > "$RUN_ROOT/logs/deepseek_labels.log" 2>&1 &
DS_PID=$!

# Doubao in background
$PY $SHARED/generate_teacher_labels.py \
  --dataset "$NEW_SAMPLES" \
  --candidate "$DB_CANDIDATE" \
  --system_prompt "$SYSTEM_PROMPT" \
  --output "$DB_OUTPUT" \
  --max_tokens 16 \
  --max_retries 3 \
  --request_interval_sec 0.3 \
  --resume \
  > "$RUN_ROOT/logs/doubao_labels.log" 2>&1 &
DB_PID=$!

echo "[$(date)] DeepSeek PID=$DS_PID, Doubao PID=$DB_PID"
echo "Waiting for both to finish..."

wait $DS_PID
DS_RC=$?
echo "[$(date)] DeepSeek done rc=$DS_RC, lines=$(wc -l < $DS_OUTPUT)"

wait $DB_PID
DB_RC=$?
echo "[$(date)] Doubao done rc=$DB_RC, lines=$(wc -l < $DB_OUTPUT)"

if [ $DS_RC -ne 0 ] || [ $DB_RC -ne 0 ]; then
  echo "[ERROR] Some generation failed"
  exit 1
fi

echo "[$(date)] Both teachers done!"
echo "DeepSeek: $(wc -l < $DS_OUTPUT) labels"
echo "Doubao:   $(wc -l < $DB_OUTPUT) labels"
