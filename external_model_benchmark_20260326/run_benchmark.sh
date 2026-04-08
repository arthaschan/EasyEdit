#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
OUT_DIR="$ROOT_DIR/outputs"
mkdir -p "$LOG_DIR" "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${TS}.log"

DATASET_DEFAULT="$ROOT_DIR/../data/cmexam_dental_choice.jsonl"
DATASET_PATH="${DATASET_PATH:-$DATASET_DEFAULT}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
SEED="${SEED:-42}"
TIMEOUT_SEC="${TIMEOUT_SEC:-90}"
MAX_TOKENS="${MAX_TOKENS:-16}"
SLEEP_SEC="${SLEEP_SEC:-0.0}"
PYTHON_BIN="${PYTHON_BIN:-/home/student/arthas/EasyEdit3/.venv/bin/python}"

cd "$ROOT_DIR"

echo "[INFO] ROOT_DIR=$ROOT_DIR" | tee -a "$LOG_FILE"
echo "[INFO] DATASET_PATH=$DATASET_PATH" | tee -a "$LOG_FILE"
echo "[INFO] SAMPLE_SIZE=$SAMPLE_SIZE" | tee -a "$LOG_FILE"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN" | tee -a "$LOG_FILE"

"$PYTHON_BIN" external_benchmark.py \
  --dataset "$DATASET_PATH" \
  --candidates "$ROOT_DIR/candidates.json" \
  --system_prompt "$ROOT_DIR/prompt_system.txt" \
  --output_dir "$OUT_DIR" \
  --sample_size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --timeout_sec "$TIMEOUT_SEC" \
  --max_tokens "$MAX_TOKENS" \
  --sleep_sec "$SLEEP_SEC" 2>&1 | tee -a "$LOG_FILE"

echo "[INFO] done. log=$LOG_FILE" | tee -a "$LOG_FILE"
