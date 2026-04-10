#!/usr/bin/env bash
# Augmented-data distillation: Generate teacher labels for ALL 1162 samples,
# then run two-stage distillation with DeepSeek and Doubao teachers independently.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
BENCH_DIR="$PROJECT_ROOT/external_model_benchmark_20260326"
SHARED_DIR="$BENCH_DIR/distill_runs/shared"
PY="${EASYEDIT_PY:-$PROJECT_ROOT/.venv/bin/python}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_augmented_distill"
RUN_ROOT="$ROOT_DIR/runs/$RUN_ID"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs" "$RUN_ROOT/artifacts"

# =========== Paths ===========
AUGMENTED_DATA="$PROJECT_ROOT/data/augment/merged_train.jsonl"
SYSTEM_PROMPT="$BENCH_DIR/prompt_system.txt"
VAL_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="$PROJECT_ROOT/Qwen2.5-7B-Instruct"

# Existing teacher labels (672 original CMExam)
DS_TEACHER_EXISTING="$BENCH_DIR/distill_runs/deepseek_v3_selective/runs/20260405_152319/artifacts/teacher_train_selective.jsonl"
DB_TEACHER_EXISTING="$BENCH_DIR/distill_runs/doubao/artifacts/teacher_train.jsonl"

# Candidate configs
DS_CANDIDATE="$BENCH_DIR/distill_runs/deepseek_v3/teacher_candidate.json"
DB_CANDIDATE="$BENCH_DIR/distill_runs/doubao/teacher_candidate.json"

# Output artifacts
DS_TEACHER_NEW="$RUN_ROOT/artifacts/deepseek_teacher_new490.jsonl"
DB_TEACHER_NEW="$RUN_ROOT/artifacts/doubao_teacher_new490.jsonl"
DS_TEACHER_FULL="$RUN_ROOT/artifacts/deepseek_teacher_full_1162.jsonl"
DB_TEACHER_FULL="$RUN_ROOT/artifacts/doubao_teacher_full_1162.jsonl"
DS_TEACHER_SOFT="$RUN_ROOT/artifacts/deepseek_teacher_soft.jsonl"
DB_TEACHER_SOFT="$RUN_ROOT/artifacts/doubao_teacher_soft.jsonl"
DS_TRAIN_HEAD="$RUN_ROOT/artifacts/ds_train_head_distill.jsonl"
DB_TRAIN_HEAD="$RUN_ROOT/artifacts/db_train_head_distill.jsonl"

MASTER_LOG="$RUN_ROOT/logs/master.log"

# Extract new samples (huatuo_converted + deepseek_autogen) not in original CMExam
NEW_SAMPLES="$RUN_ROOT/artifacts/new_samples_490.jsonl"

{
  echo "============================================"
  echo "[INFO] RUN_ID=$RUN_ID"
  echo "[INFO] Augmented distillation: DeepSeek + Doubao teachers on 1162 samples"
  echo "============================================"

  # ---- STEP 0: Extract 490 new-only samples ----
  echo ""
  echo "[STEP 0] Extract new samples (non-cmexam_original)"
  "$PY" -c "
import json
new = []
with open('$AUGMENTED_DATA') as f:
    for line in f:
        d = json.loads(line)
        if d.get('Source') != 'cmexam_original':
            new.append(d)
with open('$NEW_SAMPLES', 'w') as f:
    for d in new:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print(f'Extracted {len(new)} new samples')
"

  # ---- STEP 1: Generate DeepSeek teacher labels for 490 new samples ----
  echo ""
  echo "[STEP 1a] Generate DeepSeek teacher labels for new samples"
  "$PY" "$SHARED_DIR/generate_teacher_labels.py" \
    --dataset "$NEW_SAMPLES" \
    --candidate "$DS_CANDIDATE" \
    --system_prompt "$SYSTEM_PROMPT" \
    --output "$DS_TEACHER_NEW" \
    --max_tokens 16 \
    --max_retries 3 \
    --request_interval_sec 0.5 \
    --resume

  # ---- STEP 2: Generate Doubao teacher labels for 490 new samples ----
  echo ""
  echo "[STEP 1b] Generate Doubao teacher labels for new samples"
  "$PY" "$SHARED_DIR/generate_teacher_labels.py" \
    --dataset "$NEW_SAMPLES" \
    --candidate "$DB_CANDIDATE" \
    --system_prompt "$SYSTEM_PROMPT" \
    --output "$DB_TEACHER_NEW" \
    --max_tokens 16 \
    --max_retries 3 \
    --request_interval_sec 0.5 \
    --resume

  # ---- STEP 3: Merge existing + new teacher labels ----
  echo ""
  echo "[STEP 2] Merge teacher labels: existing 672 + new 490"
  "$PY" "$ROOT_DIR/merge_teacher_labels.py" \
    --augmented_data "$AUGMENTED_DATA" \
    --existing_teacher "$DS_TEACHER_EXISTING" \
    --new_teacher "$DS_TEACHER_NEW" \
    --output "$DS_TEACHER_FULL"

  "$PY" "$ROOT_DIR/merge_teacher_labels.py" \
    --augmented_data "$AUGMENTED_DATA" \
    --existing_teacher "$DB_TEACHER_EXISTING" \
    --new_teacher "$DB_TEACHER_NEW" \
    --output "$DB_TEACHER_FULL"

  # ---- STEP 4: Convert hard labels to soft (label smoothing) ----
  echo ""
  echo "[STEP 3] Prepare soft labels (eps=0.25)"
  SOFT_LABEL_SCRIPT="$BENCH_DIR/distill_runs/deepseek_opus/prepare_soft_labels.py"

  "$PY" "$SOFT_LABEL_SCRIPT" \
    --input "$DS_TEACHER_FULL" \
    --output "$DS_TEACHER_SOFT" \
    --smooth_eps 0.25

  "$PY" "$SOFT_LABEL_SCRIPT" \
    --input "$DB_TEACHER_FULL" \
    --output "$DB_TEACHER_SOFT" \
    --smooth_eps 0.25

  # ---- STEP 5: Build head-distill datasets ----
  echo ""
  echo "[STEP 4] Build selective distill datasets"
  "$PY" "$SHARED_DIR/build_selective_distill_dataset.py" \
    --gt_data "$AUGMENTED_DATA" \
    --teacher_soft "$DS_TEACHER_SOFT" \
    --output "$DS_TRAIN_HEAD" \
    --report "$RUN_ROOT/artifacts/ds_head_report.json" \
    --min_entropy 0.20 \
    --smooth_eps 0.25 \
    --min_margin 0.03

  "$PY" "$SHARED_DIR/build_selective_distill_dataset.py" \
    --gt_data "$AUGMENTED_DATA" \
    --teacher_soft "$DB_TEACHER_SOFT" \
    --output "$DB_TRAIN_HEAD" \
    --report "$RUN_ROOT/artifacts/db_head_report.json" \
    --min_entropy 0.20 \
    --smooth_eps 0.25 \
    --min_margin 0.03

  # ---- STEP 6: Two-stage training ----
  echo ""
  echo "[STEP 5] Two-stage training: DeepSeek teacher"
  "$PY" "$ROOT_DIR/run_two_stage_training.py" \
    --params "$ROOT_DIR/grid_params_augmented_distill.json" \
    --run_root "$RUN_ROOT" \
    --project_root "$PROJECT_ROOT" \
    --base_model "$BASE_MODEL" \
    --train_head "$DS_TRAIN_HEAD" \
    --train_gt "$AUGMENTED_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --teacher_prefix "ds" \
    --py "$PY"

  echo ""
  echo "[STEP 6] Two-stage training: Doubao teacher"
  "$PY" "$ROOT_DIR/run_two_stage_training.py" \
    --params "$ROOT_DIR/grid_params_augmented_distill.json" \
    --run_root "$RUN_ROOT" \
    --project_root "$PROJECT_ROOT" \
    --base_model "$BASE_MODEL" \
    --train_head "$DB_TRAIN_HEAD" \
    --train_gt "$AUGMENTED_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --teacher_prefix "db" \
    --py "$PY"

  echo ""
  echo "============================================"
  echo "[DONE] All training complete"
  echo "Results in: $RUN_ROOT"
  echo "============================================"

} 2>&1 | tee "$MASTER_LOG"
