#!/usr/bin/env bash
# Continue from where we left off: merge doubao labels, build dataset,
# then run both DeepSeek v2 (lower alpha) and Doubao distillation training.
set -euo pipefail

PROJECT_ROOT="/home/student/arthas/EasyEdit3"
BENCH_DIR="$PROJECT_ROOT/external_model_benchmark_20260326"
SHARED_DIR="$BENCH_DIR/distill_runs/shared"
PY="$PROJECT_ROOT/.venv/bin/python"
DISTILL_DIR="$BENCH_DIR/distill_runs/augmented_distill"

# Reuse existing run root
RUN_ROOT="$DISTILL_DIR/runs/20260409_214033_augmented_distill"

AUGMENTED_DATA="$PROJECT_ROOT/data/augment/merged_train.jsonl"
VAL_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="$PROJECT_ROOT/Qwen2.5-7B-Instruct"

# Existing teacher labels
DB_TEACHER_EXISTING="$BENCH_DIR/distill_runs/doubao/artifacts/teacher_train.jsonl"
DB_TEACHER_NEW="$RUN_ROOT/artifacts/doubao_teacher_new490.jsonl"
DB_TEACHER_FULL="$RUN_ROOT/artifacts/doubao_teacher_full_1162.jsonl"
DB_TEACHER_SOFT="$RUN_ROOT/artifacts/doubao_teacher_soft.jsonl"
DB_TRAIN_HEAD="$RUN_ROOT/artifacts/db_train_head_distill.jsonl"

DS_TRAIN_HEAD="$RUN_ROOT/artifacts/ds_train_head_distill.jsonl"

SOFT_LABEL_SCRIPT="$BENCH_DIR/distill_runs/deepseek_opus/prepare_soft_labels.py"

echo "============================================"
echo "[PHASE 1] Doubao label merge & dataset build"
echo "============================================"

# Verify doubao labels complete
DB_COUNT=$(wc -l < "$DB_TEACHER_NEW")
echo "Doubao new labels: $DB_COUNT/490"
if [ "$DB_COUNT" -lt 490 ]; then
    echo "[ERROR] Doubao labels incomplete ($DB_COUNT/490). Waiting..."
    while [ "$(wc -l < "$DB_TEACHER_NEW")" -lt 490 ]; do
        sleep 30
        echo "  ... $(wc -l < "$DB_TEACHER_NEW")/490"
    done
    echo "  Doubao labels complete!"
fi

# Merge: existing 672 + new 490 → 1162
echo "[STEP 1] Merge doubao teacher labels"
"$PY" "$DISTILL_DIR/merge_teacher_labels.py" \
    --augmented_data "$AUGMENTED_DATA" \
    --existing_teacher "$DB_TEACHER_EXISTING" \
    --new_teacher "$DB_TEACHER_NEW" \
    --output "$DB_TEACHER_FULL"

# Soft labels
echo "[STEP 2] Soft labels (eps=0.25)"
"$PY" "$SOFT_LABEL_SCRIPT" \
    --input "$DB_TEACHER_FULL" \
    --output "$DB_TEACHER_SOFT" \
    --smooth_eps 0.25

# Build distill dataset
echo "[STEP 3] Build doubao head-distill dataset"
"$PY" "$SHARED_DIR/build_selective_distill_dataset.py" \
    --gt_data "$AUGMENTED_DATA" \
    --teacher_soft "$DB_TEACHER_SOFT" \
    --output "$DB_TRAIN_HEAD" \
    --report "$RUN_ROOT/artifacts/db_head_report.json" \
    --min_entropy 0.20 \
    --smooth_eps 0.25 \
    --min_margin 0.03

echo ""
echo "============================================"
echo "[PHASE 2] Doubao distillation training (5 configs)"
echo "============================================"
"$PY" "$DISTILL_DIR/run_two_stage_training.py" \
    --params "$DISTILL_DIR/grid_params_db_v1.json" \
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
echo "[PHASE 3] DeepSeek v2 distillation (lower alpha, 5 configs)"
echo "============================================"
"$PY" "$DISTILL_DIR/run_two_stage_training.py" \
    --params "$DISTILL_DIR/grid_params_ds_v2.json" \
    --run_root "$RUN_ROOT" \
    --project_root "$PROJECT_ROOT" \
    --base_model "$BASE_MODEL" \
    --train_head "$DS_TRAIN_HEAD" \
    --train_gt "$AUGMENTED_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --teacher_prefix "ds_v2" \
    --py "$PY"

echo ""
echo "============================================"
echo "[DONE] All training complete!"
echo "============================================"

# Collect results
echo ""
echo "=== RESULTS ==="
echo "--- Doubao distillation ---"
grep "测试集准确率" "$RUN_ROOT"/logs/stage2_db_*.log 2>/dev/null || echo "(no doubao results)"
echo ""
echo "--- DeepSeek v2 distillation ---"
grep "测试集准确率" "$RUN_ROOT"/logs/stage2_ds_v2_*.log 2>/dev/null || echo "(no ds_v2 results)"
echo ""
echo "--- DeepSeek v1 (original, alpha=0.35) ---"
grep "测试集准确率" "$RUN_ROOT"/logs/stage2_ds_*.log 2>/dev/null | grep -v ds_v2 || echo "(no ds results)"
