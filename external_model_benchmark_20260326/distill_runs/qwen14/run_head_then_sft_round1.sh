#!/bin/bash
set -euo pipefail

# ================================================================
# Qwen2.5-14B-Instruct → 7B 蒸馏全流程
#   Step 0: 14B 教师推理测试集 (获取教师准确率)
#   Step 1: 14B 教师推理训练集 (生成教师标签)
#   Step 2: Label smoothing → 软标签
#   Step 3: 构建选择性蒸馏数据集
#   Step 4: 两阶段 Choice-Head→SFT 训练 (3 seeds)
#   Step 5: 汇总结果
# ================================================================

PROJECT_ROOT="/home/student/arthas/EasyEdit3"
DISTILL_DIR="$PROJECT_ROOT/external_model_benchmark_20260326/distill_runs/qwen14"
SHARED_DIR="$PROJECT_ROOT/external_model_benchmark_20260326/distill_runs/shared"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

TRAIN_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_train_nooverlap.jsonl"
VAL_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_test.jsonl"
MODEL_14B="$PROJECT_ROOT/Qwen2.5-14B-Instruct"

TEACHER_TRAIN="$DISTILL_DIR/teacher_train.jsonl"
TEACHER_SOFT="$DISTILL_DIR/teacher_train_soft.jsonl"
DISTILL_DATA="$DISTILL_DIR/train_head_distill.jsonl"

SMOOTH_EPS=0.25
MIN_ENTROPY=0.20
MIN_MARGIN=0.03

# 时间戳
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$DISTILL_DIR/runs/${TS}_qwen14_headsft_r1"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/outputs"

echo "============================================================"
echo "Qwen2.5-14B→7B 两阶段蒸馏实验 (Choice-Head → SFT)"
echo "Run: $RUN_DIR"
echo "============================================================"

# ------ Step 0: 教师推理测试集 ------
echo ""
echo ">>> Step 0: 14B 教师直接推理测试集 (83题)"
cd "$PROJECT_ROOT"
$PYTHON "$DISTILL_DIR/generate_local_teacher_labels.py" \
    --model_path "$MODEL_14B" \
    --dataset "$TEST_DATA" \
    --output "$DISTILL_DIR/teacher_test.jsonl" \
    2>&1 | tee "$DISTILL_DIR/logs/teacher_test_${TS}.log"

# 从日志提取 teacher accuracy on test set
TEST_ACC=$(grep "Teacher Accuracy on train set" "$DISTILL_DIR/logs/teacher_test_${TS}.log" | tail -1 | grep -oP '[\d.]+%')
echo "14B Teacher Test Accuracy: $TEST_ACC"

# ------ Step 1: 教师推理训练集 ------
echo ""
echo ">>> Step 1: 14B 教师推理训练集 (672题)"
$PYTHON "$DISTILL_DIR/generate_local_teacher_labels.py" \
    --model_path "$MODEL_14B" \
    --dataset "$TRAIN_DATA" \
    --output "$TEACHER_TRAIN" \
    2>&1 | tee "$DISTILL_DIR/logs/teacher_train_${TS}.log"

# ------ Step 2: Label Smoothing → 软标签 ------
echo ""
echo ">>> Step 2: Label Smoothing (eps=$SMOOTH_EPS)"
$PYTHON "$SHARED_DIR/../deepseek_opus/prepare_soft_labels.py" \
    --input "$TEACHER_TRAIN" \
    --output "$TEACHER_SOFT" \
    --smooth_eps $SMOOTH_EPS

# ------ Step 3: 构建选择性蒸馏数据集 ------
echo ""
echo ">>> Step 3: 构建选择性蒸馏数据集"
$PYTHON "$SHARED_DIR/build_selective_distill_dataset.py" \
    --input "$TEACHER_SOFT" \
    --output "$DISTILL_DATA" \
    --min_entropy $MIN_ENTROPY \
    --min_margin $MIN_MARGIN

# ------ Step 4: 两阶段训练 (3 seeds) ------
echo ""
echo ">>> Step 4: 两阶段训练 (seeds: 42, 11, 55)"
GRID_PARAMS="$DISTILL_DIR/grid_params_head_then_sft_round1.json"

cd "$PROJECT_ROOT"
$PYTHON "$SHARED_DIR/../augmented_distill/run_two_stage_training.py" \
    --params "$GRID_PARAMS" \
    --run_root "$RUN_DIR" \
    --project_root "$PROJECT_ROOT" \
    --base_model "$PROJECT_ROOT/Qwen2.5-7B-Instruct" \
    --train_head "$DISTILL_DATA" \
    --train_gt "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --teacher_prefix "q14" \
    --py "$PYTHON" \
    2>&1 | tee "$RUN_DIR/logs/two_stage_${TS}.log"

# ------ Step 5: 汇总 ------
echo ""
echo ">>> Step 5: 汇总结果"
echo "{"  > "$RUN_DIR/qwen14_headsft_results_latest.json"
echo '  "teacher": "Qwen2.5-14B-Instruct",' >> "$RUN_DIR/qwen14_headsft_results_latest.json"
echo "  \"teacher_test_accuracy\": \"$TEST_ACC\"," >> "$RUN_DIR/qwen14_headsft_results_latest.json"
echo "  \"run_id\": \"${TS}_qwen14_headsft_r1\"," >> "$RUN_DIR/qwen14_headsft_results_latest.json"
echo '  "approach": "Choice-Head→SFT (two-stage)",' >> "$RUN_DIR/qwen14_headsft_results_latest.json"
echo '  "note": "See logs and outputs for per-seed results"' >> "$RUN_DIR/qwen14_headsft_results_latest.json"
echo "}" >> "$RUN_DIR/qwen14_headsft_results_latest.json"

echo ""
echo "============================================================"
echo "实验完成!"
echo "  结果目录: $RUN_DIR"
echo "  教师测试准确率: $TEST_ACC"
echo "============================================================"
