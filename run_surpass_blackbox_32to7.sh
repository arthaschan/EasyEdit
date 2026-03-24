#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 32B(Teacher) -> 7B(Student) Black-box Distillation Plan
# Goal: Increase the chance that student reaches or exceeds teacher
# ============================================================
#
# Why this recipe is designed for black-box surpassing:
# 1) KL weight is low (alpha=0.04):
#    - Reduce direct imitation of teacher mistakes.
#    - Let CE (ground-truth supervision) dominate.
# 2) KL warmup (alpha_warmup_epochs=2):
#    - First two epochs use pure CE to build student decision boundary.
#    - Then introduce small KL as regularization.
# 3) Lower distillation temperature (temperature=1.2):
#    - Avoid overly soft teacher distribution in black-box-like settings.
# 4) Hard-example upsampling (hard_upsample=3):
#    - Keep teacher-correct/student-wrong cases visible without fully
#      turning the run into teacher imitation.
# 5) Longer schedule (num_epochs=6):
#    - Give CE-dominated training enough steps to correct difficult cases.
# 6) Higher LoRA capacity (rank=32, lora_alpha=64):
#    - Give 7B enough adaptation space for domain transfer.
#
# Fixed protocol for fair comparison:
# - Same test set: ./data/cmexam_dental_choice_test.jsonl
# - Deterministic decoding during evaluation (already in trainer eval):
#   max_new_tokens=4, do_sample=False
# - Multi-seed run to avoid single-run luck.
#
# Outputs:
# - One log per seed: ./train32_surpass_seed<seed>.log
# - One model dir per seed:
#   ./dental_qwen2.5_7b_choice_lora_distill_from32_surpass_v1_seed<seed>
# - Summary file:
#   ./surpass_blackbox_32to7_summary.md

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

TEACHER_MODEL="./Qwen2.5-32B-Instruct"
STUDENT_MODEL="./Qwen2.5-7B-Instruct"

TRAIN_PATH="./data/cmexam_dental_choice_train.jsonl"
VAL_PATH="./data/cmexam_dental_choice_val.jsonl"
TEST_PATH="./data/cmexam_dental_choice_test.jsonl"

# Optional: teacher direct baseline for side-by-side context.
TEACHER_BASELINE_LOG="./teacher32_baseline_autotest.log"

echo "[INFO] Workspace: $ROOT_DIR"
echo "[INFO] Start time: $(date '+%F %T')"

echo "[INFO] Running direct 32B baseline (autoTestQwen32.py)..."
conda run -n easyedit python autoTestQwen32.py > "$TEACHER_BASELINE_LOG" 2>&1 || true
echo "[INFO] Baseline done. Log: $TEACHER_BASELINE_LOG"

# Multi-seed run for stability.
SEEDS=(2 3 4)

# Recipe knobs (mapped to train_dental_lora32.py args).
NUM_EPOCHS=6
BATCH_SIZE=1
GRAD_ACC=8
LR=8e-5
RANK=32
LORA_ALPHA=64
TEMP=1.2
KL_ALPHA=0.04
ALPHA_WARMUP=2
HARD_UPSAMPLE=3
MAX_LENGTH=768

for SEED in "${SEEDS[@]}"; do
	OUT_DIR="./dental_qwen2.5_7b_choice_lora_distill_from32_surpass_v1_seed${SEED}"
	LOG_FILE="./train32_surpass_seed${SEED}.log"

	echo "============================================================"
	echo "[INFO] Running seed=${SEED}"
	echo "[INFO] Output dir: $OUT_DIR"
	echo "[INFO] Log file:   $LOG_FILE"
	echo "============================================================"

	conda run -n easyedit python train_dental_lora32.py \
		--teacher_model "$TEACHER_MODEL" \
		--student_model "$STUDENT_MODEL" \
		--data_path "$TRAIN_PATH" \
		--val_path "$VAL_PATH" \
		--test_path "$TEST_PATH" \
		--output_dir "$OUT_DIR" \
		--num_epochs "$NUM_EPOCHS" \
		--batch_size "$BATCH_SIZE" \
		--gradient_accumulation_steps "$GRAD_ACC" \
		--learning_rate "$LR" \
		--rank "$RANK" \
		--lora_alpha "$LORA_ALPHA" \
		--temperature "$TEMP" \
		--alpha "$KL_ALPHA" \
		--alpha_warmup_epochs "$ALPHA_WARMUP" \
		--hard_upsample "$HARD_UPSAMPLE" \
		--seed "$SEED" \
		--max_length "$MAX_LENGTH" \
		--augment \
		> "$LOG_FILE" 2>&1

	echo "[INFO] Seed ${SEED} completed."
done

echo "[INFO] Building summary..."

{
	echo "# Surpass Blackbox 32->7 Summary"
	echo
	echo "- Generated at: $(date '+%F %T')"
	echo "- Teacher baseline log: $TEACHER_BASELINE_LOG"
	echo
	echo "## Seed Results"
	echo
	echo "| Seed | Train Log | Test Accuracy | Wrong File |"
	echo "|---|---|---:|---|"

	for SEED in "${SEEDS[@]}"; do
		OUT_DIR="./dental_qwen2.5_7b_choice_lora_distill_from32_surpass_v1_seed${SEED}"
		LOG_FILE="./train32_surpass_seed${SEED}.log"
		ACC="$(grep -E '测试集准确率' "$LOG_FILE" | tail -n 1 | sed -E 's/.*测试集准确率: *//' || true)"
		WRONG_FILE="$OUT_DIR/test_wrong.jsonl"
		if [[ -z "$ACC" ]]; then
			ACC="N/A"
		fi
		echo "| ${SEED} | ${LOG_FILE} | ${ACC} | ${WRONG_FILE} |"
	done
} > ./surpass_blackbox_32to7_summary.md

echo "[INFO] Summary ready: ./surpass_blackbox_32to7_summary.md"
echo "[INFO] End time: $(date '+%F %T')"
