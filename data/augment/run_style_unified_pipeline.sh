#!/usr/bin/env bash
###############################################################################
# 全自动流水线: 风格改写 → 构建数据集 → 训练(多配置) → 测试 → 生成报告
# 用户睡觉前运行, 醒来看结果
###############################################################################
set -euo pipefail

PROJECT_ROOT="/home/student/arthas/EasyEdit3"
PY="$PROJECT_ROOT/.venv/bin/python"
BASE_MODEL="$PROJECT_ROOT/Qwen2.5-7B-Instruct"
VAL_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_test.jsonl"
MERGED_INPUT="$PROJECT_ROOT/data/augment/merged_train.jsonl"

# Output paths
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$PROJECT_ROOT/external_model_benchmark_20260326/distill_runs/style_unified/runs/${TIMESTAMP}_style_unified"
mkdir -p "$RUN_DIR/outputs" "$RUN_DIR/logs" "$RUN_DIR/artifacts"

REWRITTEN_DATA="$RUN_DIR/artifacts/merged_train_cmexam_style.jsonl"
REPORT="$RUN_DIR/REPORT.txt"

cd "$PROJECT_ROOT"

echo "============================================================" | tee "$REPORT"
echo " Style-Unified Pipeline - $TIMESTAMP" | tee -a "$REPORT"
echo "============================================================" | tee -a "$REPORT"

###############################################################################
# Step 1: Rewrite HuaTuo + AutoGen data to CMExam style via DeepSeek API
###############################################################################
echo "" | tee -a "$REPORT"
echo "[Step 1] Rewriting HuaTuo + AutoGen data to CMExam style..." | tee -a "$REPORT"
echo "Start: $(date)" | tee -a "$REPORT"

$PY data/augment/rewrite_to_cmexam_style.py \
    --input "$MERGED_INPUT" \
    --output "$REWRITTEN_DATA" \
    --request_interval 0.3 \
    --balance_answers \
    2>&1 | tee "$RUN_DIR/logs/rewrite.log"

TOTAL_SAMPLES=$(wc -l < "$REWRITTEN_DATA")
echo "Rewritten data: $TOTAL_SAMPLES samples" | tee -a "$REPORT"
echo "Step 1 done: $(date)" | tee -a "$REPORT"

###############################################################################
# Step 1b: Quick quality check on rewritten data
###############################################################################
echo "" | tee -a "$REPORT"
echo "[Step 1b] Quality check on rewritten data..." | tee -a "$REPORT"

$PY -c "
import json
from collections import Counter

data = [json.loads(l) for l in open('$REWRITTEN_DATA')]
sources = Counter(d.get('Source','?') for d in data)
answers = Counter(d.get('Answer','?')[0] for d in data if d.get('Answer'))

# Question length stats by source
for src in sorted(sources):
    items = [d for d in data if d.get('Source','?')==src]
    lens = [len(d['Question']) for d in items]
    avg = sum(lens)/len(lens)
    print(f'  {src}: {len(items)} samples, avg question len={avg:.1f}')

print(f'Sources: {dict(sources)}')
print(f'Answer distribution: {dict(sorted(answers.items()))}')

# Check for remaining style issues
dialog_count = sum(1 for d in data if '根据对话' in d.get('Question',''))
colon_end = sum(1 for d in data if d.get('Question','').rstrip().endswith(':') or d.get('Question','').rstrip().endswith('：'))
print(f'Remaining 根据对话: {dialog_count}/{len(data)}')
print(f'Remaining colon-ending: {colon_end}/{len(data)}')
" 2>&1 | tee -a "$REPORT"

###############################################################################
# Step 2: Train multiple SFT configs on style-unified data
###############################################################################
echo "" | tee -a "$REPORT"
echo "[Step 2] Training SFT models on style-unified data..." | tee -a "$REPORT"

# Define training configs - best-performing hyperparameters from previous experiments
declare -A CONFIGS
# config_name -> "seed lr epochs rank lora_alpha"
CONFIGS[sty_r16_s11_e2]="11 0.00012 2 16 32"
CONFIGS[sty_r16_s42_e2]="42 0.00012 2 16 32"
CONFIGS[sty_r16_s55_e2]="55 0.00012 2 16 32"
CONFIGS[sty_r16_s42_e3]="42 0.00014 3 16 32"
CONFIGS[sty_r16_s11_e3]="11 0.00014 3 16 32"

BEST_ACC=0
BEST_CONFIG=""

for config_name in sty_r16_s11_e2 sty_r16_s42_e2 sty_r16_s55_e2 sty_r16_s42_e3 sty_r16_s11_e3; do
    read -r SEED LR EPOCHS RANK LORA_ALPHA <<< "${CONFIGS[$config_name]}"
    
    OUT_DIR="$RUN_DIR/outputs/$config_name"
    LOG_FILE="$RUN_DIR/logs/train_${config_name}.log"
    
    echo "" | tee -a "$REPORT"
    echo "--- Training config: $config_name (seed=$SEED, lr=$LR, epochs=$EPOCHS, rank=$RANK) ---" | tee -a "$REPORT"
    echo "Start: $(date)" | tee -a "$REPORT"
    
    $PY train_dental_lora7_deepseek.py \
        --model_name "$BASE_MODEL" \
        --data_path "$REWRITTEN_DATA" \
        --val_path "$VAL_DATA" \
        --test_path "$TEST_DATA" \
        --output_dir "$OUT_DIR" \
        --num_epochs "$EPOCHS" \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate "$LR" \
        --rank "$RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --alpha 0.0 \
        --seed "$SEED" \
        --deterministic \
        --use_teacher_dist \
        2>&1 | tee "$LOG_FILE"
    
    # Extract test accuracy from log (format: 测试集准确率: XX.XX%)
    ACC=$(grep -oP '测试集准确率:\s*\K[\d.]+' "$LOG_FILE" | tail -1 || echo "0")
    if [ -z "$ACC" ] || [ "$ACC" = "0" ]; then
        ACC=$(grep -oP '\d+\.\d+(?=%)' "$LOG_FILE" | tail -1 || echo "N/A")
    fi
    
    echo "Config $config_name: Test Accuracy = ${ACC}%" | tee -a "$REPORT"
    echo "Done: $(date)" | tee -a "$REPORT"
    
    # Track best
    if (( $(echo "$ACC > $BEST_ACC" | bc -l 2>/dev/null || echo 0) )); then
        BEST_ACC=$ACC
        BEST_CONFIG=$config_name
    fi
done

###############################################################################
# Step 3: Also train pure SFT on ORIGINAL merged data for fair comparison
# (same configs as above, just different data)
###############################################################################
echo "" | tee -a "$REPORT"
echo "[Step 3] Training baseline (original data) for comparison..." | tee -a "$REPORT"

ORIG_BEST_ACC=0
ORIG_BEST_CONFIG=""

for config_name in orig_r16_s11_e2 orig_r16_s42_e2; do
    if [ "$config_name" = "orig_r16_s11_e2" ]; then
        SEED=11; LR=0.00012; EPOCHS=2; RANK=16; LORA_ALPHA=32
    else
        SEED=42; LR=0.00012; EPOCHS=2; RANK=16; LORA_ALPHA=32
    fi
    
    OUT_DIR="$RUN_DIR/outputs/$config_name"
    LOG_FILE="$RUN_DIR/logs/train_${config_name}.log"
    
    echo "" | tee -a "$REPORT"
    echo "--- Baseline config: $config_name (seed=$SEED, original data) ---" | tee -a "$REPORT"
    echo "Start: $(date)" | tee -a "$REPORT"
    
    $PY train_dental_lora7_deepseek.py \
        --model_name "$BASE_MODEL" \
        --data_path "$MERGED_INPUT" \
        --val_path "$VAL_DATA" \
        --test_path "$TEST_DATA" \
        --output_dir "$OUT_DIR" \
        --num_epochs "$EPOCHS" \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate "$LR" \
        --rank "$RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --alpha 0.0 \
        --seed "$SEED" \
        --deterministic \
        --use_teacher_dist \
        2>&1 | tee "$LOG_FILE"
    
    ACC=$(grep -oP '测试集准确率:\s*\K[\d.]+' "$LOG_FILE" | tail -1 || echo "0")
    if [ -z "$ACC" ] || [ "$ACC" = "0" ]; then
        ACC=$(grep -oP '\d+\.\d+(?=%)' "$LOG_FILE" | tail -1 || echo "N/A")
    fi
    
    echo "Baseline $config_name: Test Accuracy = ${ACC}%" | tee -a "$REPORT"
    echo "Done: $(date)" | tee -a "$REPORT"
    
    if (( $(echo "$ACC > $ORIG_BEST_ACC" | bc -l 2>/dev/null || echo 0) )); then
        ORIG_BEST_ACC=$ACC
        ORIG_BEST_CONFIG=$config_name
    fi
done

###############################################################################
# Step 4: Also try CMExam-ONLY (no augmented data at all) for comparison
###############################################################################
echo "" | tee -a "$REPORT"
echo "[Step 4] Training CMExam-only baseline..." | tee -a "$REPORT"

# Extract only cmexam_original samples
CMEXAM_ONLY="$RUN_DIR/artifacts/cmexam_only_train.jsonl"
$PY -c "
import json
data = [json.loads(l) for l in open('$MERGED_INPUT')]
cmexam = [d for d in data if d.get('Source') == 'cmexam_original']
with open('$CMEXAM_ONLY', 'w') as f:
    for d in cmexam:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print(f'CMExam-only: {len(cmexam)} samples')
"

for config_name in cmexam_r16_s11_e2 cmexam_r16_s42_e3; do
    if [ "$config_name" = "cmexam_r16_s11_e2" ]; then
        SEED=11; LR=0.00012; EPOCHS=2; RANK=16; LORA_ALPHA=32
    else
        SEED=42; LR=0.00014; EPOCHS=3; RANK=16; LORA_ALPHA=32
    fi
    
    OUT_DIR="$RUN_DIR/outputs/$config_name"
    LOG_FILE="$RUN_DIR/logs/train_${config_name}.log"
    
    echo "" | tee -a "$REPORT"
    echo "--- CMExam-only: $config_name (seed=$SEED, cmexam data only) ---" | tee -a "$REPORT"
    echo "Start: $(date)" | tee -a "$REPORT"
    
    $PY train_dental_lora7_deepseek.py \
        --model_name "$BASE_MODEL" \
        --data_path "$CMEXAM_ONLY" \
        --val_path "$VAL_DATA" \
        --test_path "$TEST_DATA" \
        --output_dir "$OUT_DIR" \
        --num_epochs "$EPOCHS" \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate "$LR" \
        --rank "$RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --alpha 0.0 \
        --seed "$SEED" \
        --deterministic \
        --use_teacher_dist \
        2>&1 | tee "$LOG_FILE"
    
    ACC=$(grep -oP '测试集准确率:\s*\K[\d.]+' "$LOG_FILE" | tail -1 || echo "0")
    if [ -z "$ACC" ] || [ "$ACC" = "0" ]; then
        ACC=$(grep -oP '\d+\.\d+(?=%)' "$LOG_FILE" | tail -1 || echo "N/A")
    fi
    
    echo "CMExam-only $config_name: Test Accuracy = ${ACC}%" | tee -a "$REPORT"
    echo "Done: $(date)" | tee -a "$REPORT"
done

###############################################################################
# Step 5: Summary Report
###############################################################################
echo "" | tee -a "$REPORT"
echo "============================================================" | tee -a "$REPORT"
echo " FINAL SUMMARY" | tee -a "$REPORT"
echo "============================================================" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"

echo "Historical Best (pure SFT, original augmented data): 79.52%" | tee -a "$REPORT"
echo "Historical Best (5-model ensemble): 80.72%" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"

echo "--- Style-Unified SFT Results ---" | tee -a "$REPORT"
for config_name in sty_r16_s11_e2 sty_r16_s42_e2 sty_r16_s55_e2 sty_r16_s42_e3 sty_r16_s11_e3; do
    LOG_FILE="$RUN_DIR/logs/train_${config_name}.log"
    ACC=$(grep -oP '测试集准确率:\s*\K[\d.]+' "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    if [ -z "$ACC" ] || [ "$ACC" = "N/A" ]; then
        ACC=$(grep -oP '\d+\.\d+(?=%)' "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    fi
    echo "  $config_name: ${ACC}%" | tee -a "$REPORT"
done

echo "" | tee -a "$REPORT"
echo "--- Original Data Baselines ---" | tee -a "$REPORT"
for config_name in orig_r16_s11_e2 orig_r16_s42_e2; do
    LOG_FILE="$RUN_DIR/logs/train_${config_name}.log"
    ACC=$(grep -oP '测试集准确率:\s*\K[\d.]+' "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    if [ -z "$ACC" ] || [ "$ACC" = "N/A" ]; then
        ACC=$(grep -oP '\d+\.\d+(?=%)' "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    fi
    echo "  $config_name: ${ACC}%" | tee -a "$REPORT"
done

echo "" | tee -a "$REPORT"
echo "--- CMExam-Only Baselines ---" | tee -a "$REPORT"
for config_name in cmexam_r16_s11_e2 cmexam_r16_s42_e3; do
    LOG_FILE="$RUN_DIR/logs/train_${config_name}.log"
    ACC=$(grep -oP '测试集准确率:\s*\K[\d.]+' "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    if [ -z "$ACC" ] || [ "$ACC" = "N/A" ]; then
        ACC=$(grep -oP '\d+\.\d+(?=%)' "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    fi
    echo "  $config_name: ${ACC}%" | tee -a "$REPORT"
done

echo "" | tee -a "$REPORT"
echo "Data statistics:" | tee -a "$REPORT"
echo "  Rewritten samples: $TOTAL_SAMPLES" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"
echo "Pipeline completed at: $(date)" | tee -a "$REPORT"
echo "Report location: $REPORT" | tee -a "$REPORT"

echo ""
echo "============== ALL DONE! =============="
echo "Report: $REPORT"
