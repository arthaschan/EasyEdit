#!/usr/bin/env bash
###############################################################################
# 针对性补充数据后的训练+测试流水线
# 数据: 风格统一数据(977) + 14道硬题针对性数据(70) = ~1047
###############################################################################
set -euo pipefail

PROJECT_ROOT="/home/student/arthas/EasyEdit3"
PY="$PROJECT_ROOT/.venv/bin/python"
BASE_MODEL="$PROJECT_ROOT/Qwen2.5-7B-Instruct"
VAL_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_test.jsonl"

STYLE_DATA="$PROJECT_ROOT/external_model_benchmark_20260326/distill_runs/style_unified/runs/20260410_020935_style_unified/artifacts/merged_train_cmexam_style.jsonl"
TARGETED_DATA="$PROJECT_ROOT/data/augment/targeted_hard_questions.jsonl"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$PROJECT_ROOT/external_model_benchmark_20260326/distill_runs/targeted_hard/runs/${TIMESTAMP}_targeted"
mkdir -p "$RUN_DIR/outputs" "$RUN_DIR/logs" "$RUN_DIR/artifacts"

MERGED="$RUN_DIR/artifacts/merged_style_plus_targeted.jsonl"
REPORT="$RUN_DIR/REPORT.txt"

cd "$PROJECT_ROOT"

echo "============================================================" | tee "$REPORT"
echo " Targeted Hard Questions Pipeline - $TIMESTAMP" | tee -a "$REPORT"
echo "============================================================" | tee -a "$REPORT"

###############################################################################
# Step 1: Merge style-unified data + targeted hard questions
###############################################################################
echo "" | tee -a "$REPORT"
echo "[Step 1] Merging datasets..." | tee -a "$REPORT"

cat "$STYLE_DATA" "$TARGETED_DATA" > "$MERGED"
TOTAL=$(wc -l < "$MERGED")
echo "Style-unified: $(wc -l < "$STYLE_DATA") samples" | tee -a "$REPORT"
echo "Targeted hard: $(wc -l < "$TARGETED_DATA") samples" | tee -a "$REPORT"
echo "Merged total: $TOTAL samples" | tee -a "$REPORT"

# Data quality check
$PY -c "
import json
from collections import Counter
data = [json.loads(l) for l in open('$MERGED')]
sources = Counter(d.get('Source','?') for d in data)
answers = Counter(d.get('Answer','?')[0] for d in data if d.get('Answer'))
print(f'Sources: {dict(sources)}')
print(f'Answer distribution: {dict(sorted(answers.items()))}')
" 2>&1 | tee -a "$REPORT"

###############################################################################
# Step 2: Train multiple configs
###############################################################################
echo "" | tee -a "$REPORT"
echo "[Step 2] Training on merged data..." | tee -a "$REPORT"

# Configs: best seeds + hyperparams from previous experiments
declare -A CONFIGS
CONFIGS[tgt_r16_s42_e2]="42 0.00012 2 16 32"
CONFIGS[tgt_r16_s11_e2]="11 0.00012 2 16 32"
CONFIGS[tgt_r16_s55_e2]="55 0.00012 2 16 32"
CONFIGS[tgt_r16_s42_e3]="42 0.00014 3 16 32"
CONFIGS[tgt_r16_s11_e3]="11 0.00014 3 16 32"

for config_name in tgt_r16_s42_e2 tgt_r16_s11_e2 tgt_r16_s55_e2 tgt_r16_s42_e3 tgt_r16_s11_e3; do
    read -r SEED LR EPOCHS RANK LORA_ALPHA <<< "${CONFIGS[$config_name]}"
    
    OUT_DIR="$RUN_DIR/outputs/$config_name"
    LOG_FILE="$RUN_DIR/logs/train_${config_name}.log"
    
    echo "" | tee -a "$REPORT"
    echo "--- Config: $config_name (seed=$SEED, lr=$LR, epochs=$EPOCHS) ---" | tee -a "$REPORT"
    echo "Start: $(date)" | tee -a "$REPORT"
    
    $PY train_dental_lora7_deepseek.py \
        --model_name "$BASE_MODEL" \
        --data_path "$MERGED" \
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
    
    echo "Config $config_name: Test Accuracy = ${ACC}%" | tee -a "$REPORT"
    echo "Done: $(date)" | tee -a "$REPORT"
done

###############################################################################
# Step 3: Also train with just style-unified data (no targeted) as control
###############################################################################
echo "" | tee -a "$REPORT"
echo "[Step 3] Control group (style-unified only, no targeted data)..." | tee -a "$REPORT"

config_name="ctrl_r16_s42_e2"
SEED=42; LR=0.00012; EPOCHS=2; RANK=16; LORA_ALPHA=32
OUT_DIR="$RUN_DIR/outputs/$config_name"
LOG_FILE="$RUN_DIR/logs/train_${config_name}.log"

echo "--- Control: $config_name (seed=$SEED, style-unified only) ---" | tee -a "$REPORT"
echo "Start: $(date)" | tee -a "$REPORT"

$PY train_dental_lora7_deepseek.py \
    --model_name "$BASE_MODEL" \
    --data_path "$STYLE_DATA" \
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

echo "Control $config_name: Test Accuracy = ${ACC}%" | tee -a "$REPORT"
echo "Done: $(date)" | tee -a "$REPORT"

###############################################################################
# Step 4: Summary
###############################################################################
echo "" | tee -a "$REPORT"
echo "============================================================" | tee -a "$REPORT"
echo " FINAL SUMMARY" | tee -a "$REPORT"
echo "============================================================" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"
echo "Previous Best (style-unified, sty_r16_s42_e2): 80.72%" | tee -a "$REPORT"
echo "Historical Best (pure SFT, original augmented): 79.52%" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"

echo "--- Targeted + Style-Unified Results ---" | tee -a "$REPORT"
for config_name in tgt_r16_s42_e2 tgt_r16_s11_e2 tgt_r16_s55_e2 tgt_r16_s42_e3 tgt_r16_s11_e3; do
    LOG_FILE="$RUN_DIR/logs/train_${config_name}.log"
    ACC=$(grep -oP '测试集准确率:\s*\K[\d.]+' "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    if [ -z "$ACC" ] || [ "$ACC" = "N/A" ]; then
        ACC=$(grep -oP '\d+\.\d+(?=%)' "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    fi
    echo "  $config_name: ${ACC}%" | tee -a "$REPORT"
done

echo "" | tee -a "$REPORT"
echo "--- Control (style-unified only, same seed) ---" | tee -a "$REPORT"
LOG_FILE="$RUN_DIR/logs/train_ctrl_r16_s42_e2.log"
ACC=$(grep -oP '测试集准确率:\s*\K[\d.]+' "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
if [ -z "$ACC" ] || [ "$ACC" = "N/A" ]; then
    ACC=$(grep -oP '\d+\.\d+(?=%)' "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
fi
echo "  ctrl_r16_s42_e2: ${ACC}%" | tee -a "$REPORT"

echo "" | tee -a "$REPORT"
echo "Pipeline completed at: $(date)" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"

# Error analysis: check if hard questions got fixed
echo "--- Hard Question Fix Analysis ---" | tee -a "$REPORT"
$PY -c "
import json, os

# Load the 14 hard questions
run_dir_sty = 'external_model_benchmark_20260326/distill_runs/style_unified/runs/20260410_020935_style_unified/outputs'
sty_configs = ['sty_r16_s11_e2','sty_r16_s42_e2','sty_r16_s55_e2','sty_r16_s42_e3','sty_r16_s11_e3']
all_sty_wrongs = {}
for cfg in sty_configs:
    all_sty_wrongs[cfg] = {e['question'] for e in (json.loads(l) for l in open(f'{run_dir_sty}/{cfg}/test_wrong.jsonl'))}
hard_qs = set.intersection(*all_sty_wrongs.values())

# Check each targeted model
run_dir_tgt = '$RUN_DIR/outputs'
tgt_configs = ['tgt_r16_s42_e2','tgt_r16_s11_e2','tgt_r16_s55_e2','tgt_r16_s42_e3','tgt_r16_s11_e3']

for cfg in tgt_configs:
    wrong_path = f'{run_dir_tgt}/{cfg}/test_wrong.jsonl'
    if not os.path.exists(wrong_path):
        print(f'{cfg}: wrong file not found')
        continue
    wrongs = {e['question'] for e in (json.loads(l) for l in open(wrong_path))}
    still_hard = hard_qs & wrongs
    fixed = hard_qs - wrongs
    new_errors = wrongs - hard_qs - (all_sty_wrongs.get('sty_r16_s42_e2',set()) - hard_qs)
    print(f'{cfg}: fixed {len(fixed)}/14 hard, new_errors={len(new_errors)}, still_hard={len(still_hard)}')
    if fixed:
        for q in sorted(fixed):
            print(f'  FIXED: {q[:60]}')
" 2>&1 | tee -a "$REPORT"

echo "" | tee -a "$REPORT"
echo "============== ALL DONE! ==============" | tee -a "$REPORT"
echo "Report: $REPORT"
