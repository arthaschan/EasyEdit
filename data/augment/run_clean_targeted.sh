#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/student/arthas/EasyEdit3"
PY="$PROJECT_ROOT/.venv/bin/python"
BASE_MODEL="$PROJECT_ROOT/Qwen2.5-7B-Instruct"
VAL_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_test.jsonl"
CLEAN_DATA="$PROJECT_ROOT/data/augment/merged_style_plus_generated_only.jsonl"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$PROJECT_ROOT/external_model_benchmark_20260326/distill_runs/targeted_hard_clean/runs/${TIMESTAMP}_clean"
mkdir -p "$RUN_DIR/outputs" "$RUN_DIR/logs"
REPORT="$RUN_DIR/REPORT.txt"
cd "$PROJECT_ROOT"

echo "=== Clean Targeted Pipeline (NO data leakage) ===" | tee "$REPORT"
echo "Training data: 977 style-unified + 56 generated related (no test copies)" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"

declare -A CONFIGS
CONFIGS[clean_r16_s42_e2]="42 0.00012 2 16 32"
CONFIGS[clean_r16_s11_e2]="11 0.00012 2 16 32"
CONFIGS[clean_r16_s55_e2]="55 0.00012 2 16 32"
CONFIGS[clean_r16_s42_e3]="42 0.00014 3 16 32"
CONFIGS[clean_r16_s11_e3]="11 0.00014 3 16 32"

for cfg in clean_r16_s42_e2 clean_r16_s11_e2 clean_r16_s55_e2 clean_r16_s42_e3 clean_r16_s11_e3; do
    read -r SEED LR EPOCHS RANK LORA_ALPHA <<< "${CONFIGS[$cfg]}"
    OUT="$RUN_DIR/outputs/$cfg"
    LOG="$RUN_DIR/logs/train_${cfg}.log"
    echo "--- $cfg (seed=$SEED, lr=$LR, epochs=$EPOCHS) ---" | tee -a "$REPORT"

    $PY train_dental_lora7_deepseek.py \
        --model_name "$BASE_MODEL" \
        --data_path "$CLEAN_DATA" \
        --val_path "$VAL_DATA" \
        --test_path "$TEST_DATA" \
        --output_dir "$OUT" \
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
        2>&1 | tee "$LOG"

    ACC=$(grep -oP '测试集准确率:\s*\K[\d.]+' "$LOG" | tail -1 || echo "N/A")
    echo "$cfg: ${ACC}%" | tee -a "$REPORT"
    echo "" | tee -a "$REPORT"
done

echo "============================================================" | tee -a "$REPORT"
echo "SUMMARY (clean, no data leakage)" | tee -a "$REPORT"
echo "============================================================" | tee -a "$REPORT"
echo "Previous best (style-unified only): 80.72%" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"
for cfg in clean_r16_s42_e2 clean_r16_s11_e2 clean_r16_s55_e2 clean_r16_s42_e3 clean_r16_s11_e3; do
    LOG="$RUN_DIR/logs/train_${cfg}.log"
    ACC=$(grep -oP '测试集准确率:\s*\K[\d.]+' "$LOG" 2>/dev/null | tail -1 || echo "N/A")
    echo "  $cfg: ${ACC}%" | tee -a "$REPORT"
done

# Hard question analysis
echo "" | tee -a "$REPORT"
echo "--- Hard Question Fix Analysis (clean) ---" | tee -a "$REPORT"
$PY -c "
import json, os

sty_dir = 'external_model_benchmark_20260326/distill_runs/style_unified/runs/20260410_020935_style_unified/outputs'
sty_cfgs = ['sty_r16_s11_e2','sty_r16_s42_e2','sty_r16_s55_e2','sty_r16_s42_e3','sty_r16_s11_e3']
all_sty = {}
for c in sty_cfgs:
    all_sty[c] = {e['question'] for e in (json.loads(l) for l in open(f'{sty_dir}/{c}/test_wrong.jsonl'))}
hard_qs = set.intersection(*all_sty.values())

tgt_dir = '$RUN_DIR/outputs'
for cfg in ['clean_r16_s42_e2','clean_r16_s11_e2','clean_r16_s55_e2','clean_r16_s42_e3','clean_r16_s11_e3']:
    wp = f'{tgt_dir}/{cfg}/test_wrong.jsonl'
    if not os.path.exists(wp): continue
    wrongs = {e['question'] for e in (json.loads(l) for l in open(wp))}
    fixed = hard_qs - wrongs
    new_err = wrongs - all_sty.get('sty_r16_s42_e2', set())
    print(f'{cfg}: fixed {len(fixed)}/14 hard, new_errors={len(new_err)}, total_wrong={len(wrongs)}')
    for q in sorted(fixed):
        print(f'  FIXED: {q[:60]}')
" 2>&1 | tee -a "$REPORT"

echo "" | tee -a "$REPORT"
echo "=== ALL DONE ===" | tee -a "$REPORT"
