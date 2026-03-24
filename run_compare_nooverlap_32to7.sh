#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 32B->7B 黑盒蒸馏（无泄漏划分）对比实验
# 方案A：历史强基线参数（seed2）
# 方案B：低KL反超导向参数（seed2）
# 输出：compare_nooverlap_32to7_summary.md
# ============================================================

cd /home/student/arthas/EasyEdit3

TRAIN=data/cmexam_dental_choice_train_nooverlap.jsonl
VAL=data/cmexam_dental_choice_val_nooverlap.jsonl
TEST=data/cmexam_dental_choice_test.jsonl

LOG_A=train32_nooverlap_hist_seed2.log
LOG_B=train32_nooverlap_lowkl_seed2.log
OUT_A=./dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_hist_seed2
OUT_B=./dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_lowkl_seed2

# 32B老师基线（固定口径）
BASELINE_LOG=teacher32_baseline_autotest_nooverlap.log
conda run -n easyedit python autoTestQwen32.py > "$BASELINE_LOG" 2>&1

# -----------------------------
# 方案A：历史强基线复测
# 来源：旧实验最优一档（lr=8e-5, temp=1.5, alpha=0.08, hard_upsample=1）
# -----------------------------
conda run -n easyedit python train_dental_lora32.py \
  --teacher_model ./Qwen2.5-32B-Instruct \
  --student_model ./Qwen2.5-7B-Instruct \
  --data_path "$TRAIN" \
  --val_path "$VAL" \
  --test_path "$TEST" \
  --output_dir "$OUT_A" \
  --num_epochs 4 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 8e-5 \
  --rank 16 \
  --lora_alpha 32 \
  --temperature 1.5 \
  --alpha 0.08 \
  --alpha_warmup_epochs 1 \
  --hard_upsample 1 \
  --seed 2 \
  --max_length 768 > "$LOG_A" 2>&1

# -----------------------------
# 方案B：低KL反超导向
# 设计：减小 teacher 约束，增强CE纠偏能力
# -----------------------------
conda run -n easyedit python train_dental_lora32.py \
  --teacher_model ./Qwen2.5-32B-Instruct \
  --student_model ./Qwen2.5-7B-Instruct \
  --data_path "$TRAIN" \
  --val_path "$VAL" \
  --test_path "$TEST" \
  --output_dir "$OUT_B" \
  --num_epochs 6 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 8e-5 \
  --rank 32 \
  --lora_alpha 64 \
  --temperature 1.2 \
  --alpha 0.03 \
  --alpha_warmup_epochs 2 \
  --hard_upsample 2 \
  --seed 2 \
  --max_length 768 \
  --augment > "$LOG_B" 2>&1

python - <<'PY'
import re
from pathlib import Path

def pick(p, pat):
    txt=Path(p).read_text(encoding='utf-8',errors='ignore')
    m=list(re.finditer(pat,txt))
    return m[-1].group(1) if m else 'N/A'

base='teacher32_baseline_autotest_nooverlap.log'
log_a='train32_nooverlap_hist_seed2.log'
log_b='train32_nooverlap_lowkl_seed2.log'

teacher=pick(base,r'正确率：([0-9.]+%)')
a=pick(log_a,r'测试集准确率: ([0-9.]+%)')
b=pick(log_b,r'测试集准确率: ([0-9.]+%)')

summary=Path('compare_nooverlap_32to7_summary.md')
summary.write_text(f'''# 32B->7B No-overlap Comparison\n\n- Teacher baseline (direct 32B): **{teacher}**\n\n| Config | Test Accuracy | Log | Wrong File |\n|---|---:|---|---|\n| A. Historical baseline params | {a} | `{log_a}` | `./dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_hist_seed2/test_wrong.jsonl` |\n| B. Low-KL surpass-oriented params | {b} | `{log_b}` | `./dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_lowkl_seed2/test_wrong.jsonl` |\n\n''',encoding='utf-8')
print('WROTE compare_nooverlap_32to7_summary.md')
PY

