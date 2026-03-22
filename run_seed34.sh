#!/usr/bin/env bash
set -euo pipefail

cd /home/student/arthas/EasyEdit3

echo "[INFO] $(date '+%F %T') start seed=3" | tee -a run_seed34.log
conda run --no-capture-output -n easyedit python -u train_dental_lora32.py \
  --num_epochs 4 --batch_size 1 --gradient_accumulation_steps 8 \
  --rank 16 --lora_alpha 32 --learning_rate 8e-5 \
  --alpha 0.08 --alpha_warmup_epochs 1 --hard_upsample 1 \
  --temperature 1.5 --seed 3 \
  --val_path ./data/cmexam_dental_choice_val.jsonl \
  --test_path ./data/cmexam_dental_choice_test.jsonl \
  --output_dir ./dental_qwen2.5_7b_choice_lora_distill_from32_seed3 \
  > train32_seed3.log 2>&1

echo "[INFO] $(date '+%F %T') seed=3 finished" | tee -a run_seed34.log

echo "[INFO] $(date '+%F %T') start seed=4" | tee -a run_seed34.log
conda run --no-capture-output -n easyedit python -u train_dental_lora32.py \
  --num_epochs 4 --batch_size 1 --gradient_accumulation_steps 8 \
  --rank 16 --lora_alpha 32 --learning_rate 8e-5 \
  --alpha 0.08 --alpha_warmup_epochs 1 --hard_upsample 1 \
  --temperature 1.5 --seed 4 \
  --val_path ./data/cmexam_dental_choice_val.jsonl \
  --test_path ./data/cmexam_dental_choice_test.jsonl \
  --output_dir ./dental_qwen2.5_7b_choice_lora_distill_from32_seed4 \
  > train32_seed4.log 2>&1

echo "[INFO] $(date '+%F %T') seed=4 finished" | tee -a run_seed34.log

python3 - <<'PY' | tee -a run_seed34.log
import os, re
logs = {
    'seed2': 'train32_seed2.log',
    'seed3': 'train32_seed3.log',
    'seed4': 'train32_seed4.log',
}
pat = re.compile(r'测试集准确率:\s*([0-9.]+)%')
acc = {}
for k, p in logs.items():
    v = None
    if os.path.exists(p):
        for line in open(p, encoding='utf-8', errors='ignore'):
            m = pat.search(line)
            if m:
                v = float(m.group(1))
    acc[k] = v
print('[SUMMARY] test accuracy =', acc)
valid = [(k, v) for k, v in acc.items() if v is not None]
if valid:
    best = max(valid, key=lambda x: x[1])
    print(f"[SUMMARY] best = {best[0]} ({best[1]:.2f}%)")
PY

echo "[INFO] $(date '+%F %T') all done" | tee -a run_seed34.log
