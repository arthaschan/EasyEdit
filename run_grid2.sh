#!/usr/bin/env bash
set -euo pipefail

cd /home/student/arthas/EasyEdit3

echo "[INFO] $(date '+%F %T') start grid-a06" | tee -a run_grid2.log
conda run --no-capture-output -n easyedit python -u train_dental_lora32.py \
  --num_epochs 4 --batch_size 1 --gradient_accumulation_steps 8 \
  --rank 16 --lora_alpha 32 --learning_rate 8e-5 \
  --alpha 0.06 --alpha_warmup_epochs 1 --hard_upsample 1 \
  --temperature 1.5 --seed 2 \
  --val_path ./data/cmexam_dental_choice_val.jsonl \
  --test_path ./data/cmexam_dental_choice_test.jsonl \
  --output_dir ./dental_qwen2.5_7b_choice_lora_distill_from32_grid_a06 \
  > train32_grid_a06.log 2>&1

echo "[INFO] $(date '+%F %T') grid-a06 finished" | tee -a run_grid2.log

echo "[INFO] $(date '+%F %T') start grid-t12" | tee -a run_grid2.log
conda run --no-capture-output -n easyedit python -u train_dental_lora32.py \
  --num_epochs 4 --batch_size 1 --gradient_accumulation_steps 8 \
  --rank 16 --lora_alpha 32 --learning_rate 8e-5 \
  --alpha 0.08 --alpha_warmup_epochs 1 --hard_upsample 1 \
  --temperature 1.2 --seed 2 \
  --val_path ./data/cmexam_dental_choice_val.jsonl \
  --test_path ./data/cmexam_dental_choice_test.jsonl \
  --output_dir ./dental_qwen2.5_7b_choice_lora_distill_from32_grid_t12 \
  > train32_grid_t12.log 2>&1

echo "[INFO] $(date '+%F %T') grid-t12 finished" | tee -a run_grid2.log

python3 - <<'PY' | tee -a run_grid2.log
import os, re, json

logs = {
    'seed2_base': 'train32_seed2.log',
    'grid_a06': 'train32_grid_a06.log',
    'grid_t12': 'train32_grid_t12.log',
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

# 错题净变化（相对 seed2）
def qset(path):
    arr = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    return {x.get('question', '') for x in arr}

base_path = './dental_qwen2.5_7b_choice_lora_distill_from32_seed2/test_wrong.jsonl'
base = qset(base_path) if os.path.exists(base_path) else None
for tag, path in [
    ('grid_a06', './dental_qwen2.5_7b_choice_lora_distill_from32_grid_a06/test_wrong.jsonl'),
    ('grid_t12', './dental_qwen2.5_7b_choice_lora_distill_from32_grid_t12/test_wrong.jsonl'),
]:
    if base is None or not os.path.exists(path):
        print(f'[SUMMARY] {tag} net_fixed_vs_seed2 = N/A')
        continue
    cur = qset(path)
    fixed = len(base - cur)
    reg = len(cur - base)
    print(f'[SUMMARY] {tag} fixed={fixed}, regressed={reg}, net_fixed={fixed-reg}')

valid = [(k, v) for k, v in acc.items() if v is not None]
if valid:
    best = max(valid, key=lambda x: x[1])
    print(f"[SUMMARY] best = {best[0]} ({best[1]:.2f}%)")
PY

echo "[INFO] $(date '+%F %T') all done" | tee -a run_grid2.log
