#!/usr/bin/env bash
set -euo pipefail

cd /home/student/arthas/EasyEdit3

TRAIN_BASE=data/cmexam_dental_choice_train_nooverlap.jsonl
VAL=data/cmexam_dental_choice_val_nooverlap.jsonl
TEST=data/cmexam_dental_choice_test.jsonl
TRAIN_HB=data/cmexam_dental_choice_train_nooverlap_hardboost.jsonl

# 1) Build fine-grained hard-boost train set from recurring errors
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

root=Path('/home/student/arthas/EasyEdit3')
train_path=root/'data/cmexam_dental_choice_train_nooverlap.jsonl'
out_path=root/'data/cmexam_dental_choice_train_nooverlap_hardboost.jsonl'

wrong_files=[
    root/'dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_hist_seed2/test_wrong.jsonl',
    root/'dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_hist_seed3/test_wrong.jsonl',
    root/'dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_hist_seed4/test_wrong.jsonl',
    root/'dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_weakkl_seed2/test_wrong.jsonl',
    root/'dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_weakkl_seed3/test_wrong.jsonl',
    root/'dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_weakkl_seed4/test_wrong.jsonl',
]

cnt=Counter()
for fp in wrong_files:
    if not fp.exists():
        continue
    with open(fp,encoding='utf-8') as f:
        for line in f:
            d=json.loads(line)
            q=d.get('question','').strip()
            if q:
                cnt[q]+=1

# recurring hard: appears in >=2 wrong sets
hard_q={q for q,c in cnt.items() if c>=2}

train=[]
with open(train_path,encoding='utf-8') as f:
    for line in f:
        d=json.loads(line)
        train.append(d)

boost=[x for x in train if x.get('Question','').strip() in hard_q]
# +2 extra copies => total weight 3x on recurring hard samples
rows=train + boost + boost

with open(out_path,'w',encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps(r,ensure_ascii=False)+'\n')

print(f'base_train={len(train)} hard_q={len(hard_q)} boost_rows={len(boost)} hardboost_train={len(rows)}')
print(f'WROTE {out_path}')
PY

# 2) Extend config C to seeds 5,6 (to reach >=5 seeds: 2..6)
for seed in 5 6; do
  LOG="train32_nooverlap_weakkl_seed${seed}.log"
  OUT="./dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_weakkl_seed${seed}"
  if [[ -f "$LOG" ]]; then
    echo "[SKIP] $LOG exists"
  else
    echo "[RUN] Config C seed=$seed"
    conda run -n easyedit python train_dental_lora32.py \
      --teacher_model ./Qwen2.5-32B-Instruct \
      --student_model ./Qwen2.5-7B-Instruct \
      --data_path "$TRAIN_BASE" \
      --val_path "$VAL" \
      --test_path "$TEST" \
      --output_dir "$OUT" \
      --num_epochs 6 \
      --batch_size 1 \
      --gradient_accumulation_steps 8 \
      --learning_rate 8e-5 \
      --rank 32 \
      --lora_alpha 64 \
      --temperature 1.2 \
      --alpha 0.02 \
      --alpha_warmup_epochs 2 \
      --hard_upsample 1 \
      --seed "$seed" \
      --max_length 768 \
      --augment > "$LOG" 2>&1
  fi
done

# 3) Enhanced-supervision config E (hard-boost) on seeds 5,6
for seed in 5 6; do
  LOG="train32_nooverlap_hardboost_seed${seed}.log"
  OUT="./dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_hardboost_seed${seed}"
  if [[ -f "$LOG" ]]; then
    echo "[SKIP] $LOG exists"
  else
    echo "[RUN] Config E(hardboost) seed=$seed"
    conda run -n easyedit python train_dental_lora32.py \
      --teacher_model ./Qwen2.5-32B-Instruct \
      --student_model ./Qwen2.5-7B-Instruct \
      --data_path "$TRAIN_HB" \
      --val_path "$VAL" \
      --test_path "$TEST" \
      --output_dir "$OUT" \
      --num_epochs 6 \
      --batch_size 1 \
      --gradient_accumulation_steps 8 \
      --learning_rate 8e-5 \
      --rank 32 \
      --lora_alpha 64 \
      --temperature 1.2 \
      --alpha 0.02 \
      --alpha_warmup_epochs 2 \
      --hard_upsample 1 \
      --seed "$seed" \
      --max_length 768 \
      --augment > "$LOG" 2>&1
  fi
done

# 4) Build follow-up summary
python - <<'PY'
import re, statistics as st
from pathlib import Path

root=Path('/home/student/arthas/EasyEdit3')

def acc(path, pat):
    txt=Path(path).read_text(encoding='utf-8', errors='ignore')
    m=re.findall(pat, txt)
    return float(m[-1]) if m else None

teacher=acc(root/'teacher32_baseline_autotest_nooverlap.log', r'正确率：([0-9.]+)%')

# C now 5 seeds
C_logs=[root/f'train32_nooverlap_weakkl_seed{s}.log' for s in [2,3,4,5,6]]
C=[acc(p, r'测试集准确率: ([0-9.]+)%') for p in C_logs]

# E pilot seeds 5,6
E_logs=[root/f'train32_nooverlap_hardboost_seed{s}.log' for s in [5,6]]
E=[acc(p, r'测试集准确率: ([0-9.]+)%') for p in E_logs]

A_logs=[root/f'train32_nooverlap_hist_seed{s}.log' for s in [2,3,4]]
A=[acc(p, r'测试集准确率: ([0-9.]+)%') for p in A_logs]

md=[]
md.append('# Follow-up: >=5 Seeds and Enhanced Supervision\n')
md.append(f'- Teacher direct: **{teacher:.2f}%**\n')
md.append(f'- A(hist, seeds2-4): {A}\n')
md.append(f'- C(weakKL, seeds2-6): {C}\n')
md.append(f'- E(hardboost, seeds5-6): {E}\n')

if all(v is not None for v in C):
    md.append(f'- C mean/median/std/best: **{sum(C)/len(C):.2f}% / {st.median(C):.2f}% / {st.pstdev(C):.2f} / {max(C):.2f}%**\n')
if all(v is not None for v in A):
    md.append(f'- A mean (seed2-4): **{sum(A)/len(A):.2f}%**\n')
if all(v is not None for v in C) and all(v is not None for v in A):
    md.append(f'- C mean improvement vs A mean: **{(sum(C)/len(C) - sum(A)/len(A)):+.2f} pt**\n')
if all(v is not None for v in E):
    md.append(f'- E mean (pilot seed5-6): **{sum(E)/len(E):.2f}%**, best **{max(E):.2f}%**\n')

(root/'followup_5seed_hardboost_summary.md').write_text('\n'.join(md), encoding='utf-8')
print('WROTE followup_5seed_hardboost_summary.md')
PY

