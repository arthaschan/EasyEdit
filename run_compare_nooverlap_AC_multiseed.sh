#!/usr/bin/env bash
set -euo pipefail
cd /home/student/arthas/EasyEdit3

TRAIN=data/cmexam_dental_choice_train_nooverlap.jsonl
VAL=data/cmexam_dental_choice_val_nooverlap.jsonl
TEST=data/cmexam_dental_choice_test.jsonl

run_one () {
  local cfg="$1" seed="$2" out log
  if [[ "$cfg" == "A" ]]; then
    out="./dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_hist_seed${seed}"
    log="train32_nooverlap_hist_seed${seed}.log"
    [[ -f "$log" ]] && grep -q '测试集准确率' "$log" && { echo "[SKIP] A seed${seed} exists"; return; }
    conda run -n easyedit python train_dental_lora32.py \
      --teacher_model ./Qwen2.5-32B-Instruct \
      --student_model ./Qwen2.5-7B-Instruct \
      --data_path "$TRAIN" --val_path "$VAL" --test_path "$TEST" \
      --output_dir "$out" --num_epochs 4 --batch_size 1 --gradient_accumulation_steps 8 \
      --learning_rate 8e-5 --rank 16 --lora_alpha 32 --temperature 1.5 --alpha 0.08 \
      --alpha_warmup_epochs 1 --hard_upsample 1 --seed "$seed" --max_length 768 > "$log" 2>&1
  else
    out="./dental_qwen2.5_7b_choice_lora_distill_from32_nooverlap_weakkl_seed${seed}"
    log="train32_nooverlap_weakkl_seed${seed}.log"
    [[ -f "$log" ]] && grep -q '测试集准确率' "$log" && { echo "[SKIP] C seed${seed} exists"; return; }
    conda run -n easyedit python train_dental_lora32.py \
      --teacher_model ./Qwen2.5-32B-Instruct \
      --student_model ./Qwen2.5-7B-Instruct \
      --data_path "$TRAIN" --val_path "$VAL" --test_path "$TEST" \
      --output_dir "$out" --num_epochs 6 --batch_size 1 --gradient_accumulation_steps 8 \
      --learning_rate 8e-5 --rank 32 --lora_alpha 64 --temperature 1.2 --alpha 0.02 \
      --alpha_warmup_epochs 2 --hard_upsample 1 --seed "$seed" --max_length 768 --augment > "$log" 2>&1
  fi
  echo "[DONE] $cfg seed${seed}"
}

echo "[INFO] Start $(date '+%F %T')"
for s in 2 3 4; do run_one A "$s"; done
for s in 2 3 4; do run_one C "$s"; done

python - <<'PY'
import re, statistics as st
from pathlib import Path

def read_acc(path):
    t=Path(path).read_text(encoding='utf-8',errors='ignore')
    m=re.findall(r'测试集准确率: ([0-9.]+)%', t)
    return float(m[-1]) if m else None

def row(cfg,seed):
    p=f'train32_nooverlap_{cfg}_seed{seed}.log'
    a=read_acc(p)
    return p,a

A=[]; C=[]
for s in [2,3,4]:
    A.append((s, read_acc(f'train32_nooverlap_hist_seed{s}.log')))
    C.append((s, read_acc(f'train32_nooverlap_weakkl_seed{s}.log')))

teacher_txt=Path('teacher32_baseline_autotest_nooverlap.log').read_text(encoding='utf-8',errors='ignore')
mt=re.findall(r'正确率：([0-9.]+)%', teacher_txt)
teacher=float(mt[-1]) if mt else None

def stats(vals):
    vals=[v for v in vals if v is not None]
    return (sum(vals)/len(vals), st.median(vals), st.pstdev(vals), max(vals), min(vals))

Avals=[v for _,v in A]; Cvals=[v for _,v in C]
Am, Amed, Astd, Amax, Amin = stats(Avals)
Cm, Cmed, Cstd, Cmax, Cmin = stats(Cvals)

lines=[]
lines.append('# No-overlap Multi-seed Comparison (A vs C)\n')
lines.append(f'- Teacher 32B direct: **{teacher:.2f}%**\n')
lines.append('## Per-seed\n')
lines.append('| Config | Seed2 | Seed3 | Seed4 | Mean | Median | Std | Best | Gap to Teacher(best) |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
lines.append(f"| A(hist) | {Avals[0]:.2f}% | {Avals[1]:.2f}% | {Avals[2]:.2f}% | {Am:.2f}% | {Amed:.2f}% | {Astd:.2f} | {Amax:.2f}% | {Amax-teacher:+.2f}pt |")
lines.append(f"| C(weakKL) | {Cvals[0]:.2f}% | {Cvals[1]:.2f}% | {Cvals[2]:.2f}% | {Cm:.2f}% | {Cmed:.2f}% | {Cstd:.2f} | {Cmax:.2f}% | {Cmax-teacher:+.2f}pt |")
Path('compare_nooverlap_AC_multiseed_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('WROTE compare_nooverlap_AC_multiseed_summary.md')
PY

echo "[INFO] End $(date '+%F %T')"
