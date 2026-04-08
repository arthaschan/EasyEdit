#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILL_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
BENCH_DIR="$(cd "$DISTILL_ROOT/.." && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
MODEL_PATH="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
PY="/home/student/arthas/EasyEdit3/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

TRAIN_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
SYSTEM_PROMPT="$BENCH_DIR/prompt_system.txt"
TEACHER_DATA="$ROOT_DIR/artifacts/teacher_train.jsonl"
CANDIDATE_JSON="$ROOT_DIR/teacher_candidate.json"
GRID="$ROOT_DIR/param_grid.tsv"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT_DIR/sweeps/$TS"
LOG_DIR="$RUN_DIR/logs"
OUT_DIR="$RUN_DIR/models"
REPORT_JSON="$RUN_DIR/sweep_results.json"
REPORT_MD="$RUN_DIR/sweep_results.md"
CSV="$RUN_DIR/sweep_results.csv"
mkdir -p "$LOG_DIR" "$OUT_DIR"

# 1) Ensure teacher-labeled data exists and is resumable
"$PY" "$DISTILL_ROOT/shared/generate_teacher_labels.py" \
  --dataset "$TRAIN_DATA" \
  --candidate "$CANDIDATE_JSON" \
  --system_prompt "$SYSTEM_PROMPT" \
  --output "$TEACHER_DATA" \
  --sample_size 0 \
  --seed 42 \
  --timeout_sec 120 \
  --max_tokens 16 \
  --max_retries 2 \
  --request_interval_sec 0.8 \
  --resume | tee "$LOG_DIR/teacher_labeling.log"

LINES=$(wc -l < "$TEACHER_DATA")
if [[ "$LINES" -le 0 ]]; then
  echo "[FATAL] teacher data empty"
  exit 2
fi

echo "id,epochs,batch_size,grad_acc,lr,rank,lora_alpha,temperature,alpha,status,accuracy,log_path,out_dir" > "$CSV"

while IFS=$'\t' read -r id epochs batch grad lr rank lora_alpha temp alpha; do
  [[ -z "${id:-}" ]] && continue
  [[ "$id" =~ ^# ]] && continue

  combo_out="$OUT_DIR/$id"
  combo_log="$LOG_DIR/${id}.log"
  mkdir -p "$combo_out"

  echo "[RUN] $id epochs=$epochs batch=$batch grad_acc=$grad lr=$lr rank=$rank lora_alpha=$lora_alpha temp=$temp alpha=$alpha"
  set +e
  "$PY" "$PROJECT_ROOT/train_dental_lora7.py" \
    --model_name "$MODEL_PATH" \
    --data_path "$TEACHER_DATA" \
    --val_path "$VAL_DATA" \
    --test_path "$TEST_DATA" \
    --output_dir "$combo_out" \
    --num_epochs "$epochs" \
    --batch_size "$batch" \
    --gradient_accumulation_steps "$grad" \
    --learning_rate "$lr" \
    --rank "$rank" \
    --lora_alpha "$lora_alpha" \
    --temperature "$temp" \
    --alpha "$alpha" \
    2>&1 | tee "$combo_log"
  rc=${PIPESTATUS[0]}
  set -e

  acc=$(grep -oP '测试集准确率:\s*\K[0-9.]+(?=%)' "$combo_log" | tail -n 1 || true)
  if [[ -z "$acc" ]]; then
    acc=""
  fi

  status="ok"
  if [[ "$rc" -ne 0 ]]; then
    status="failed"
  fi

  echo "$id,$epochs,$batch,$grad,$lr,$rank,$lora_alpha,$temp,$alpha,$status,$acc,$combo_log,$combo_out" >> "$CSV"
done < "$GRID"

"$PY" - "$CSV" "$REPORT_JSON" "$REPORT_MD" <<'PY'
import csv, json, sys
from pathlib import Path

csv_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
md_path = Path(sys.argv[3])

rows = []
with csv_path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            r['accuracy'] = float(r['accuracy']) if r['accuracy'] else None
        except Exception:
            r['accuracy'] = None
        rows.append(r)

ok_rows = [r for r in rows if r.get('status') == 'ok' and r.get('accuracy') is not None]
ok_rows.sort(key=lambda x: x['accuracy'], reverse=True)

payload = {'results': rows, 'best': ok_rows[0] if ok_rows else None}
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

lines = [
    '# DeepSeek Distillation Parameter Sweep',
    '',
    f'- total_combinations: {len(rows)}',
    f'- successful_with_accuracy: {len(ok_rows)}',
    '',
    '| Rank | ID | Accuracy(%) | Epochs | Batch | GradAcc | LR | Rank(r) | LoraAlpha | Temp | Alpha | Status |',
    '|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|',
]

rank = 1
for r in ok_rows:
    lines.append(
        f"| {rank} | {r['id']} | {r['accuracy']:.2f} | {r['epochs']} | {r['batch_size']} | {r['grad_acc']} | {r['lr']} | {r['rank']} | {r['lora_alpha']} | {r['temperature']} | {r['alpha']} | {r['status']} |"
    )
    rank += 1

for r in rows:
    if r in ok_rows:
        continue
    acc = '-' if r.get('accuracy') is None else f"{r['accuracy']:.2f}"
    lines.append(
        f"| - | {r['id']} | {acc} | {r['epochs']} | {r['batch_size']} | {r['grad_acc']} | {r['lr']} | {r['rank']} | {r['lora_alpha']} | {r['temperature']} | {r['alpha']} | {r['status']} |"
    )

if ok_rows:
    b = ok_rows[0]
    lines += [
        '',
        '## Best Combo',
        '',
        f"- id: {b['id']}",
        f"- accuracy: {b['accuracy']:.2f}%",
        f"- params: lr={b['lr']}, rank={b['rank']}, lora_alpha={b['lora_alpha']}, temp={b['temperature']}, alpha={b['alpha']}",
    ]

md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f"[OUT] {json_path}")
print(f"[OUT] {md_path}")
PY

echo "[DONE] sweep_dir=$RUN_DIR"
