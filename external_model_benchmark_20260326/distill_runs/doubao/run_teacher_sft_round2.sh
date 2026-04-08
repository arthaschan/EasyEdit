#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
PY="${EASYEDIT_PY:-/home/student/arthas/EasyEdit3/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_doubao_sft_round2"
RUN_ROOT="$ROOT_DIR/runs/$RUN_ID"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs"

TEACHER_DATA="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/doubao/artifacts/teacher_train.jsonl"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
PARAMS_JSON="$ROOT_DIR/grid_params_teacher_sft_round2.json"
MASTER_LOG="$RUN_ROOT/logs/sft_master.log"

{
  echo "[INFO] RUN_ID=$RUN_ID"
  if [[ ! -f "$TEACHER_DATA" ]]; then
    echo "[ERROR] teacher_data missing: $TEACHER_DATA"
    exit 2
  fi

  "$PY" - "$PARAMS_JSON" "$RUN_ROOT" "$PROJECT_ROOT" "$BASE_MODEL" "$TEACHER_DATA" "$VAL_DATA" "$TEST_DATA" "$PY" <<'PY2'
import json
import subprocess
import sys
from pathlib import Path

params_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
project_root = Path(sys.argv[3])
base_model = sys.argv[4]
teacher_data = sys.argv[5]
val_data = sys.argv[6]
test_data = sys.argv[7]
py = sys.argv[8]

params = json.loads(params_path.read_text(encoding='utf-8'))
for p in params:
    name = p['name']
    out_dir = run_root / 'outputs' / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_root / 'logs' / f'train_{name}.log'

    cmd = [
        py, str(project_root / 'train_dental_lora7_doubao.py'),
        '--model_name', base_model,
        '--data_path', teacher_data,
        '--val_path', val_data,
        '--test_path', test_data,
        '--output_dir', str(out_dir),
        '--num_epochs', str(p['num_epochs']),
        '--batch_size', str(p['batch_size']),
        '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
        '--learning_rate', str(p['learning_rate']),
        '--rank', str(p['rank']),
        '--lora_alpha', str(p['lora_alpha']),
        '--temperature', '1.5',
        '--alpha', '0.0',
        '--default_distill_mask', '0',
        '--resume',
    ]

    print(f"[RUN][sft] {name}", flush=True)
    with log_path.open('w', encoding='utf-8') as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"[FAIL][sft] {name} rc={rc}", flush=True)
    else:
        print(f"[DONE][sft] {name}", flush=True)
PY2

  "$PY" - "$PARAMS_JSON" "$RUN_ROOT" <<'PY3'
import json
import re
import statistics as st
import sys
from pathlib import Path

params = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
run_root = Path(sys.argv[2])
baseline = 77.11

rows = []
for p in params:
    name = p['name']
    log_path = run_root / 'logs' / f'train_{name}.log'
    acc = None
    if log_path.exists():
        txt = log_path.read_text(encoding='utf-8', errors='ignore')
        m = re.findall(r"测试集准确率:\s*([0-9]+(?:\.[0-9]+)?)%", txt)
        if m:
            acc = float(m[-1])
    rows.append({'name': name, 'learning_rate': p['learning_rate'], 'num_epochs': p['num_epochs'], 'accuracy': acc, 'status': 'ok' if acc is not None else 'failed'})

ok = [r for r in rows if r['status'] == 'ok']
ok.sort(key=lambda x: x['accuracy'], reverse=True)
vals = [r['accuracy'] for r in ok]
best = ok[0] if ok else None

summary = {
  'run_root': str(run_root),
  'baseline': baseline,
  'n': len(vals),
  'best': best,
  'mean': round(st.mean(vals), 2) if vals else None,
  'std': round(st.pstdev(vals), 2) if len(vals) > 1 else 0.0 if vals else None,
  'rows': rows,
}

(run_root / 'sft_results_latest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
lines = [
  '# Doubao Teacher-Label SFT Results',
  '',
  f'- baseline_student_acc: {baseline:.2f}%',
  '',
  '| Rank | Name | lr | epochs | Accuracy(%) | Delta vs Baseline | Status |',
  '|---:|---|---:|---:|---:|---:|---|',
]
rank = 1
for r in ok:
    lines.append(f"| {rank} | {r['name']} | {r['learning_rate']:.5f} | {r['num_epochs']} | {r['accuracy']:.2f} | {r['accuracy']-baseline:+.2f} | ok |")
    rank += 1
for r in rows:
    if r['status'] != 'ok':
        lines.append(f"| - | {r['name']} | {r['learning_rate']:.5f} | {r['num_epochs']} | - | - | failed |")

(run_root / 'sft_results_latest.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('[OUT]', run_root / 'sft_results_latest.json')
print('[OUT]', run_root / 'sft_results_latest.md')
PY3

  echo "[DONE] doubao sft round2 finished"
  echo "[OUT] run_root=$RUN_ROOT"
} 2>&1 | tee "$MASTER_LOG"
