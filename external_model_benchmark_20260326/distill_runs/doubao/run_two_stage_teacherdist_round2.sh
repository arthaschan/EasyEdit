#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
PY="${EASYEDIT_PY:-/home/student/arthas/EasyEdit3/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_doubao_td_r2"
RUN_ROOT="$ROOT_DIR/runs/$RUN_ID"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs"

TRAIN_DATA_GT="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl"
TRAIN_DATA_SOFT="$ROOT_DIR/artifacts/teacher_train_soft_r5.jsonl"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
PARAMS_JSON="$ROOT_DIR/grid_params_two_stage_teacherdist_round2.json"
MASTER_LOG="$RUN_ROOT/logs/td_master.log"

{
  echo "[INFO] RUN_ID=$RUN_ID"

  if [[ ! -f "$TRAIN_DATA_SOFT" ]]; then
    echo "[ERROR] missing soft teacher data: $TRAIN_DATA_SOFT"
    exit 2
  fi

  "$PY" - "$PARAMS_JSON" "$RUN_ROOT" "$PROJECT_ROOT" "$BASE_MODEL" "$TRAIN_DATA_GT" "$TRAIN_DATA_SOFT" "$VAL_DATA" "$TEST_DATA" "$PY" <<'PY2'
import json
import subprocess
import sys
from pathlib import Path

params_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
project_root = Path(sys.argv[3])
base_model = sys.argv[4]
train_gt = sys.argv[5]
train_soft = sys.argv[6]
val_data = sys.argv[7]
test_data = sys.argv[8]
py = sys.argv[9]

params = json.loads(params_path.read_text(encoding='utf-8'))
for p in params:
    name = p['name']
    out_dir = run_root / 'outputs' / name
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_log = run_root / 'logs' / f'stage1_{name}.log'
    stage2_log = run_root / 'logs' / f'stage2_{name}.log'

    stage1 = [
        py, str(project_root / 'train_dental_lora7_doubao.py'),
        '--model_name', base_model,
        '--data_path', train_gt,
        '--val_path', val_data,
        '--test_path', test_data,
        '--output_dir', str(out_dir),
        '--num_epochs', str(p['num_epochs_stage1']),
        '--batch_size', str(p['batch_size']),
        '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
        '--learning_rate', str(p['learning_rate']),
        '--rank', str(p['rank']),
        '--lora_alpha', str(p['lora_alpha']),
        '--alpha', '0.0',
        '--default_distill_mask', '0',
        '--seed', str(p['seed']),
        '--deterministic',
        '--resume',
    ]

    stage2 = [
        py, str(project_root / 'train_dental_lora7_doubao.py'),
        '--model_name', base_model,
        '--data_path', train_soft,
        '--val_path', val_data,
        '--test_path', test_data,
        '--output_dir', str(out_dir),
        '--num_epochs', str(p['num_epochs_stage2']),
        '--batch_size', str(p['batch_size']),
        '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
        '--learning_rate', str(p['learning_rate']),
        '--rank', str(p['rank']),
        '--lora_alpha', str(p['lora_alpha']),
        '--alpha', str(p['alpha_stage2']),
        '--default_distill_mask', '1',
        '--seed', str(p['seed']),
        '--deterministic',
        '--resume',
        '--use_teacher_dist',
    ]

    print(f"[RUN][stage1] {name} seed={p['seed']}", flush=True)
    with stage1_log.open('w', encoding='utf-8') as lf:
        rc1 = subprocess.run(stage1, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc1 != 0:
        print(f"[FAIL][stage1] {name} rc={rc1}", flush=True)
        continue

    print(f"[RUN][stage2] {name} seed={p['seed']}", flush=True)
    with stage2_log.open('w', encoding='utf-8') as lf:
        rc2 = subprocess.run(stage2, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc2 != 0:
        print(f"[FAIL][stage2] {name} rc={rc2}", flush=True)
    else:
        print(f"[DONE] {name}", flush=True)
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
prev_best = 80.72

rows = []
for p in params:
    name = p['name']
    log_path = run_root / 'logs' / f'stage2_{name}.log'
    acc = None
    if log_path.exists():
        txt = log_path.read_text(encoding='utf-8', errors='ignore')
        m = re.findall(r"测试集准确率:\s*([0-9]+(?:\.[0-9]+)?)%", txt)
        if m:
            acc = float(m[-1])
    rows.append({
      'name': name,
      'seed': p['seed'],
      'accuracy': acc,
      'status': 'ok' if acc is not None else 'failed'
    })

ok = [r for r in rows if r['status'] == 'ok']
ok.sort(key=lambda x: x['accuracy'], reverse=True)
vals = [r['accuracy'] for r in ok]
summary = {
  'run_root': str(run_root),
  'baseline': baseline,
  'previous_best': prev_best,
  'n': len(vals),
  'best': ok[0] if ok else None,
  'mean': round(st.mean(vals), 2) if vals else None,
  'std': round(st.pstdev(vals), 2) if len(vals) > 1 else 0.0 if vals else None,
  'rows': rows,
}

(run_root / 'td_results_latest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
lines = [
  '# Doubao Two-Stage TeacherDist Round2 Results',
  '',
  f'- baseline_student_acc: {baseline:.2f}%',
  f'- previous_best_acc: {prev_best:.2f}%',
  '',
  '| Rank | Name | Seed | Accuracy(%) | Delta vs Baseline | Delta vs PrevBest | Status |',
  '|---:|---|---:|---:|---:|---:|---|',
]
rank = 1
for r in ok:
    lines.append(f"| {rank} | {r['name']} | {r['seed']} | {r['accuracy']:.2f} | {r['accuracy']-baseline:+.2f} | {r['accuracy']-prev_best:+.2f} | ok |")
    rank += 1
for r in rows:
    if r['status'] != 'ok':
        lines.append(f"| - | {r['name']} | {r['seed']} | - | - | - | failed |")

(run_root / 'td_results_latest.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('[OUT]', run_root / 'td_results_latest.json')
print('[OUT]', run_root / 'td_results_latest.md')
PY3

  echo "[DONE] doubao two-stage teacherdist round2 finished"
  echo "[OUT] run_root=$RUN_ROOT"
} 2>&1 | tee "$MASTER_LOG"
