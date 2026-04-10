#!/usr/bin/env bash
# Doubao Round9: Exact replication of R7 (best stable: mean=79.12%) with 5 seeds
# Purpose: Definitive stability measurement of choice-head → SFT approach
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
BENCH_DIR="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326"
PY="${EASYEDIT_PY:-/home/student/arthas/EasyEdit3/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_doubao_headsft_r9"
RUN_ROOT="$ROOT_DIR/runs/$RUN_ID"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs" "$RUN_ROOT/artifacts"

TRAIN_DATA_GT="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl"
TRAIN_DATA_SOFT="$ROOT_DIR/artifacts/teacher_train_soft_r6_mv5.jsonl"
TRAIN_DATA_HEAD="$RUN_ROOT/artifacts/train_head_distill.jsonl"
HEAD_REPORT="$RUN_ROOT/head_selection_report.json"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
PARAMS_JSON="$ROOT_DIR/grid_params_two_stage_head_then_sft_round9.json"
MASTER_LOG="$RUN_ROOT/logs/headsft_master.log"

{
  echo "[INFO] RUN_ID=$RUN_ID"
  echo "[INFO] Strategy: Exact R7 replication with 5 seeds (42,11,55,7,99)"
  echo "[INFO] Params: alpha_stage1=0.35, lr=0.00012, stage1_ep=1, stage2_ep=2"

  echo "[STEP] build head-distill dataset"
  "$PY" "$BENCH_DIR/distill_runs/shared/build_selective_distill_dataset.py" \
    --gt_data "$TRAIN_DATA_GT" \
    --teacher_soft "$TRAIN_DATA_SOFT" \
    --output "$TRAIN_DATA_HEAD" \
    --report "$HEAD_REPORT" \
    --min_entropy 0.20 \
    --smooth_eps 0.25 \
    --min_margin 0.03

  "$PY" - "$PARAMS_JSON" "$RUN_ROOT" "$PROJECT_ROOT" "$BASE_MODEL" "$TRAIN_DATA_HEAD" "$TRAIN_DATA_GT" "$VAL_DATA" "$TEST_DATA" "$PY" <<'PY2'
import json
import subprocess
import sys
from pathlib import Path

params_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
project_root = Path(sys.argv[3])
base_model = sys.argv[4]
train_head = sys.argv[5]
train_gt = sys.argv[6]
val_data = sys.argv[7]
test_data = sys.argv[8]
py = sys.argv[9]

params = json.loads(params_path.read_text(encoding='utf-8'))
for p in params:
    name = p['name']
    out_dir = run_root / 'outputs' / name
    stage1_dir = out_dir / 'stage1_head'
    stage2_dir = out_dir / 'stage2_sft'
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_log = run_root / 'logs' / f'stage1_{name}.log'
    stage2_log = run_root / 'logs' / f'stage2_{name}.log'

    stage1 = [
        py, str(project_root / 'train_dental_choice_head_distill_doubao.py'),
        '--model_name', base_model,
        '--data_path', train_head,
        '--val_path', val_data,
        '--test_path', test_data,
        '--output_dir', str(stage1_dir),
        '--num_epochs', str(p['num_epochs_stage1']),
        '--batch_size', str(p['batch_size']),
        '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
        '--learning_rate', str(p['learning_rate_stage1']),
        '--rank', str(p['rank']),
        '--lora_alpha', str(p['lora_alpha']),
        '--alpha', str(p['alpha_stage1']),
        '--default_distill_mask', '0',
        '--seed', str(p['seed']),
        '--deterministic',
    ]

    stage2 = [
        py, str(project_root / 'train_dental_lora7_doubao.py'),
        '--model_name', base_model,
        '--data_path', train_gt,
        '--val_path', val_data,
        '--test_path', test_data,
        '--output_dir', str(stage2_dir),
        '--num_epochs', str(p['num_epochs_stage2']),
        '--batch_size', str(p['batch_size']),
        '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
        '--learning_rate', str(p['learning_rate_stage2']),
        '--rank', str(p['rank']),
        '--lora_alpha', str(p['lora_alpha']),
        '--alpha', '0.0',
        '--default_distill_mask', '0',
        '--seed', str(p['seed']),
        '--deterministic',
        '--resume_from', str(stage1_dir),
    ]

    print(f"[RUN][stage1-head] {name} seed={p['seed']}", flush=True)
    with stage1_log.open('w', encoding='utf-8') as lf:
        rc1 = subprocess.run(stage1, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc1 != 0:
        print(f"[FAIL][stage1-head] {name} rc={rc1}", flush=True)
        continue

    print(f"[RUN][stage2-sft] {name} seed={p['seed']}", flush=True)
    with stage2_log.open('w', encoding='utf-8') as lf:
        rc2 = subprocess.run(stage2, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc2 != 0:
        print(f"[FAIL][stage2-sft] {name} rc={rc2}", flush=True)
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
    rows.append({'name': name, 'seed': p['seed'], 'accuracy': acc, 'status': 'ok' if acc is not None else 'failed'})

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

(run_root / 'headsft_results_latest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('[OUT]', run_root / 'headsft_results_latest.json')
for r in rows:
    status = f"{r['accuracy']}%" if r['accuracy'] else 'FAILED'
    print(f"  {r['name']} (seed={r['seed']}): {status}")
if vals:
    print(f"  MEAN={round(st.mean(vals),2)}%  STD={round(st.pstdev(vals),2)}%  BEST={max(vals)}%")
PY3

  echo "[DONE] doubao R9 finished"
  echo "[OUT] run_root=$RUN_ROOT"
} 2>&1 | tee "$MASTER_LOG"
