#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
BENCH_DIR="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326"
PY="${EASYEDIT_PY:-/home/student/arthas/EasyEdit3/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_doubao_selmix_r6b"
RUN_ROOT="$ROOT_DIR/runs/$RUN_ID"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs" "$RUN_ROOT/artifacts"

TRAIN_DATA_GT="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl"
TRAIN_DATA_SOFT="$ROOT_DIR/artifacts/teacher_train_soft_r6_mv5.jsonl"
TRAIN_DATA_MIX="$RUN_ROOT/artifacts/train_selective_mix.jsonl"
MIX_REPORT="$RUN_ROOT/selection_report.json"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
PARAMS_JSON="$ROOT_DIR/grid_params_selective_mix_round6b.json"
MASTER_LOG="$RUN_ROOT/logs/selmix_master.log"

{
  echo "[INFO] RUN_ID=$RUN_ID"

  echo "[STEP] build selective mixed dataset (broader selection)"
  "$PY" "$BENCH_DIR/distill_runs/shared/build_selective_distill_dataset.py" \
    --gt_data "$TRAIN_DATA_GT" \
    --teacher_soft "$TRAIN_DATA_SOFT" \
    --output "$TRAIN_DATA_MIX" \
    --report "$MIX_REPORT" \
    --min_entropy 0.20 \
    --smooth_eps 0.25 \
    --min_margin 0.03

  "$PY" - "$PARAMS_JSON" "$RUN_ROOT" "$PROJECT_ROOT" "$BASE_MODEL" "$TRAIN_DATA_GT" "$TRAIN_DATA_MIX" "$VAL_DATA" "$TEST_DATA" "$PY" <<'PY2'
import json
import subprocess
import sys
from pathlib import Path

params_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
project_root = Path(sys.argv[3])
base_model = sys.argv[4]
train_gt = sys.argv[5]
train_mix = sys.argv[6]
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
        '--data_path', train_mix,
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
        '--default_distill_mask', '0',
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

(run_root / 'selmix_results_latest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('[OUT]', run_root / 'selmix_results_latest.json')
PY3

  echo "[DONE] doubao selective-mix round6b finished"
  echo "[OUT] run_root=$RUN_ROOT"
} 2>&1 | tee "$MASTER_LOG"
