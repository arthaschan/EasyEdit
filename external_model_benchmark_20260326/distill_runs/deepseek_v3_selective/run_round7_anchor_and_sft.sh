#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
PY="${EASYEDIT_PY:-/home/student/arthas/EasyEdit3/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

RUN_BASE="$(date +%Y%m%d_%H%M%S)_round7"
RUN_ANCHOR_ROOT="$ROOT_DIR/runs/${RUN_BASE}_anchor_lowalpha"
RUN_SFT_ROOT="$ROOT_DIR/runs/${RUN_BASE}_sft5"
mkdir -p "$RUN_ANCHOR_ROOT/logs" "$RUN_ANCHOR_ROOT/outputs" "$RUN_SFT_ROOT/logs" "$RUN_SFT_ROOT/outputs"

TRAIN_DATA_GT="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
FIXED_SELECTIVE="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective/runs/20260407_094814/artifacts/teacher_train_selective.jsonl"

if [[ ! -f "$FIXED_SELECTIVE" ]]; then
  echo "[ERROR] missing fixed selective labels: $FIXED_SELECTIVE"
  exit 2
fi

run_anchor_grid() {
  local params_json="$1"
  local run_root="$2"

  "$PY" - "$params_json" "$run_root" "$PROJECT_ROOT" "$BASE_MODEL" "$TRAIN_DATA_GT" "$FIXED_SELECTIVE" "$VAL_DATA" "$TEST_DATA" "$PY" <<'PY2'
import json
import subprocess
import sys
from pathlib import Path

params_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
project_root = Path(sys.argv[3])
base_model = sys.argv[4]
train_gt = sys.argv[5]
train_selective = sys.argv[6]
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
        py, str(project_root / 'train_dental_lora7_deepseek.py'),
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
        '--temperature', '1.5',
        '--alpha', '0.0',
        '--default_distill_mask', '0',
        '--resume',
    ]

    stage2 = [
        py, str(project_root / 'train_dental_lora7_deepseek.py'),
        '--model_name', base_model,
        '--data_path', train_selective,
        '--val_path', val_data,
        '--test_path', test_data,
        '--output_dir', str(out_dir),
        '--num_epochs', str(p['num_epochs_stage2']),
        '--batch_size', str(p['batch_size']),
        '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
        '--learning_rate', str(p['learning_rate']),
        '--rank', str(p['rank']),
        '--lora_alpha', str(p['lora_alpha']),
        '--temperature', str(p['temperature']),
        '--alpha', str(p['alpha']),
        '--default_distill_mask', '0',
        '--resume',
    ]

    print(f"[RUN][anchor][stage1] {name}", flush=True)
    with stage1_log.open('w', encoding='utf-8') as lf:
        r1 = subprocess.run(stage1, stdout=lf, stderr=subprocess.STDOUT)
    if r1.returncode != 0:
        print(f"[FAIL][anchor][stage1] {name} rc={r1.returncode}", flush=True)
        continue

    print(f"[RUN][anchor][stage2] {name}", flush=True)
    with stage2_log.open('w', encoding='utf-8') as lf:
        r2 = subprocess.run(stage2, stdout=lf, stderr=subprocess.STDOUT)
    if r2.returncode != 0:
        print(f"[FAIL][anchor][stage2] {name} rc={r2.returncode}", flush=True)
    else:
        print(f"[DONE][anchor] {name}", flush=True)
PY2
}

run_sft_repro() {
  local params_json="$1"
  local run_root="$2"

  "$PY" - "$params_json" "$run_root" "$PROJECT_ROOT" "$BASE_MODEL" "$TRAIN_DATA_GT" "$VAL_DATA" "$TEST_DATA" "$PY" <<'PY3'
import json
import subprocess
import sys
from pathlib import Path

params_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
project_root = Path(sys.argv[3])
base_model = sys.argv[4]
train_gt = sys.argv[5]
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
        py, str(project_root / 'train_dental_lora7_deepseek.py'),
        '--model_name', base_model,
        '--data_path', train_gt,
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
PY3
}

summarize_sft() {
  local params_json="$1"
  local run_root="$2"
  "$PY" - "$params_json" "$run_root" <<'PY4'
import json
import re
import statistics as st
import sys
from pathlib import Path

params_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
baseline = 77.11
hist_best = 79.52

params = json.loads(params_path.read_text(encoding='utf-8'))
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
    rows.append({"name": name, "accuracy": acc, "status": "ok" if acc is not None else "failed"})

ok = [r for r in rows if r['status'] == 'ok']
vals = [r['accuracy'] for r in ok]
ok.sort(key=lambda x: x['accuracy'], reverse=True)

summary = {
    "run_root": str(run_root),
    "n": len(vals),
    "baseline": baseline,
    "historical_best": hist_best,
    "mean": round(st.mean(vals), 2) if vals else None,
    "std": round(st.pstdev(vals), 2) if len(vals) > 1 else 0.0 if vals else None,
    "min": round(min(vals), 2) if vals else None,
    "max": round(max(vals), 2) if vals else None,
    "best_vs_baseline": round(max(vals) - baseline, 2) if vals else None,
    "best_vs_historical_best": round(max(vals) - hist_best, 2) if vals else None,
    "rows": rows,
}

(run_root / 'sft_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
lines = [
    '# SFT-5 Summary',
    '',
    f"- run_root: {run_root}",
    f"- n: {summary['n']}",
    f"- baseline: {baseline:.2f}%",
    f"- historical_best: {hist_best:.2f}%",
]
if vals:
    lines += [
        f"- mean: {summary['mean']:.2f}%",
        f"- std: {summary['std']:.2f}",
        f"- min/max: {summary['min']:.2f}% / {summary['max']:.2f}%",
        f"- best_vs_baseline: {summary['best_vs_baseline']:+.2f}",
        f"- best_vs_historical_best: {summary['best_vs_historical_best']:+.2f}",
        '',
        '| Rank | Name | Accuracy(%) | Delta vs Baseline |',
        '|---:|---|---:|---:|',
    ]
    for i, r in enumerate(ok, start=1):
        lines.append(f"| {i} | {r['name']} | {r['accuracy']:.2f} | {r['accuracy']-baseline:+.2f} |")
else:
    lines.append('- no valid runs')

(run_root / 'sft_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('[OUT]', run_root / 'sft_summary.json')
print('[OUT]', run_root / 'sft_summary.md')
PY4
}

MASTER_LOG="$ROOT_DIR/runs/${RUN_BASE}_master.log"
{
  echo "[INFO] RUN_BASE=$RUN_BASE"
  echo "[INFO] fixed_teacher_labels=$FIXED_SELECTIVE"

  echo "[STEP] anchor low-alpha distillation"
  run_anchor_grid "$ROOT_DIR/grid_params_round7_anchor_lowalpha.json" "$RUN_ANCHOR_ROOT"
  "$PY" "$ROOT_DIR/shared/summarize_two_stage_results.py" \
    --run_root "$RUN_ANCHOR_ROOT" \
    --params "$ROOT_DIR/grid_params_round7_anchor_lowalpha.json" \
    --baseline 77.11

  echo "[STEP] pure sft 5-repeat"
  run_sft_repro "$ROOT_DIR/grid_params_round7_sft5.json" "$RUN_SFT_ROOT"
  summarize_sft "$ROOT_DIR/grid_params_round7_sft5.json" "$RUN_SFT_ROOT"

  echo "[DONE] round7 finished"
  echo "[OUT] anchor_root=$RUN_ANCHOR_ROOT"
  echo "[OUT] sft_root=$RUN_SFT_ROOT"
} 2>&1 | tee "$MASTER_LOG"

echo "[OUT] master_log=$MASTER_LOG"
