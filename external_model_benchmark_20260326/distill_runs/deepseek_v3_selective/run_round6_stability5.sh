#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
PY="${EASYEDIT_PY:-/home/student/arthas/EasyEdit3/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_stability5"
RUN_ROOT="$ROOT_DIR/runs/$RUN_ID"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs"

TRAIN_DATA_GT="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl"
VAL_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
FIXED_SELECTIVE="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective/runs/20260407_094814/artifacts/teacher_train_selective.jsonl"
PARAMS_JSON="$ROOT_DIR/grid_params_round6_stability5.json"
BASELINE="77.11"

if [[ ! -f "$FIXED_SELECTIVE" ]]; then
  echo "[ERROR] fixed selective labels not found: $FIXED_SELECTIVE"
  exit 2
fi

MASTER_LOG="$RUN_ROOT/logs/stability5_master.log"

{
  echo "[INFO] RUN_ID=$RUN_ID"
  echo "[INFO] fixed_teacher_labels=$FIXED_SELECTIVE"

  "$PY" - "$PARAMS_JSON" "$RUN_ROOT" "$PROJECT_ROOT" "$BASE_MODEL" "$TRAIN_DATA_GT" "$FIXED_SELECTIVE" "$VAL_DATA" "$TEST_DATA" "$PY" <<'PY2'
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
        py,
        str(project_root / 'train_dental_lora7_deepseek.py'),
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
        py,
        str(project_root / 'train_dental_lora7_deepseek.py'),
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
        '--default_distill_mask', '1',
        '--resume',
    ]

    print(f"[RUN][stage1] {name}", flush=True)
    with stage1_log.open('w', encoding='utf-8') as lf:
        r1 = subprocess.run(stage1, stdout=lf, stderr=subprocess.STDOUT)
    if r1.returncode != 0:
        print(f"[FAIL][stage1] {name} rc={r1.returncode}", flush=True)
        continue

    print(f"[RUN][stage2] {name}", flush=True)
    with stage2_log.open('w', encoding='utf-8') as lf:
        r2 = subprocess.run(stage2, stdout=lf, stderr=subprocess.STDOUT)
    if r2.returncode != 0:
        print(f"[FAIL][stage2] {name} rc={r2.returncode}", flush=True)
    else:
        print(f"[DONE] {name}", flush=True)
PY2

  "$PY" "$ROOT_DIR/shared/summarize_two_stage_results.py" \
    --run_root "$RUN_ROOT" \
    --params "$PARAMS_JSON" \
    --baseline "$BASELINE"

  "$PY" - "$RUN_ROOT" <<'PY3'
import json
from pathlib import Path
import statistics as st
import sys

run_root = Path(sys.argv[1])
baseline = 77.11
hist_best = 79.52
obj = json.loads((run_root / "two_stage_results_latest.json").read_text(encoding="utf-8"))
vals = [r["accuracy"] for r in obj.get("results", []) if r.get("status") == "ok" and r.get("accuracy") is not None]

summary = {
    "run_root": str(run_root),
    "n": len(vals),
    "baseline": baseline,
    "historical_best": hist_best,
    "mean": round(float(st.mean(vals)), 2) if vals else None,
    "std": round(float(st.pstdev(vals)), 2) if len(vals) > 1 else 0.0 if vals else None,
    "min": round(float(min(vals)), 2) if vals else None,
    "max": round(float(max(vals)), 2) if vals else None,
    "best_vs_baseline": round(float(max(vals) - baseline), 2) if vals else None,
    "best_vs_hist_best": round(float(max(vals) - hist_best), 2) if vals else None,
    "all_accuracies": vals,
}

(run_root / "stability_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

lines = []
lines.append("# Stability-5 Summary")
lines.append("")
lines.append(f"- run_root: {run_root}")
lines.append(f"- n: {summary['n']}")
lines.append(f"- baseline: {baseline:.2f}%")
lines.append(f"- historical_best: {hist_best:.2f}%")
if vals:
    lines.append(f"- mean: {summary['mean']:.2f}%")
    lines.append(f"- std: {summary['std']:.2f}")
    lines.append(f"- min/max: {summary['min']:.2f}% / {summary['max']:.2f}%")
    lines.append(f"- best_vs_baseline: {summary['best_vs_baseline']:+.2f}")
    lines.append(f"- best_vs_historical_best: {summary['best_vs_hist_best']:+.2f}")
else:
    lines.append("- no valid runs")

(run_root / "stability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[OUT]", run_root / "stability_summary.json")
print("[OUT]", run_root / "stability_summary.md")
PY3

  echo "[DONE] round6 stability5 finished"
  echo "[OUT] run_root=$RUN_ROOT"
} 2>&1 | tee "$MASTER_LOG"

echo "[OUT] master_log=$MASTER_LOG"
