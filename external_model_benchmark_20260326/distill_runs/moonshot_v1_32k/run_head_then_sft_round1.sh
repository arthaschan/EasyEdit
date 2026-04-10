#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/student/arthas/EasyEdit3"
BENCH_DIR="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326"
PY="${EASYEDIT_PY:-/home/student/arthas/EasyEdit3/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_moonshot_headsft_r1"
RUN_ROOT="$ROOT_DIR/runs/$RUN_ID"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs" "$RUN_ROOT/artifacts"

MOONSHOT_TEACHER_RAW="$ROOT_DIR/artifacts/teacher_train.jsonl"
MOONSHOT_TEACHER_SOFT="$RUN_ROOT/artifacts/teacher_train_soft.jsonl"
TRAIN_DATA_GT="$PROJECT_ROOT/data/cmexam_dental_choice_train_nooverlap.jsonl"
TRAIN_DATA_HEAD="$RUN_ROOT/artifacts/train_head_distill.jsonl"
HEAD_REPORT="$RUN_ROOT/head_selection_report.json"
VAL_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="$PROJECT_ROOT/Qwen2.5-7B-Instruct"
PARAMS_JSON="$ROOT_DIR/grid_params_head_then_sft_round1.json"
MASTER_LOG="$RUN_ROOT/logs/moonshot_master.log"

{
  echo "[INFO] RUN_ID=$RUN_ID"
  echo "[INFO] Strategy: Choice-head distill (Moonshot teacher) -> GT SFT"
  echo "[INFO] Note: current moonshot teacher labels exactly match GT on train set; this run tests whether two-stage smoothing still helps"
  echo "[INFO] Params: alpha_stage1=0.35, lr=0.00012, stage1_ep=1, stage2_ep=2"

  echo "[STEP 1] Convert Moonshot hard labels to soft (label smoothing eps=0.25)"
  "$PY" "$BENCH_DIR/distill_runs/deepseek_opus/prepare_soft_labels.py" \
    --input "$MOONSHOT_TEACHER_RAW" \
    --output "$MOONSHOT_TEACHER_SOFT" \
    --smooth_eps 0.25

  echo "[STEP 2] Build head-distill dataset with entropy/margin filtering"
  "$PY" "$BENCH_DIR/distill_runs/shared/build_selective_distill_dataset.py" \
    --gt_data "$TRAIN_DATA_GT" \
    --teacher_soft "$MOONSHOT_TEACHER_SOFT" \
    --output "$TRAIN_DATA_HEAD" \
    --report "$HEAD_REPORT" \
    --min_entropy 0.20 \
    --smooth_eps 0.25 \
    --min_margin 0.03

  echo "[STEP 3] Two-stage training (3 seeds)"
  "$PY" "$BENCH_DIR/distill_runs/augmented_distill/run_two_stage_training.py" \
    --params "$PARAMS_JSON" \
    --run_root "$RUN_ROOT" \
    --project_root "$PROJECT_ROOT" \
    --base_model "$BASE_MODEL" \
    --train_head "$TRAIN_DATA_HEAD" \
    --train_gt "$TRAIN_DATA_GT" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --teacher_prefix moonshot \
    --py "$PY"

  echo "[STEP 4] Collect results"
  "$PY" - "$PARAMS_JSON" "$RUN_ROOT" <<'PY2'
import json
import re
import statistics as st
import sys
from pathlib import Path

params = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
run_root = Path(sys.argv[2])
baseline = 77.11
single_stage = 74.70

rows = []
for p in params:
    name = f"moonshot_{p['name']}"
    log_path = run_root / 'logs' / f'stage2_{name}.log'
    acc = None
    val_acc = None
    if log_path.exists():
        txt = log_path.read_text(encoding='utf-8', errors='ignore')
        test_hits = re.findall(r"测试集准确率:\s*([0-9]+(?:\.[0-9]+)?)%", txt)
        val_hits = re.findall(r"验证准确率:\s*([0-9]+(?:\.[0-9]+)?)%", txt)
        if test_hits:
            acc = float(test_hits[-1])
        if val_hits:
            val_acc = float(val_hits[-1])
    rows.append({
        'name': name,
        'seed': p['seed'],
        'val_accuracy': val_acc,
        'test_accuracy': acc,
        'status': 'ok' if acc is not None else 'failed',
    })

ok = [r for r in rows if r['status'] == 'ok']
ok.sort(key=lambda x: x['test_accuracy'], reverse=True)
vals = [r['test_accuracy'] for r in ok]
summary = {
    'run_root': str(run_root),
    'teacher': 'moonshot-v1-32k',
    'approach': 'choice-head-distill -> GT-SFT',
    'baseline': baseline,
    'single_stage_moonshot': single_stage,
    'n': len(vals),
    'best': ok[0] if ok else None,
    'mean': round(st.mean(vals), 2) if vals else None,
    'std': round(st.pstdev(vals), 2) if len(vals) > 1 else 0.0 if vals else None,
    'rows': rows,
}

(run_root / 'moonshot_headsft_results_latest.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding='utf-8',
)
print('[OUT]', run_root / 'moonshot_headsft_results_latest.json')
for r in rows:
    status = f"{r['test_accuracy']}%" if r['test_accuracy'] is not None else 'FAILED'
    print(f"  {r['name']} (seed={r['seed']}): {status}")
if vals:
    print(f"  MEAN={round(st.mean(vals),2)}%  STD={round(st.pstdev(vals),2)}%  BEST={max(vals)}%")
PY2

  echo "[DONE] Moonshot two-stage round1 finished"
  echo "[OUT] run_root=$RUN_ROOT"
} 2>&1 | tee "$MASTER_LOG"