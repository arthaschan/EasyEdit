#!/usr/bin/env bash
# Resume Doubao R9: s11 test-eval + s55/s7/s99 full runs, then DeepSeek Opus R1
set -euo pipefail

PROJECT_ROOT="/home/student/arthas/EasyEdit3"
BENCH_DIR="$PROJECT_ROOT/external_model_benchmark_20260326"
PY="${EASYEDIT_PY:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then PY="$(command -v python3)"; fi

DOUBAO_DIR="$BENCH_DIR/distill_runs/doubao"
OPUS_DIR="$BENCH_DIR/distill_runs/deepseek_opus"
R9_ROOT="$DOUBAO_DIR/runs/20260408_224133_doubao_headsft_r9"
R9_PARAMS="$DOUBAO_DIR/grid_params_two_stage_head_then_sft_round9.json"
TRAIN_DATA_GT="$PROJECT_ROOT/data/cmexam_dental_choice_train_nooverlap.jsonl"
TRAIN_DATA_HEAD="$R9_ROOT/artifacts/train_head_distill.jsonl"
VAL_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_val_nooverlap.jsonl"
TEST_DATA="$PROJECT_ROOT/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="$PROJECT_ROOT/Qwen2.5-7B-Instruct"

RESUME_LOG="$R9_ROOT/logs/resume_master.log"

{
echo "============================================="
echo "[RESUME] $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# --- DOUBAO R9: Complete remaining seeds ---
echo "[PHASE 1] Doubao R9 - completing remaining seeds"

"$PY" - "$R9_PARAMS" "$R9_ROOT" "$PROJECT_ROOT" "$BASE_MODEL" "$TRAIN_DATA_HEAD" "$TRAIN_DATA_GT" "$VAL_DATA" "$TEST_DATA" "$PY" <<'PY2'
import json, subprocess, sys, re
from pathlib import Path

params_path, run_root, project_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
base_model, train_head, train_gt = sys.argv[4], sys.argv[5], sys.argv[6]
val_data, test_data, py = sys.argv[7], sys.argv[8], sys.argv[9]

params = json.loads(params_path.read_text(encoding='utf-8'))

def get_test_acc(log_path):
    if not log_path.exists(): return None
    txt = log_path.read_text(encoding='utf-8', errors='ignore')
    m = re.findall(r"测试集准确率:\s*([0-9]+(?:\.[0-9]+)?)%", txt)
    return float(m[-1]) if m else None

for p in params:
    name = p['name']
    out_dir = run_root / 'outputs' / name
    stage1_dir = out_dir / 'stage1_head'
    stage2_dir = out_dir / 'stage2_sft'
    stage1_log = run_root / 'logs' / f'stage1_{name}.log'
    stage2_log = run_root / 'logs' / f'stage2_{name}.log'

    # Check if already completed
    acc = get_test_acc(stage2_log)
    if acc is not None:
        print(f"[SKIP] {name} already done: {acc}%", flush=True)
        continue

    # Check if stage2 training completed but test eval failed
    if stage2_log.exists():
        txt = stage2_log.read_text(encoding='utf-8', errors='ignore')
        if '蒸馏训练完成' in txt and '测试集准确率' not in txt:
            print(f"[RESUME-EVAL] {name} - rerun stage2 for test eval only (model exists)", flush=True)
            # Need to rerun entire stage2 since test eval isn't separable easily
            # Just rerun stage2
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if stage1 already completed
    stage1_done = stage1_log.exists() and stage1_dir.exists() and \
                  any(stage1_dir.glob("adapter_config.json")) or any(stage1_dir.glob("*/adapter_config.json"))
    
    if not stage1_done:
        print(f"[RUN][stage1-head] {name} seed={p['seed']}", flush=True)
        stage1_cmd = [
            py, str(Path(project_root) / 'train_dental_choice_head_distill_doubao.py'),
            '--model_name', base_model, '--data_path', train_head,
            '--val_path', val_data, '--test_path', test_data,
            '--output_dir', str(stage1_dir),
            '--num_epochs', str(p['num_epochs_stage1']),
            '--batch_size', str(p['batch_size']),
            '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
            '--learning_rate', str(p['learning_rate_stage1']),
            '--rank', str(p['rank']), '--lora_alpha', str(p['lora_alpha']),
            '--alpha', str(p['alpha_stage1']),
            '--default_distill_mask', '0', '--seed', str(p['seed']), '--deterministic',
        ]
        with stage1_log.open('w', encoding='utf-8') as lf:
            rc1 = subprocess.run(stage1_cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc1 != 0:
            print(f"[FAIL][stage1] {name} rc={rc1}", flush=True)
            continue
    else:
        print(f"[SKIP][stage1] {name} already done", flush=True)

    print(f"[RUN][stage2-sft] {name} seed={p['seed']}", flush=True)
    stage2_cmd = [
        py, str(Path(project_root) / 'train_dental_lora7_doubao.py'),
        '--model_name', base_model, '--data_path', train_gt,
        '--val_path', val_data, '--test_path', test_data,
        '--output_dir', str(stage2_dir),
        '--num_epochs', str(p['num_epochs_stage2']),
        '--batch_size', str(p['batch_size']),
        '--gradient_accumulation_steps', str(p['gradient_accumulation_steps']),
        '--learning_rate', str(p['learning_rate_stage2']),
        '--rank', str(p['rank']), '--lora_alpha', str(p['lora_alpha']),
        '--alpha', '0.0', '--default_distill_mask', '0',
        '--seed', str(p['seed']), '--deterministic',
        '--resume_from', str(stage1_dir),
    ]
    with stage2_log.open('w', encoding='utf-8') as lf:
        rc2 = subprocess.run(stage2_cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc2 != 0:
        print(f"[FAIL][stage2] {name} rc={rc2}", flush=True)
    else:
        acc = get_test_acc(stage2_log)
        print(f"[DONE] {name} acc={acc}%", flush=True)
PY2

# Collect Doubao R9 results
"$PY" - "$R9_PARAMS" "$R9_ROOT" <<'PY3'
import json, re, statistics as st, sys
from pathlib import Path
params = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
run_root = Path(sys.argv[2])
rows = []
for p in params:
    name, log_path = p['name'], run_root / 'logs' / f'stage2_{name}.log'
    acc = None
    if log_path.exists():
        m = re.findall(r"测试集准确率:\s*([0-9]+(?:\.[0-9]+)?)%", log_path.read_text(encoding='utf-8', errors='ignore'))
        if m: acc = float(m[-1])
    rows.append({'name': name, 'seed': p['seed'], 'accuracy': acc, 'status': 'ok' if acc else 'failed'})
ok = [r for r in rows if r['status'] == 'ok']
ok.sort(key=lambda x: x['accuracy'], reverse=True)
vals = [r['accuracy'] for r in ok]
summary = {'run_root': str(run_root), 'baseline': 77.11, 'previous_best': 80.72,
  'n': len(vals), 'best': ok[0] if ok else None,
  'mean': round(st.mean(vals), 2) if vals else None,
  'std': round(st.pstdev(vals), 2) if len(vals) > 1 else 0.0,
  'rows': rows}
(run_root / 'headsft_results_latest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n[DOUBAO R9 RESULTS]')
for r in rows:
    print(f"  {r['name']}: {r['accuracy']}%" if r['accuracy'] else f"  {r['name']}: FAILED")
if vals:
    print(f"  MEAN={round(st.mean(vals),2)}%  STD={round(st.pstdev(vals),2)}%  BEST={max(vals)}%")
PY3

echo ""
echo "============================================="
echo "[PHASE 2] DeepSeek Opus R1"
echo "============================================="
chmod +x "$OPUS_DIR/run_head_then_sft_round1.sh"
bash "$OPUS_DIR/run_head_then_sft_round1.sh"

echo ""
echo "============================================="
echo "[ALL DONE] $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# Final summary
echo ""
DOUBAO_R="$R9_ROOT/headsft_results_latest.json"
OPUS_R=$(find "$OPUS_DIR/runs" -name "opus_results_latest.json" 2>/dev/null | sort | tail -1)
echo "=== DOUBAO R9 ==="
cat "$DOUBAO_R" 2>/dev/null || echo "N/A"
echo ""
echo "=== DEEPSEEK OPUS R1 ==="
cat "$OPUS_R" 2>/dev/null || echo "N/A"

} 2>&1 | tee "$RESUME_LOG"
