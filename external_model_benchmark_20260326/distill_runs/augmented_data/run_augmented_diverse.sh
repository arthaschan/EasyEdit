#!/usr/bin/env bash
# ==============================================================================
# Training on augmented dataset (1162 samples) with diverse hyperparameters
# for better ensemble diversity.
#
# Uses train_dental_lora7_deepseek.py as the base trainer (SFT mode, no teacher).
# ==============================================================================
set -euo pipefail

PROJECT_ROOT="/home/student/arthas/EasyEdit3"
PY="${PROJECT_ROOT}/.venv/bin/python"
TRAIN_SCRIPT="${PROJECT_ROOT}/train_dental_lora7_deepseek.py"

# ----- Data paths -----
TRAIN_DATA="${PROJECT_ROOT}/data/augment/merged_train.jsonl"
VAL_DATA="${PROJECT_ROOT}/data/cmexam_dental_choice_val.jsonl"
TEST_DATA="${PROJECT_ROOT}/data/cmexam_dental_choice_test.jsonl"
BASE_MODEL="${PROJECT_ROOT}/Qwen2.5-7B-Instruct"

# ----- Output root -----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="${PROJECT_ROOT}/external_model_benchmark_20260326/distill_runs/augmented_data/runs/${TIMESTAMP}_augmented_diverse"
mkdir -p "${RUN_ROOT}/logs"

PARAMS_JSON="${RUN_ROOT}/grid_params.json"

# ----- Diverse configs for ensemble -----
# Vary: seed, learning_rate, rank, lora_alpha, num_epochs
cat > "${PARAMS_JSON}" << 'JSON_END'
[
  {"name": "aug_r16_lr12_s42",  "seed": 42, "learning_rate": 0.00012, "rank": 16, "lora_alpha": 32, "num_epochs": 2, "batch_size": 2, "gradient_accumulation_steps": 4, "alpha": 0.0},
  {"name": "aug_r16_lr12_s11",  "seed": 11, "learning_rate": 0.00012, "rank": 16, "lora_alpha": 32, "num_epochs": 2, "batch_size": 2, "gradient_accumulation_steps": 4, "alpha": 0.0},
  {"name": "aug_r16_lr12_s55",  "seed": 55, "learning_rate": 0.00012, "rank": 16, "lora_alpha": 32, "num_epochs": 2, "batch_size": 2, "gradient_accumulation_steps": 4, "alpha": 0.0},
  {"name": "aug_r32_lr10_s42",  "seed": 42, "learning_rate": 0.0001,  "rank": 32, "lora_alpha": 64, "num_epochs": 3, "batch_size": 2, "gradient_accumulation_steps": 4, "alpha": 0.0},
  {"name": "aug_r32_lr10_s11",  "seed": 11, "learning_rate": 0.0001,  "rank": 32, "lora_alpha": 64, "num_epochs": 3, "batch_size": 2, "gradient_accumulation_steps": 4, "alpha": 0.0},
  {"name": "aug_r8_lr15_s42",   "seed": 42, "learning_rate": 0.00015, "rank": 8,  "lora_alpha": 16, "num_epochs": 2, "batch_size": 2, "gradient_accumulation_steps": 4, "alpha": 0.0},
  {"name": "aug_r8_lr15_s11",   "seed": 11, "learning_rate": 0.00015, "rank": 8,  "lora_alpha": 16, "num_epochs": 2, "batch_size": 2, "gradient_accumulation_steps": 4, "alpha": 0.0}
]
JSON_END

echo "=== Augmented-data diverse training ==="
echo "Train data: ${TRAIN_DATA} ($(wc -l < "${TRAIN_DATA}") samples)"
echo "Configs:    ${PARAMS_JSON}"
echo "Run root:   ${RUN_ROOT}"
echo ""

# ----- Train each config -----
"$PY" - "${PARAMS_JSON}" "${RUN_ROOT}" "${TRAIN_SCRIPT}" "${TRAIN_DATA}" "${VAL_DATA}" "${TEST_DATA}" "${BASE_MODEL}" << 'PY_TRAIN'
import json, subprocess, sys, os
from pathlib import Path

params     = json.loads(Path(sys.argv[1]).read_text())
run_root   = Path(sys.argv[2])
script     = sys.argv[3]
train_data = sys.argv[4]
val_data   = sys.argv[5]
test_data  = sys.argv[6]
base_model = sys.argv[7]

py = sys.executable

for p in params:
    name = p["name"]
    out_dir = str(run_root / "outputs" / name)
    log_path = run_root / "logs" / f"{name}.log"

    if (Path(out_dir) / "training_state.json").exists():
        print(f"[SKIP] {name}: already has training_state.json")
        continue

    cmd = [
        py, script,
        "--model_name",   base_model,
        "--data_path",    train_data,
        "--val_path",     val_data,
        "--test_path",    test_data,
        "--output_dir",   out_dir,
        "--num_epochs",   str(p["num_epochs"]),
        "--batch_size",   str(p["batch_size"]),
        "--gradient_accumulation_steps", str(p["gradient_accumulation_steps"]),
        "--learning_rate", str(p["learning_rate"]),
        "--rank",         str(p["rank"]),
        "--lora_alpha",   str(p["lora_alpha"]),
        "--alpha",        str(p.get("alpha", 0.0)),
        "--seed",         str(p["seed"]),
        "--deterministic",
        "--use_teacher_dist",  # skip loading teacher model; with alpha=0 this is pure SFT
    ]

    print(f"\n[RUN] {name}  rank={p['rank']} lr={p['learning_rate']} seed={p['seed']} epochs={p['num_epochs']}", flush=True)
    with open(log_path, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"  [FAIL] {name} rc={rc}")
    else:
        print(f"  [DONE] {name}")

PY_TRAIN

# ----- Collect results -----
echo ""
echo "=== Collecting results ==="

"$PY" - "${PARAMS_JSON}" "${RUN_ROOT}" << 'PY_COLLECT'
import json, re, statistics as st, sys
from pathlib import Path

params   = json.loads(Path(sys.argv[1]).read_text())
run_root = Path(sys.argv[2])

rows = []
for p in params:
    name = p["name"]
    log_path = run_root / "logs" / f"{name}.log"
    acc = None
    if log_path.exists():
        txt = log_path.read_text(errors="ignore")
        m = re.findall(r"测试集准确率:\s*([0-9]+(?:\.[0-9]+)?)%", txt)
        if m:
            acc = float(m[-1])
    rows.append({"name": name, "seed": p["seed"], "rank": p["rank"], "lr": p["learning_rate"], "accuracy": acc})

ok = [r for r in rows if r["accuracy"] is not None]
ok.sort(key=lambda x: x["accuracy"], reverse=True)
vals = [r["accuracy"] for r in ok]

summary = {
    "n": len(vals),
    "baseline": 77.11,
    "best": ok[0] if ok else None,
    "mean": round(st.mean(vals), 2) if vals else None,
    "std": round(st.pstdev(vals), 2) if len(vals) > 1 else 0.0,
    "all": rows,
}

out_path = run_root / "augmented_results.json"
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nResults saved to {out_path}")

print(f"\n{'Name':<30s} {'Rank':>4s} {'LR':>8s} {'Seed':>4s} {'Acc':>8s}")
print("-" * 60)
for r in rows:
    acc_str = f"{r['accuracy']:.2f}%" if r['accuracy'] else "FAIL"
    print(f"{r['name']:<30s} {r['rank']:>4d} {r['lr']:>8.5f} {r['seed']:>4d} {acc_str:>8s}")

if vals:
    print(f"\nMEAN={st.mean(vals):.2f}%  STD={st.pstdev(vals):.2f}%  BEST={max(vals):.2f}%")

PY_COLLECT

echo ""
echo "=== Done ==="
