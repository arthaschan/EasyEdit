import csv
import os
import re
import subprocess
import time
from datetime import datetime

PYTHON_BIN = os.environ.get("EXP_PYTHON", "python")
TRAIN_SCRIPT = "train_dental_lora7.py"
BASE_OUTPUT_ROOT = "auto_experiments"

# Minimal plan: only tune alpha around best lr=1e-4, temperature=1.5
EXPERIMENTS = [
    {
        "name": "lr1e4_a03_t15",
        "num_epochs": 4,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "rank": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "alpha": 0.3,
        "temperature": 1.5,
    },
    {
        "name": "lr1e4_a05_t15",
        "num_epochs": 4,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "rank": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "alpha": 0.5,
        "temperature": 1.5,
    },
    {
        "name": "lr1e4_a06_t15",
        "num_epochs": 4,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "rank": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "alpha": 0.6,
        "temperature": 1.5,
    },
    {
        "name": "lr1e4_a07_t15",
        "num_epochs": 4,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "rank": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "alpha": 0.7,
        "temperature": 1.5,
    },
]


def parse_metric(pattern: str, text: str):
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def run_one_experiment(exp: dict, run_root: str):
    exp_output = os.path.join(run_root, exp["name"])
    os.makedirs(exp_output, exist_ok=True)

    cmd = [
        PYTHON_BIN,
        TRAIN_SCRIPT,
        "--num_epochs", str(exp["num_epochs"]),
        "--batch_size", str(exp["batch_size"]),
        "--gradient_accumulation_steps", str(exp["gradient_accumulation_steps"]),
        "--rank", str(exp["rank"]),
        "--lora_alpha", str(exp["lora_alpha"]),
        "--learning_rate", str(exp["learning_rate"]),
        "--temperature", str(exp["temperature"]),
        "--alpha", str(exp["alpha"]),
        "--augment",
        "--val_path", "./data/cmexam_dental_choice_val.jsonl",
        "--test_path", "./data/cmexam_dental_choice_test.jsonl",
        "--output_dir", exp_output,
    ]

    print("\n" + "=" * 80)
    print(f"[START] {exp['name']}")
    print("Command:", " ".join(cmd))
    print("=" * 80)

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    log_path = os.path.join(exp_output, "train.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n\n[STDERR]\n")
            f.write(proc.stderr)

    merged_text = proc.stdout + "\n" + proc.stderr
    test_acc = parse_metric(r"测试集准确率:\s*([0-9.]+)", merged_text)
    val_acc = parse_metric(r"第\s*\d+\s*轮验证准确率:\s*([0-9.]+)", merged_text)

    return {
        "name": exp["name"],
        "return_code": proc.returncode,
        "elapsed_min": round(elapsed / 60.0, 2),
        "test_acc": test_acc,
        "val_acc": val_acc,
        "output_dir": exp_output,
        "log_file": log_path,
        "num_epochs": exp["num_epochs"],
        "batch_size": exp["batch_size"],
        "gradient_accumulation_steps": exp["gradient_accumulation_steps"],
        "rank": exp["rank"],
        "lora_alpha": exp["lora_alpha"],
        "learning_rate": exp["learning_rate"],
        "alpha": exp["alpha"],
        "temperature": exp["temperature"],
    }


def write_reports(results: list, run_root: str):
    csv_path = os.path.join(run_root, "summary.csv")
    md_path = os.path.join(run_root, "summary.md")

    fieldnames = [
        "name", "return_code", "elapsed_min", "test_acc", "val_acc",
        "num_epochs", "batch_size", "gradient_accumulation_steps",
        "rank", "lora_alpha", "learning_rate", "alpha", "temperature",
        "output_dir", "log_file",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    lines = []
    lines.append("| exp | rc | min | test_acc | val_acc | lr | alpha | temp | rank | lora_alpha | output |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['return_code']} | {r['elapsed_min']} | "
            f"{r['test_acc']} | {r['val_acc']} | {r['learning_rate']} | "
            f"{r['alpha']} | {r['temperature']} | {r['rank']} | {r['lora_alpha']} | {r['output_dir']} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[REPORT]")
    print("CSV:", csv_path)
    print("MD:", md_path)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(BASE_OUTPUT_ROOT, f"run_minimal_{ts}")
    os.makedirs(run_root, exist_ok=True)

    results = []
    for exp in EXPERIMENTS:
        res = run_one_experiment(exp, run_root)
        results.append(res)
        print(f"[DONE] {exp['name']} rc={res['return_code']} test_acc={res['test_acc']} elapsed={res['elapsed_min']}min")

    results_sorted = sorted(results, key=lambda x: (x["test_acc"] is None, -(x["test_acc"] or -1.0)))
    write_reports(results_sorted, run_root)
    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
