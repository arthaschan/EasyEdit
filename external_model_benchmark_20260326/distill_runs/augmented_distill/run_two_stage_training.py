#!/usr/bin/env python3
"""Run two-stage training (choice-head distill → GT SFT) for a given teacher's data."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True, help="Grid params JSON")
    p.add_argument("--run_root", required=True)
    p.add_argument("--project_root", required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--train_head", required=True, help="Head distill dataset (with TeacherDist)")
    p.add_argument("--train_gt", required=True, help="GT SFT dataset")
    p.add_argument("--val_data", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--teacher_prefix", required=True, help="e.g. 'ds' or 'db'")
    p.add_argument("--py", default="python3")
    args = p.parse_args()

    params = json.loads(Path(args.params).read_text(encoding="utf-8"))
    run_root = Path(args.run_root)
    project_root = Path(args.project_root)

    for p_cfg in params:
        name = f"{args.teacher_prefix}_{p_cfg['name']}"
        out_dir = run_root / "outputs" / name
        stage1_dir = out_dir / "stage1_head"
        stage2_dir = out_dir / "stage2_sft"
        out_dir.mkdir(parents=True, exist_ok=True)

        stage1_log = run_root / "logs" / f"stage1_{name}.log"
        stage2_log = run_root / "logs" / f"stage2_{name}.log"

        # Stage 1: choice-head distillation
        stage1_cmd = [
            args.py, str(project_root / "train_dental_choice_head_distill_doubao.py"),
            "--model_name", args.base_model,
            "--data_path", args.train_head,
            "--val_path", args.val_data,
            "--test_path", args.test_data,
            "--output_dir", str(stage1_dir),
            "--num_epochs", str(p_cfg["num_epochs_stage1"]),
            "--batch_size", str(p_cfg["batch_size"]),
            "--gradient_accumulation_steps", str(p_cfg["gradient_accumulation_steps"]),
            "--learning_rate", str(p_cfg["learning_rate_stage1"]),
            "--rank", str(p_cfg["rank"]),
            "--lora_alpha", str(p_cfg["lora_alpha"]),
            "--alpha", str(p_cfg["alpha_stage1"]),
            "--default_distill_mask", "0",
            "--seed", str(p_cfg["seed"]),
            "--deterministic",
        ]

        print(f"\n[RUN] Stage1 head-distill: {name} seed={p_cfg['seed']}", flush=True)
        with stage1_log.open("w", encoding="utf-8") as lf:
            rc1 = subprocess.run(stage1_cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc1 != 0:
            print(f"[FAIL] Stage1 {name} rc={rc1}", flush=True)
            continue

        # Stage 2: GT SFT, resume from stage1 LoRA
        stage2_cmd = [
            args.py, str(project_root / "train_dental_lora7_deepseek.py"),
            "--model_name", args.base_model,
            "--data_path", args.train_gt,
            "--val_path", args.val_data,
            "--test_path", args.test_data,
            "--output_dir", str(stage2_dir),
            "--num_epochs", str(p_cfg["num_epochs_stage2"]),
            "--batch_size", str(p_cfg["batch_size"]),
            "--gradient_accumulation_steps", str(p_cfg["gradient_accumulation_steps"]),
            "--learning_rate", str(p_cfg["learning_rate_stage2"]),
            "--rank", str(p_cfg["rank"]),
            "--lora_alpha", str(p_cfg["lora_alpha"]),
            "--alpha", "0.0",
            "--default_distill_mask", "0",
            "--seed", str(p_cfg["seed"]),
            "--deterministic",
            "--resume_from", str(stage1_dir),
        ]

        print(f"[RUN] Stage2 GT-SFT: {name} seed={p_cfg['seed']}", flush=True)
        with stage2_log.open("w", encoding="utf-8") as lf:
            rc2 = subprocess.run(stage2_cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc2 != 0:
            print(f"[FAIL] Stage2 {name} rc={rc2}", flush=True)
        else:
            print(f"[DONE] {name}", flush=True)


if __name__ == "__main__":
    main()
