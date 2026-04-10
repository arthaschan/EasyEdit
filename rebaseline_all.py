#!/usr/bin/env python3
"""
Re-baseline all historical best models with deterministic decoding (do_sample=False).
Tests both 'best' (val-selected) and 'last epoch' checkpoints.
"""
import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

BASE_MODEL = "./Qwen2.5-7B-Instruct"
TEST_PATH  = "./data/cmexam_dental_choice_test.jsonl"
SYSTEM_PROMPT = "你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。"

# All key models: (label, lora_path, reported_acc)
MODELS = [
    # --- DeepSeek Opus R1 (5 seeds) ---
    ("opus_s11_best",  "external_model_benchmark_20260326/distill_runs/deepseek_opus/runs/20260409_103609_deepseek_opus_r1/outputs/opus_s11/stage2_sft/best",  81.93),
    ("opus_s11_last",  "external_model_benchmark_20260326/distill_runs/deepseek_opus/runs/20260409_103609_deepseek_opus_r1/outputs/opus_s11/stage2_sft",       81.93),
    ("opus_s42_best",  "external_model_benchmark_20260326/distill_runs/deepseek_opus/runs/20260409_103609_deepseek_opus_r1/outputs/opus_s42/stage2_sft/best",  77.11),
    ("opus_s55_best",  "external_model_benchmark_20260326/distill_runs/deepseek_opus/runs/20260409_103609_deepseek_opus_r1/outputs/opus_s55/stage2_sft/best",  74.70),
    ("opus_s7_best",   "external_model_benchmark_20260326/distill_runs/deepseek_opus/runs/20260409_103609_deepseek_opus_r1/outputs/opus_s7/stage2_sft/best",   77.11),
    ("opus_s99_best",  "external_model_benchmark_20260326/distill_runs/deepseek_opus/runs/20260409_103609_deepseek_opus_r1/outputs/opus_s99/stage2_sft/best",  73.49),
    # --- Doubao R9 (5 seeds) ---
    ("doubao_r9_s42_best", "external_model_benchmark_20260326/distill_runs/doubao/runs/20260408_224133_doubao_headsft_r9/outputs/headsft9_s42/stage2_sft/best", 79.52),
    ("doubao_r9_s55_best", "external_model_benchmark_20260326/distill_runs/doubao/runs/20260408_224133_doubao_headsft_r9/outputs/headsft9_s55/stage2_sft/best", 79.52),
    ("doubao_r9_s11_best", "external_model_benchmark_20260326/distill_runs/doubao/runs/20260408_224133_doubao_headsft_r9/outputs/headsft9_s11/stage2_sft/best", 78.31),
    ("doubao_r9_s7_best",  "external_model_benchmark_20260326/distill_runs/doubao/runs/20260408_224133_doubao_headsft_r9/outputs/headsft9_s7/stage2_sft/best",  74.70),
    ("doubao_r9_s99_best", "external_model_benchmark_20260326/distill_runs/doubao/runs/20260408_224133_doubao_headsft_r9/outputs/headsft9_s99/stage2_sft/best", 74.70),
    # --- Doubao R7 best (historical "best") ---
    ("doubao_r7_s42_best", "external_model_benchmark_20260326/distill_runs/doubao/runs/20260408_194306_doubao_headsft_r7/outputs/headsft_s42/stage2_sft/best", 79.52),
    # --- DeepSeek V3 Round7 best ---
    ("dsv3_r7_a008_best",  "external_model_benchmark_20260326/distill_runs/deepseek_v3_selective/runs/20260407_190558_round7_anchor_lowalpha/outputs/a008_lr10/stage2_sft/best", 79.52),
    # --- Non-distilled baselines ---
    ("dental_7b_choice_lora", "dental_qwen2.5_7b_choice_lora", None),
]

def extract_answer(text):
    for ch in text.strip().upper():
        if ch in "ABCDE":
            return ch
    return ""

def load_test_data(path):
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            q, opts, ans = d.get("Question",""), d.get("Options",""), d.get("Answer","")
            if q and opts and ans:
                samples.append((q, opts, ans))
    return samples

def eval_model(base_model_path, tokenizer, lora_path, samples):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()

    correct = 0
    with torch.no_grad():
        for q, opts, ans in tqdm(samples, desc=f"  {os.path.basename(lora_path)}", leave=False):
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
                f"<|im_start|>user\n问题：{q}\n选项：\n{opts}\n<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            gen = tokenizer.decode(output_ids[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
            pred = extract_answer(gen)
            if pred == ans:
                correct += 1

    acc = 100.0 * correct / len(samples)
    del model, base
    torch.cuda.empty_cache()
    return correct, acc

def main():
    samples = load_test_data(TEST_PATH)
    print(f"Test samples: {len(samples)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    results = []
    for label, lora_path, reported in MODELS:
        if not os.path.isdir(lora_path):
            print(f"\n[SKIP] {label}: path not found ({lora_path})")
            results.append({"label": label, "status": "skip", "path": lora_path})
            continue

        print(f"\n[EVAL] {label} (reported: {reported}%)")
        try:
            correct, acc = eval_model(BASE_MODEL, tokenizer, lora_path, samples)
            delta = (acc - reported) if reported else None
            print(f"  => deterministic: {correct}/{len(samples)} = {acc:.2f}%", end="")
            if delta is not None:
                print(f"  (Δ = {delta:+.2f}%)")
            else:
                print()
            results.append({
                "label": label,
                "path": lora_path,
                "reported": reported,
                "deterministic": round(acc, 2),
                "correct": correct,
                "total": len(samples),
                "delta": round(delta, 2) if delta is not None else None,
                "status": "ok",
            })
        except Exception as e:
            print(f"  => ERROR: {e}")
            results.append({"label": label, "status": "error", "error": str(e), "path": lora_path})

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Model':<30s} {'Reported':>10s} {'Deterministic':>14s} {'Delta':>8s}")
    print("-" * 80)
    for r in results:
        if r["status"] == "ok":
            rep = f"{r['reported']:.2f}%" if r['reported'] else "N/A"
            det = f"{r['deterministic']:.2f}%"
            dlt = f"{r['delta']:+.2f}%" if r['delta'] is not None else "—"
            print(f"{r['label']:<30s} {rep:>10s} {det:>14s} {dlt:>8s}")
        else:
            print(f"{r['label']:<30s} {'SKIP/ERR':>10s}")

    # Save
    out = "rebaseline_results.json"
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
