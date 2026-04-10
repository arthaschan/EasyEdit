#!/usr/bin/env python3
"""
Ensemble inference for dental MCQ: load multiple LoRA models, run inference,
and use majority voting to determine the final answer.

Uses transformers + PEFT (no vLLM required). Loads base model once, then swaps
LoRA adapters for each model.

Usage:
  python ensemble_eval.py \
    --lora_dirs dir1 dir2 dir3 ... \
    --test_path ./data/cmexam_dental_choice_test.jsonl \
    --base_model ./Qwen2.5-7B-Instruct
"""
import argparse
import json
import os
import sys
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


SYSTEM_PROMPT = (
    "你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，"
    "不要附带任何解释或空格。"
)


def extract_answer_char(text: str) -> str:
    for ch in text.strip().upper():
        if ch in "ABCDE":
            return ch
    return ""


def load_test_data(path: str) -> list[dict]:
    """Load test set. Supports both conversations format and raw CMExam format."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            d = json.loads(line.strip())

            # Conversations format
            if "conversations" in d:
                user_content = ""
                correct_answer = ""
                for conv in d["conversations"]:
                    if conv["role"] == "user":
                        user_content = conv["content"]
                    elif conv["role"] == "assistant":
                        correct_answer = conv["content"]

                # Parse question and options from user content
                lines = user_content.strip().split("\n")
                question = ""
                options = {}
                is_option = False
                for ln in lines:
                    ln = ln.strip()
                    if not ln:
                        continue
                    if ln.startswith("问题："):
                        question = ln.replace("问题：", "").strip()
                    elif ln.startswith("选项："):
                        is_option = True
                    elif is_option and len(ln) >= 2 and ln[1] in ":：":
                        options[ln[0].upper()] = ln[2:].strip()

                answer = extract_answer_char(correct_answer)
                if question and options and answer:
                    samples.append({
                        "idx": idx,
                        "question": question,
                        "options": options,
                        "answer": answer,
                    })

            # Raw CMExam format
            elif "Question" in d and "Options" in d:
                question = d["Question"]
                raw_options = d["Options"]  # Keep raw string for prompt
                options = {}
                for opt_line in raw_options.split("\n"):
                    opt_line = opt_line.strip()
                    if len(opt_line) >= 2 and opt_line[1] == " ":
                        options[opt_line[0].upper()] = opt_line[2:].strip()
                answer = extract_answer_char(d.get("Answer", ""))
                if question and options and answer:
                    samples.append({
                        "idx": idx,
                        "question": question,
                        "options": options,
                        "raw_options": raw_options,
                        "answer": answer,
                    })

    print(f"Loaded {len(samples)} test samples from {path}")
    return samples


def build_prompt(question: str, options: dict, raw_options: str = "") -> str:
    # Use raw_options (from CMExam) when available to match training eval exactly
    if raw_options:
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n问题：{question}\n选项：\n{raw_options}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n问题：{question}\n选项：\n{options_text}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def eval_single_model(base_model_path: str, tokenizer, lora_path: str, samples: list[dict]) -> list[str]:
    """Evaluate a single LoRA model on all samples, return predicted answer chars.
    Loads a fresh base model each time to avoid adapter stacking issues."""
    print(f"\n{'='*60}")
    print(f"Loading LoRA adapter: {lora_path}")

    # Load fresh base model to avoid adapter stacking
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    predictions = []
    with torch.no_grad():
        for s in tqdm(samples, desc=f"  Inference ({os.path.basename(lora_path)})"):
            prompt = build_prompt(s["question"], s["options"], s.get("raw_options", ""))
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Only decode newly generated tokens
            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            predictions.append(extract_answer_char(text))

    correct = sum(1 for p, s in zip(predictions, samples) if p == s["answer"])
    acc = correct / len(samples) * 100
    print(f"  Accuracy: {correct}/{len(samples)} = {acc:.2f}%")

    # Clean up
    del model, base_model
    torch.cuda.empty_cache()

    return predictions


def majority_vote(all_predictions: list[list[str]], samples: list[dict]) -> dict:
    """Perform majority voting across all model predictions."""
    n_models = len(all_predictions)
    n_samples = len(samples)

    ensemble_answers = []
    per_sample_details = []

    for i in range(n_samples):
        votes = [all_predictions[m][i] for m in range(n_models)]
        counter = Counter(votes)
        winner = counter.most_common(1)[0][0]
        ensemble_answers.append(winner)

        per_sample_details.append({
            "idx": samples[i]["idx"],
            "question": samples[i]["question"][:80],
            "correct": samples[i]["answer"],
            "ensemble": winner,
            "votes": dict(counter),
            "is_correct": winner == samples[i]["answer"],
            "unanimous": len(counter) == 1,
        })

    correct = sum(1 for a, s in zip(ensemble_answers, samples) if a == s["answer"])
    acc = correct / n_samples * 100

    # Disagreement analysis
    n_unanimous = sum(1 for d in per_sample_details if d["unanimous"])
    n_disagreed = n_samples - n_unanimous

    return {
        "n_models": n_models,
        "n_samples": n_samples,
        "ensemble_accuracy": acc,
        "ensemble_correct": correct,
        "unanimous_count": n_unanimous,
        "disagreed_count": n_disagreed,
        "details": per_sample_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation for dental MCQ")
    parser.add_argument("--lora_dirs", nargs="+", required=True,
                        help="Paths to LoRA adapter directories")
    parser.add_argument("--base_model", default="./Qwen2.5-7B-Instruct",
                        help="Base model path")
    parser.add_argument("--test_path", default="./data/cmexam_dental_choice_test.jsonl",
                        help="Test set path (conversations or CMExam format)")
    parser.add_argument("--output", default="./ensemble_results.json",
                        help="Output JSON path for results")
    args = parser.parse_args()

    samples = load_test_data(args.test_path)
    if not samples:
        print("No test samples loaded, exiting.")
        return

    # Load tokenizer once
    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Evaluate each LoRA model (fresh base model loaded per adapter)
    all_predictions = []
    individual_accs = []
    for lora_dir in args.lora_dirs:
        preds = eval_single_model(args.base_model, tokenizer, lora_dir, samples)
        all_predictions.append(preds)
        correct = sum(1 for p, s in zip(preds, samples) if p == s["answer"])
        individual_accs.append(correct / len(samples) * 100)

    # Majority vote
    result = majority_vote(all_predictions, samples)

    # Summary
    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)
    print(f"Models: {len(args.lora_dirs)}")
    for i, (d, acc) in enumerate(zip(args.lora_dirs, individual_accs)):
        print(f"  [{i+1}] {os.path.basename(d)}: {acc:.2f}%")
    print(f"\nIndividual mean: {sum(individual_accs)/len(individual_accs):.2f}%")
    print(f"Ensemble (majority vote): {result['ensemble_accuracy']:.2f}%")
    print(f"Ensemble gain: {result['ensemble_accuracy'] - sum(individual_accs)/len(individual_accs):+.2f}%")
    print(f"Unanimous: {result['unanimous_count']}/{result['n_samples']}")
    print(f"Disagreed: {result['disagreed_count']}/{result['n_samples']}")

    # Save full results
    # Use parent dir names as keys to avoid collision (all dirs may end with /best)
    def unique_key(path):
        parts = path.rstrip('/').split('/')
        # Use last 2 levels, e.g. "opus_s11/stage2_sft/best" → "opus_s11__best"
        return '__'.join(parts[-3:-1]) if len(parts) >= 3 else os.path.basename(path)
    result["individual_accuracies"] = {
        unique_key(d): acc for d, acc in zip(args.lora_dirs, individual_accs)
    }
    result["lora_dirs"] = args.lora_dirs

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
