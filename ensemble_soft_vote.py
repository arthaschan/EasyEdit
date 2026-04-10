#!/usr/bin/env python3
"""
Soft-voting ensemble: use token probabilities from each model to weight votes.
Instead of just taking the predicted letter, average the probabilities for A/B/C/D/E
across all models and pick the highest average probability.
"""
import argparse
import json
import os
import sys
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


SYSTEM_PROMPT = (
    "你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，"
    "不要附带任何解释或空格。"
)

ANSWER_CHARS = list("ABCDE")


def load_test_data(path: str) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            d = json.loads(line.strip())
            if "Question" in d and "Options" in d:
                question = d["Question"]
                raw_options = d["Options"]
                options = {}
                for opt_line in raw_options.split("\n"):
                    opt_line = opt_line.strip()
                    if len(opt_line) >= 2 and opt_line[1] == " ":
                        options[opt_line[0].upper()] = opt_line[2:].strip()
                answer = ""
                for ch in d.get("Answer", "").strip().upper():
                    if ch in "ABCDE":
                        answer = ch
                        break
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


def build_prompt(question: str, raw_options: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n问题：{question}\n选项：\n{raw_options}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def get_answer_probs(model, tokenizer, samples: list[dict], answer_token_ids: dict) -> np.ndarray:
    """Get probability distribution over ABCDE for each sample.
    Returns: np.array of shape (n_samples, 5)
    """
    probs = np.zeros((len(samples), 5))

    with torch.no_grad():
        for i, s in enumerate(tqdm(samples, desc="  Getting probs")):
            prompt = build_prompt(s["question"], s["raw_options"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

            outputs = model(**inputs)
            # Get logits for the last token position (what the model would generate next)
            last_logits = outputs.logits[0, -1, :]  # (vocab_size,)

            # Extract logits for A, B, C, D, E tokens
            answer_logits = torch.tensor([last_logits[answer_token_ids[ch]].item() for ch in ANSWER_CHARS])
            # Softmax over just the answer tokens
            answer_probs = torch.softmax(answer_logits, dim=0).numpy()
            probs[i] = answer_probs

    return probs


def main():
    parser = argparse.ArgumentParser(description="Soft-voting ensemble for dental MCQ")
    parser.add_argument("--lora_dirs", nargs="+", required=True)
    parser.add_argument("--base_model", default="./Qwen2.5-7B-Instruct")
    parser.add_argument("--test_path", default="./data/cmexam_dental_choice_test.jsonl")
    parser.add_argument("--output", default="./ensemble_soft_results.json")
    args = parser.parse_args()

    samples = load_test_data(args.test_path)
    if not samples:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Get token IDs for A, B, C, D, E
    answer_token_ids = {}
    for ch in ANSWER_CHARS:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        answer_token_ids[ch] = ids[0]
    print(f"Answer token IDs: {answer_token_ids}")

    # Collect probabilities from each model
    all_probs = []  # list of (n_samples, 5) arrays
    individual_accs = []

    for lora_dir in args.lora_dirs:
        print(f"\n{'='*60}")
        print(f"Loading: {lora_dir}")

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, lora_dir)
        model.eval()

        probs = get_answer_probs(model, tokenizer, samples, answer_token_ids)
        all_probs.append(probs)

        # Hard prediction from this model alone
        hard_preds = [ANSWER_CHARS[np.argmax(probs[i])] for i in range(len(samples))]
        correct = sum(1 for p, s in zip(hard_preds, samples) if p == s["answer"])
        acc = correct / len(samples) * 100
        individual_accs.append(acc)
        print(f"  Hard accuracy: {correct}/{len(samples)} = {acc:.2f}%")

        del model, base_model
        torch.cuda.empty_cache()

    # ---- Ensemble methods ----
    n_models = len(all_probs)
    avg_probs = np.mean(all_probs, axis=0)  # (n_samples, 5)

    # Method 1: Uniform average (soft vote)
    soft_preds = [ANSWER_CHARS[np.argmax(avg_probs[i])] for i in range(len(samples))]
    soft_correct = sum(1 for p, s in zip(soft_preds, samples) if p == s["answer"])
    soft_acc = soft_correct / len(samples) * 100

    # Method 2: Weighted average (weight by individual accuracy)
    weights = np.array(individual_accs) / sum(individual_accs)
    weighted_probs = np.zeros_like(avg_probs)
    for m in range(n_models):
        weighted_probs += weights[m] * all_probs[m]
    weighted_preds = [ANSWER_CHARS[np.argmax(weighted_probs[i])] for i in range(len(samples))]
    weighted_correct = sum(1 for p, s in zip(weighted_preds, samples) if p == s["answer"])
    weighted_acc = weighted_correct / len(samples) * 100

    # Method 3: Hard majority vote (for comparison)
    from collections import Counter
    hard_ensemble = []
    for i in range(len(samples)):
        votes = [ANSWER_CHARS[np.argmax(all_probs[m][i])] for m in range(n_models)]
        counter = Counter(votes)
        hard_ensemble.append(counter.most_common(1)[0][0])
    hard_correct = sum(1 for p, s in zip(hard_ensemble, samples) if p == s["answer"])
    hard_acc = hard_correct / len(samples) * 100

    # Print results
    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS COMPARISON")
    print("=" * 60)
    for i, (d, acc) in enumerate(zip(args.lora_dirs, individual_accs)):
        print(f"  [{i+1}] {os.path.basename(d)}: {acc:.2f}%")
    print(f"\nIndividual mean: {np.mean(individual_accs):.2f}%")
    print(f"Hard majority vote:     {hard_correct}/{len(samples)} = {hard_acc:.2f}%")
    print(f"Soft vote (uniform):    {soft_correct}/{len(samples)} = {soft_acc:.2f}%")
    print(f"Soft vote (weighted):   {weighted_correct}/{len(samples)} = {weighted_acc:.2f}%")

    # Detailed per-sample comparison
    details = []
    for i, s in enumerate(samples):
        detail = {
            "idx": s["idx"],
            "correct": s["answer"],
            "hard_vote": hard_ensemble[i],
            "soft_vote": soft_preds[i],
            "weighted_vote": weighted_preds[i],
            "avg_probs": {ch: float(avg_probs[i][j]) for j, ch in enumerate(ANSWER_CHARS)},
        }
        details.append(detail)

    # Show where soft and hard voting differ
    print(f"\n=== Soft vs Hard differences ===")
    for d in details:
        if d["soft_vote"] != d["hard_vote"]:
            corr = "✓" if d["soft_vote"] == d["correct"] else "✗"
            print(f"  Q{d['idx']}: hard={d['hard_vote']}, soft={d['soft_vote']}, correct={d['correct']} {corr}")
            print(f"    probs: {d['avg_probs']}")

    result = {
        "n_models": n_models,
        "n_samples": len(samples),
        "individual_accuracies": {os.path.basename(d): acc for d, acc in zip(args.lora_dirs, individual_accs)},
        "hard_majority_vote_accuracy": hard_acc,
        "soft_vote_accuracy": soft_acc,
        "weighted_vote_accuracy": weighted_acc,
        "details": details,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
