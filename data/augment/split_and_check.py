#!/usr/bin/env python3
"""
Step 3: Split augmented MCQ datasets into train/val/test and check correctness via DeepSeek API.

Also merges augmented data with original CMExam data and creates final training sets.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

# ─── API Config ───
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-e025208e764648ce8da92d26596e246f"
DEEPSEEK_MODEL = "deepseek-chat"

CORRECTNESS_SYSTEM = """你是一名口腔医学考试审题专家。请判断以下口腔医学选择题的答案是否正确。

给你题目、选项和标注答案，请分析：
1. 题目是否合理（没有歧义、没有事实错误）
2. 标注的答案是否正确
3. 是否有其他选项也可能正确

输出格式（严格JSON）：
{
  "is_correct": true或false,
  "confidence": 0.0到1.0之间的数字,
  "reason": "简要说明",
  "suggested_answer": "你认为正确的答案字母(A-E)"
}

只输出JSON，不要输出其他内容。"""


def call_deepseek(system_prompt: str, user_prompt: str, temperature: float = 0.1,
                  max_tokens: int = 500, retries: int = 3) -> str | None:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[API ERROR] {e}", file=sys.stderr)
                return None


def check_correctness(item: dict) -> dict | None:
    """Use DeepSeek to check if an MCQ answer is correct."""
    prompt = f"""题目：{item['Question']}
选项：
{item['Options']}
标注答案：{item['Answer']}"""
    if item.get("Explanation"):
        prompt += f"\n解析：{item['Explanation']}"

    resp = call_deepseek(CORRECTNESS_SYSTEM, prompt)
    if not resp:
        return None
    try:
        obj = json.loads(resp)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*"is_correct"[^{}]*\}', resp, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    return obj


def split_dataset(items: list[dict], train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split items into train/val/test."""
    random.seed(seed)
    shuffled = items.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def save_jsonl(items: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_correctness_check(items: list[dict], sample_size: int = 50, source_name: str = ""):
    """Sample and check correctness of items."""
    if len(items) <= sample_size:
        sample = items
    else:
        random.seed(42)
        sample = random.sample(items, sample_size)

    print(f"[Correctness] Checking {len(sample)} items from {source_name}...")
    results = {"correct": 0, "incorrect": 0, "disagree_answer": 0, "check_failed": 0}
    details = []

    for i, item in enumerate(sample):
        check = check_correctness(item)
        if check is None:
            results["check_failed"] += 1
            continue

        is_correct = check.get("is_correct", False)
        suggested = check.get("suggested_answer", item["Answer"])
        confidence = check.get("confidence", 0)

        if is_correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1

        if suggested != item["Answer"]:
            results["disagree_answer"] += 1

        details.append({
            "question": item["Question"][:50],
            "labeled_answer": item["Answer"],
            "judge_correct": is_correct,
            "judge_answer": suggested,
            "confidence": confidence,
            "reason": check.get("reason", ""),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(sample)}] correct={results['correct']} incorrect={results['incorrect']}")
        time.sleep(0.3)

    total_checked = results["correct"] + results["incorrect"]
    accuracy = results["correct"] / total_checked if total_checked > 0 else 0

    report = {
        "source": source_name,
        "total_items": len(items),
        "sample_size": len(sample),
        "results": results,
        "accuracy": f"{accuracy:.1%}",
        "details": details,
    }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--huatuo_mcq", default="/home/student/arthas/EasyEdit3/data/augment/huatuo_dental_mcq.jsonl")
    p.add_argument("--autogen_mcq", default="/home/student/arthas/EasyEdit3/data/augment/deepseek_autogen_mcq.jsonl")
    p.add_argument("--original_train", default="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_train_nooverlap.jsonl")
    p.add_argument("--original_val", default="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_val.jsonl")
    p.add_argument("--original_test", default="/home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl")
    p.add_argument("--output_dir", default="/home/student/arthas/EasyEdit3/data/augment")
    p.add_argument("--check_correctness", action="store_true", help="Run DeepSeek correctness check")
    p.add_argument("--check_sample_size", type=int, default=50)
    p.add_argument("--merge", action="store_true", help="Merge augmented data with original training set")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load augmented data
    huatuo_items = load_jsonl(args.huatuo_mcq) if os.path.exists(args.huatuo_mcq) else []
    autogen_items = load_jsonl(args.autogen_mcq) if os.path.exists(args.autogen_mcq) else []

    print(f"HuaTuo MCQ: {len(huatuo_items)} items")
    print(f"AutoGen MCQ: {len(autogen_items)} items")

    # Split each source independently
    for name, items in [("huatuo", huatuo_items), ("autogen", autogen_items)]:
        if not items:
            continue
        train, val, test = split_dataset(items)
        save_jsonl(train, str(out / f"{name}_train.jsonl"))
        save_jsonl(val, str(out / f"{name}_val.jsonl"))
        save_jsonl(test, str(out / f"{name}_test.jsonl"))
        print(f"[{name}] Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Correctness check
    if args.check_correctness:
        reports = []
        for name, items in [("huatuo_mcq", huatuo_items), ("autogen_mcq", autogen_items)]:
            if not items:
                continue
            report = run_correctness_check(items, args.check_sample_size, name)
            reports.append(report)
            print(f"\n[{name}] Accuracy: {report['accuracy']} "
                  f"(correct={report['results']['correct']}, "
                  f"incorrect={report['results']['incorrect']}, "
                  f"disagree={report['results']['disagree_answer']})")

        with open(str(out / "correctness_report.json"), "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        print(f"\nFull report saved to {out / 'correctness_report.json'}")

    # Merge with original training data
    if args.merge:
        orig_train = load_jsonl(args.original_train) if os.path.exists(args.original_train) else []
        huatuo_train = load_jsonl(str(out / "huatuo_train.jsonl")) if os.path.exists(out / "huatuo_train.jsonl") else []
        autogen_train = load_jsonl(str(out / "autogen_train.jsonl")) if os.path.exists(out / "autogen_train.jsonl") else []

        # Mark source for each item
        for item in orig_train:
            item.setdefault("Source", "cmexam_original")
        for item in huatuo_train:
            item.setdefault("Source", "huatuo_converted")
        for item in autogen_train:
            item.setdefault("Source", "deepseek_autogen")

        merged = orig_train + huatuo_train + autogen_train
        random.seed(42)
        random.shuffle(merged)
        save_jsonl(merged, str(out / "merged_train.jsonl"))
        print(f"\n[Merge] orig={len(orig_train)} + huatuo={len(huatuo_train)} + autogen={len(autogen_train)} = {len(merged)}")
        print(f"Saved to {out / 'merged_train.jsonl'}")


if __name__ == "__main__":
    main()
