#!/usr/bin/env python3
"""
Rewrite HuaTuo and AutoGen questions to match CMExam style using DeepSeek API.
Also performs answer-balanced sampling to fix HuaTuo's B/C bias.

CMExam style characteristics:
- Short, concise questions (avg 29 chars)
- No "根据对话内容" phrasing
- Straightforward medical exam question format
- Options formatted as "A xxxx\nB xxxx\n..." (letter + space + text)
- Balanced answer distribution (~20% per option)
"""
import json
import os
import sys
import time
import random
import hashlib
import requests
from pathlib import Path
from collections import Counter

API_KEY = "sk-e025208e764648ce8da92d26596e246f"
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

REWRITE_SYSTEM = """你是一个医学考试题目编辑专家。你的任务是将给定的医学选择题改写为标准CMExam（中国医师资格考试）风格。

CMExam风格的特点：
1. 题干简洁精炼，通常20-40个字，直接提问
2. 不使用"根据对话内容"、"根据以上描述"等引导语
3. 临床病例题格式：先给出患者基本信息和主要症状，然后直接提问
4. 知识点题格式：直接问"...是"、"...错误的是"、"...不包括"
5. 选项格式：每个选项一行，字母后跟一个空格，然后是选项内容
6. 不以冒号结尾

你必须：
- 保持医学知识内容和正确答案不变
- 只改变表述风格和格式
- 压缩冗长的描述
- 移除"根据对话"等非考试风格表述
- 输出JSON格式：{"Question": "...", "Options": "A ...\nB ...\nC ...\nD ...\nE ...", "Answer": "X"}"""


def call_api(question: str, options: str, answer: str, max_retries: int = 3) -> dict | None:
    user_msg = f"""请将以下题目改写为CMExam标准风格，保持正确答案为{answer}不变。

原题：
问题：{question}
选项：
{options}
答案：{answer}

请输出改写后的JSON（只输出JSON，不要其他内容）："""

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": REWRITE_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 512,
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # Validate: must have Question, Options, Answer
            if "Question" not in result or "Options" not in result or "Answer" not in result:
                print(f"  [WARN] Missing fields, retry {attempt+1}")
                continue

            # Validate: answer must match
            rewritten_answer = result["Answer"].strip().upper()
            if rewritten_answer[0] != answer.strip().upper()[0]:
                print(f"  [WARN] Answer changed from {answer} to {rewritten_answer}, retry {attempt+1}")
                continue

            return result

        except Exception as e:
            print(f"  [ERROR] attempt {attempt+1}: {e}")
            time.sleep(2 * (attempt + 1))

    return None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/augment/merged_train.jsonl")
    p.add_argument("--output", default="data/augment/merged_train_cmexam_style.jsonl")
    p.add_argument("--request_interval", type=float, default=0.3)
    p.add_argument("--balance_answers", action="store_true", default=True,
                   help="Balance answer distribution for non-cmexam sources")
    p.add_argument("--max_per_answer", type=int, default=0,
                   help="Max samples per answer letter for non-cmexam (0=auto)")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    # Load all data
    all_data = []
    with open(args.input) as f:
        for line in f:
            all_data.append(json.loads(line.strip()))

    # Separate by source
    cmexam = [d for d in all_data if d.get("Source") == "cmexam_original"]
    huatuo = [d for d in all_data if d.get("Source") == "huatuo_converted"]
    autogen = [d for d in all_data if d.get("Source") == "deepseek_autogen"]
    print(f"Loaded: cmexam={len(cmexam)}, huatuo={len(huatuo)}, autogen={len(autogen)}")

    # Answer balance: downsample HuaTuo B/C heavy samples
    if args.balance_answers:
        def get_answer(d):
            ans = d.get("Answer", "").strip().upper()
            for ch in ans:
                if ch in "ABCDE":
                    return ch
            return ""

        # Target: match cmexam distribution (~20% each)
        # For HuaTuo: B=130, C=135, but A=20, D=39, E=22
        # Keep all A/D/E, downsample B/C to ~40 each to balance
        huatuo_by_answer = {}
        for d in huatuo:
            a = get_answer(d)
            huatuo_by_answer.setdefault(a, []).append(d)

        # Find median count (excluding the overrepresented ones)
        counts = {k: len(v) for k, v in huatuo_by_answer.items()}
        print(f"HuaTuo answer distribution: {counts}")

        # Target per answer: use max of minority classes
        minority_max = max(counts.get("A", 0), counts.get("D", 0), counts.get("E", 0))
        target = max(minority_max, 40)  # At least 40 per answer
        print(f"Target per answer: {target}")

        balanced_huatuo = []
        random.seed(42)
        for letter in "ABCDE":
            items = huatuo_by_answer.get(letter, [])
            if len(items) <= target:
                balanced_huatuo.extend(items)
            else:
                balanced_huatuo.extend(random.sample(items, target))

        print(f"HuaTuo after balance: {len(huatuo)} -> {len(balanced_huatuo)}")
        bal_counts = Counter(get_answer(d) for d in balanced_huatuo)
        print(f"  Balanced distribution: {dict(sorted(bal_counts.items()))}")
        huatuo = balanced_huatuo

    # Resume support
    done_keys = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                d = json.loads(line)
                key = d.get("Question", "")[:50]
                done_keys.add(key)
        print(f"Resume: {len(done_keys)} already done")

    # Process: rewrite huatuo + autogen, keep cmexam as-is
    to_rewrite = huatuo + autogen
    total = len(to_rewrite)
    print(f"\nWill rewrite {total} samples (huatuo={len(huatuo)}, autogen={len(autogen)})")
    print(f"CMExam passthrough: {len(cmexam)}")

    mode = "a" if args.resume else "w"
    rewritten_count = 0
    failed_count = 0

    with open(args.output, mode, encoding="utf-8") as out_f:
        # Write cmexam samples first (only if not resuming)
        if not args.resume:
            for d in cmexam:
                out_f.write(json.dumps(d, ensure_ascii=False) + "\n")
            print(f"Wrote {len(cmexam)} cmexam samples (passthrough)")

        # Rewrite huatuo + autogen
        for i, d in enumerate(to_rewrite):
            if d.get("Question", "")[:50] in done_keys:
                continue

            print(f"[{i+1}/{total}] Rewriting ({d.get('Source','')}): {d['Question'][:40]}...")
            result = call_api(d["Question"], d["Options"], d["Answer"])

            if result:
                # Preserve source and original info
                out_item = {
                    "Question": result["Question"],
                    "Options": result["Options"],
                    "Answer": result["Answer"].strip().upper()[0],
                    "Source": d.get("Source", "") + "_rewritten",
                    "OriginalQuestion": d["Question"],
                }
                out_f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                out_f.flush()
                rewritten_count += 1
            else:
                print(f"  [FAIL] Skipping sample")
                failed_count += 1

            time.sleep(args.request_interval)

            if (i + 1) % 50 == 0:
                print(f"[PROGRESS] {i+1}/{total} rewritten={rewritten_count} failed={failed_count}")

    print(f"\n[DONE] rewritten={rewritten_count} failed={failed_count}")
    print(f"Total samples in output: {len(cmexam) + rewritten_count}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
