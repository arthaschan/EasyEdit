#!/usr/bin/env python3
"""
用本地 Qwen2.5-14B-Instruct 模型对训练集进行推理，
生成教师标签文件 (teacher_train.jsonl)，格式与 API 教师一致。
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ANSWER_RE = re.compile(r"\b([A-E])\b")
SYSTEM_PROMPT = "你是一名专业的牙科医生，只需输出一个大写字母（A、B、C、D、E）作为答案，不要附带任何解释或空格。"


def build_question_text(item):
    q = str(item.get("Question") or item.get("question") or "").strip()
    options = item.get("Options") or item.get("options") or {}
    lines = [q]
    if isinstance(options, dict):
        for k in ["A", "B", "C", "D", "E"]:
            if k in options:
                lines.append(f"{k}. {str(options[k]).strip()}")
    else:
        opt_text = str(options).strip()
        if opt_text:
            lines.append(opt_text)
    lines.append("请只输出一个大写字母（A/B/C/D/E）。")
    return "\n".join(lines)


def extract_answer(text):
    if not text:
        return ""
    t = text.strip().upper()
    if len(t) == 1 and t in "ABCDE":
        return t
    m = ANSWER_RE.search(t)
    return m.group(1) if m else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./Qwen2.5-14B-Instruct")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Training JSONL file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output teacher_train.jsonl")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    rows = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} samples from {args.dataset}")

    # Load model
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    print("Model loaded.")

    total = 0
    parsed = 0
    correct = 0

    with open(args.output, "w", encoding="utf-8") as wf:
        for i, item in enumerate(rows):
            user_text = build_question_text(item)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=4, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            teacher_ans = extract_answer(gen_text)
            gt_ans = str(item.get("Answer") or item.get("answer") or "").strip().upper()

            total += 1
            if teacher_ans:
                parsed += 1
                if teacher_ans == gt_ans:
                    correct += 1

            out_row = dict(item)
            out_row["TeacherAnswer"] = teacher_ans
            out_row["TeacherRaw"] = gen_text.strip()
            wf.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(rows)}] parsed={parsed}/{total}, correct={correct}/{total}")

    acc = correct / total * 100 if total else 0
    mismatch = parsed - correct
    print(f"\n=== Teacher Label Summary ===")
    print(f"Total: {total}, Parsed: {parsed}, Correct: {correct}")
    print(f"Teacher Accuracy on train set: {acc:.2f}%")
    print(f"Teacher-GT Mismatch: {mismatch} ({mismatch/total*100:.2f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
