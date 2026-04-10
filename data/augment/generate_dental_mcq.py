#!/usr/bin/env python3
"""
Step 1: Filter dental QA from HuaTuo and convert to MCQ format using DeepSeek API.
Step 2: DeepSeek auto-generate dental MCQ from scratch (knowledge-based).

Both steps produce CMExam-compatible JSONL format.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ─── API Config ───
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-e025208e764648ce8da92d26596e246f"
DEEPSEEK_MODEL = "deepseek-chat"

DENTAL_KEYWORDS = [
    '牙', '口腔', '龋', '牙龈', '牙周', '拔牙', '种植牙', '正畸', '烤瓷', '义齿',
    '根管', '牙髓', '牙冠', '智齿', '磨牙', '切牙', '犬牙', '乳牙', '恒牙',
    '牙齿', '牙痛', '牙疼', '补牙', '洗牙', '牙套', '牙列', '咬合', '颌',
    '唇裂', '腮腺', '下颌', '上颌', '颞颌', '口臭', '口疮', '口角',
    '龈', '髓', '釉质', '牙本质', '牙骨质', '牙槽', '颊', '腭',
]

DENTAL_TOPICS = [
    "口腔解剖生理学", "口腔组织病理学", "龋病", "牙髓病", "根尖周病",
    "牙周病", "口腔黏膜病", "口腔颌面外科", "口腔修复学", "口腔正畸学",
    "儿童口腔医学", "口腔预防医学", "牙体牙髓治疗", "口腔种植学",
    "口腔影像诊断", "口腔急症处理", "口腔材料学", "口腔麻醉学",
]


def call_deepseek(system_prompt: str, user_prompt: str, temperature: float = 0.7,
                  max_tokens: int = 1500, retries: int = 3) -> str | None:
    """Call DeepSeek API with retries."""
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


# ─── Task 1: HuaTuo QA → MCQ ───

HUATUO_TO_MCQ_SYSTEM = """你是一名口腔医学考试出题专家。给你一段口腔/牙科问答对话，请基于对话内容生成一道单项选择题。

输出格式（严格JSON）：
{
  "Question": "题目文本",
  "Options": "A 选项A\\nB 选项B\\nC 选项C\\nD 选项D\\nE 选项E",
  "Answer": "正确答案字母(A-E)",
  "Explanation": "解析"
}

要求：
1. 题目必须与口腔/牙科医学相关
2. 必须有5个选项(A-E)，只有1个正确答案
3. 干扰项必须合理且有迷惑性，不能太离谱
4. 难度适中，接近执业医师考试水平
5. 只输出JSON，不要输出其他内容"""


def convert_huatuo_to_mcq(qa_text: str) -> dict | None:
    """Convert a HuaTuo QA pair to MCQ format via DeepSeek."""
    resp = call_deepseek(HUATUO_TO_MCQ_SYSTEM, qa_text, temperature=0.5)
    if not resp:
        return None
    # Extract JSON from response
    try:
        # Try direct parse
        obj = json.loads(resp)
    except json.JSONDecodeError:
        # Try extracting JSON block
        m = re.search(r'\{[^{}]*"Question"[^{}]*\}', resp, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return None

    # Validate
    required = ["Question", "Options", "Answer", "Explanation"]
    if not all(k in obj for k in required):
        return None
    ans = obj["Answer"].strip().upper()
    if ans not in "ABCDE":
        return None
    obj["Answer"] = ans
    obj["Clinical Department"] = "口腔科"
    obj["Medical Discipline"] = "口腔医学"
    obj["Source"] = "huatuo_converted"
    return obj


# ─── Task 2: DeepSeek Auto-Generate ───

AUTOGEN_SYSTEM = """你是一名口腔医学考试出题专家。请根据给定的口腔医学主题，生成3道高质量的单项选择题。

输出格式（严格JSON数组）：
[
  {
    "Question": "题目文本",
    "Options": "A 选项A\\nB 选项B\\nC 选项C\\nD 选项D\\nE 选项E",
    "Answer": "正确答案字母(A-E)",
    "Explanation": "解析"
  },
  ...
]

要求：
1. 每道题必须与给定主题强相关
2. 必须有5个选项(A-E)，只有1个正确答案
3. 干扰项必须合理且有迷惑性
4. 3道题的难度分布：1道基础、1道中等、1道较难
5. 3道题不能考查相同的知识点
6. 题目风格参照中国口腔执业医师考试
7. 只输出JSON数组，不要输出其他内容"""


def autogen_mcq(topic: str, batch_id: int) -> list[dict]:
    """Auto-generate MCQ questions for a dental topic."""
    prompt = f"主题：{topic}\n请生成3道口腔医学单项选择题（第{batch_id}批，请出新题，不要重复常见题）。"
    resp = call_deepseek(AUTOGEN_SYSTEM, prompt, temperature=0.8, max_tokens=2000)
    if not resp:
        return []
    try:
        items = json.loads(resp)
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', resp, re.DOTALL)
        if not m:
            return []
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            return []

    results = []
    if not isinstance(items, list):
        return []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        if not all(k in obj for k in ["Question", "Options", "Answer"]):
            continue
        ans = str(obj["Answer"]).strip().upper()
        if ans not in "ABCDE":
            continue
        obj["Answer"] = ans
        obj["Clinical Department"] = "口腔科"
        obj["Medical Discipline"] = "口腔医学"
        obj["Source"] = "deepseek_autogen"
        obj.setdefault("Explanation", "")
        results.append(obj)
    return results


def task1_huatuo_convert(input_path: str, output_path: str, max_items: int = 500):
    """Filter dental QA from HuaTuo and convert to MCQ via DeepSeek API."""
    print(f"[Task1] Loading HuaTuo data from {input_path}")
    dental_items = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            text = json.dumps(d, ensure_ascii=False)
            if any(kw in text for kw in DENTAL_KEYWORDS):
                # Extract QA text
                convs = d.get("conversations", [])
                if len(convs) >= 2:
                    qa = f"患者问：{convs[0]['content']}\n医生答：{convs[1]['content']}"
                    dental_items.append(qa)

    print(f"[Task1] Found {len(dental_items)} dental QA items")
    random.seed(42)
    random.shuffle(dental_items)
    dental_items = dental_items[:max_items]
    print(f"[Task1] Processing {len(dental_items)} items via DeepSeek API")

    results = []
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, qa in enumerate(dental_items):
        mcq = convert_huatuo_to_mcq(qa)
        if mcq:
            results.append(mcq)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(dental_items)}] converted={len(results)}")
        # Rate limiting
        time.sleep(0.3)

    with open(output_path, "w", encoding="utf-8") as wf:
        for r in results:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Task1] Done: {len(results)} MCQs saved to {output_path}")
    return results


def task2_autogen(output_path: str, batches_per_topic: int = 4):
    """Auto-generate dental MCQ questions using DeepSeek."""
    print(f"[Task2] Auto-generating MCQ for {len(DENTAL_TOPICS)} topics, {batches_per_topic} batches each")

    results = []
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ti, topic in enumerate(DENTAL_TOPICS):
        for batch_id in range(1, batches_per_topic + 1):
            items = autogen_mcq(topic, batch_id)
            results.extend(items)
            time.sleep(0.3)
        if (ti + 1) % 3 == 0:
            print(f"  [{ti+1}/{len(DENTAL_TOPICS)} topics] generated={len(results)}")

    with open(output_path, "w", encoding="utf-8") as wf:
        for r in results:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Task2] Done: {len(results)} MCQs saved to {output_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["huatuo", "autogen", "both"], default="both")
    p.add_argument("--huatuo_input", default="/home/student/arthas/EasyEdit3/data/huatuo_dental_qa.jsonl")
    p.add_argument("--huatuo_output", default="/home/student/arthas/EasyEdit3/data/augment/huatuo_dental_mcq.jsonl")
    p.add_argument("--autogen_output", default="/home/student/arthas/EasyEdit3/data/augment/deepseek_autogen_mcq.jsonl")
    p.add_argument("--huatuo_max", type=int, default=500,
                   help="Max HuaTuo items to convert (API calls)")
    p.add_argument("--autogen_batches", type=int, default=4,
                   help="Batches per topic (each batch = 3 questions)")
    args = p.parse_args()

    if args.task in ("huatuo", "both"):
        task1_huatuo_convert(args.huatuo_input, args.huatuo_output, args.huatuo_max)

    if args.task in ("autogen", "both"):
        task2_autogen(args.autogen_output, args.autogen_batches)
