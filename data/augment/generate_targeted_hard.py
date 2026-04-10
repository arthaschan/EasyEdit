#!/usr/bin/env python3
"""
为14道硬题生成针对性训练数据，每道题生成3~5道相关题目。
使用DeepSeek API，聚焦相同知识点但不同提问角度。
"""
import json
import time
import requests
import random

API_KEY = "sk-e025208e764648ce8da92d26596e246f"
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

SYSTEM_PROMPT = """你是一个中国医学考试出题专家。你的任务是根据给定的知识点和考试题目，生成3~5道相关但不重复的选择题。

要求：
1. 题目风格必须是CMExam（中国医学考试）标准格式
2. 题干简洁精炼（20-40字），直接提问
3. 每题5个选项(A-E)，每个选项一行，字母后跟一个空格再跟选项内容
4. 生成的题目必须围绕同一核心知识点，但从不同角度考查
5. 确保正确答案分布均匀（不全是同一个字母）
6. 不要直接复制原题，要有变化
7. 答案必须是医学上正确的

输出格式：JSON数组，每个元素包含：
{"Question": "...", "Options": "A ...\nB ...\nC ...\nD ...\nE ...", "Answer": "X", "KnowledgePoint": "简短描述知识点"}

只输出JSON数组，不要其他任何内容。"""


def generate_related_questions(hard_q: dict, max_retries: int = 3) -> list:
    """为一道硬题生成3~5道相关训练题"""
    user_msg = f"""请根据以下考题涉及的核心知识点，生成4道相关但不同角度的选择题。

原题：
问题：{hard_q['Question']}
选项：
{hard_q['Options']}
正确答案：{hard_q['Answer']}

模型常见错误答案：{hard_q.get('model_preds', ['?'])[0]}

请生成4道相关题目，确保能帮助学习者掌握这个知识点。输出JSON数组："""

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
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            results = json.loads(content)
            if not isinstance(results, list):
                results = [results]

            # Validate each
            valid = []
            for r in results:
                if all(k in r for k in ["Question", "Options", "Answer"]):
                    ans = r["Answer"].strip().upper()
                    if ans and ans[0] in "ABCDE":
                        r["Answer"] = ans[0]
                        valid.append(r)

            if valid:
                return valid

        except Exception as e:
            print(f"  [ERROR] attempt {attempt+1}: {e}")
            time.sleep(2 * (attempt + 1))

    return []


def main():
    hard_questions = json.load(open("/tmp/hard_questions.json"))
    print(f"Loaded {len(hard_questions)} hard questions")

    all_generated = []
    for i, hq in enumerate(hard_questions):
        print(f"\n[{i+1}/{len(hard_questions)}] Generating for: {hq['Question'][:50]}...")
        print(f"  GT={hq['Answer']}, Model常错={hq.get('model_preds', ['?'])[0]}")

        generated = generate_related_questions(hq)
        print(f"  Generated {len(generated)} questions")

        for g in generated:
            g["Source"] = "targeted_hard"
            g["OriginalHardQuestion"] = hq["Question"][:50]
            all_generated.append(g)
            print(f"    -> Q: {g['Question'][:50]}... A={g['Answer']}")

        time.sleep(0.5)

    # Also include the original 14 hard questions themselves as training data
    # (direct injection - teach the model the correct answer)
    for hq in hard_questions:
        direct = {
            "Question": hq["Question"],
            "Options": hq["Options"],
            "Answer": hq["Answer"],
            "Source": "targeted_hard_direct",
        }
        all_generated.append(direct)

    output_path = "data/augment/targeted_hard_questions.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_generated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    from collections import Counter
    ans_dist = Counter(g["Answer"] for g in all_generated)
    src_dist = Counter(g["Source"] for g in all_generated)
    print(f"\n[DONE] Total generated: {len(all_generated)}")
    print(f"Answer distribution: {dict(sorted(ans_dist.items()))}")
    print(f"Source distribution: {dict(src_dist)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
