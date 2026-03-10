"""
autoTestQwenCot.py —— 测试基座模型（无微调）用 CoT 提示的准确率
用于建立 baseline 对比：微调前 vs 微调后
"""
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== 配置 =====================
MODEL_PATH = "./Qwen2.5-7B-Instruct"
TESTSET_PATH = "./data/cmexam_dental_choice_test.jsonl"

SYSTEM_PROMPT_COT = (
    '你是一名专业的口腔医学专家。'
    '请按以下格式回答选择题：先给出"答案：X"（X为选项字母A-E），再给出"解析：..."说明理由。'
)


def extract_answer_char(text):
    for ch in text.strip().upper():
        if ch in "ABCDE":
            return ch
    return ""


def load_testset(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line.strip())
            q = d.get("Question", "")
            opts = d.get("Options", "")
            ans = d.get("Answer", "")
            if q and opts and ans:
                samples.append({"idx": idx + 1, "question": q, "options": opts, "answer": ans})
    print(f"加载 {len(samples)} 条测试样本")
    return samples


def run_test():
    print(f"加载基座模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )
    model.eval()
    print("模型加载完成！")

    samples = load_testset(TESTSET_PATH)
    if not samples:
        print("无有效测试样本")
        return

    correct = 0
    wrongs = []

    print("\n开始测试（CoT 提示，无微调）...")
    with torch.no_grad():
        for s in tqdm(samples, desc="测试进度"):
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT_COT}\n<|im_end|>\n"
                f"<|im_start|>user\n问题：{s['question']}\n选项：\n{s['options']}\n<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            gen = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
            pred = extract_answer_char(gen)

            if pred == s["answer"]:
                correct += 1
            else:
                wrongs.append({
                    "idx": s["idx"], "question": s["question"][:60],
                    "gt": s["answer"], "pred": pred, "gen": gen[:100],
                })

    total = len(samples)
    acc = 100 * correct / total if total else 0

    print("\n" + "=" * 60)
    print(f"基座模型 CoT 测试报告 ({MODEL_PATH})")
    print("=" * 60)
    print(f"总样本: {total}  正确: {correct}  错误: {len(wrongs)}")
    print(f"准确率: {acc:.2f}%")

    if wrongs:
        print(f"\n前 5 个错误样本:")
        for w in wrongs[:5]:
            print(f"  #{w['idx']} gt={w['gt']} pred={w['pred']} | {w['question']}...")


if __name__ == "__main__":
    run_test()
