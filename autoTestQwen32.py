import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 评估目标：优先加载 32B->7B 蒸馏后的 LoRA，若不存在则回退到 7B 基座
BASE_MODEL_PATH = "./Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./dental_qwen2.5_7b_choice_lora_distill_from32_best"
TESTSET_PATH = "./data/cmexam_dental_choice_test.jsonl"

MAX_NEW_TOKENS = 4

SYSTEM_PROMPT = "你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。"


def extract_answer_char(answer_text):
    for char in answer_text.strip().upper():
        if char in ["A", "B", "C", "D", "E"]:
            return char
    return ""


def load_jsonl_testset(file_path):
    test_samples = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"测试集文件不存在：{file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            try:
                data = json.loads(line.strip())
                question = data.get("Question", "")
                options_text = data.get("Options", "")
                correct_answer = data.get("Answer", "")

                if not question or not options_text or not correct_answer:
                    print(f"警告：第{line_idx + 1}行数据解析失败，跳过")
                    continue

                test_samples.append(
                    {
                        "line_idx": line_idx + 1,
                        "question": question,
                        "options_text": options_text,
                        "correct_answer": correct_answer,
                    }
                )
            except Exception as e:
                print(f"错误：第{line_idx + 1}行数据解析出错 - {str(e)}，跳过")
                continue

    print(f"成功加载{len(test_samples)}条有效测试样本（总行数：{len(lines)}）")
    return test_samples


def build_choice_prompt(question, options_text):
    return f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
问题：{question}
选项：
{options_text}
<|im_end|>
<|im_start|>assistant
"""


def run_qwen_test():
    print("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, use_fast=False)

    print("正在加载 7B 基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=False,
    )

    if os.path.isdir(ADAPTER_PATH):
        print(f"检测到 LoRA 适配器，加载: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        report_name = "Qwen2.5-7B (32B蒸馏LoRA)"
    else:
        print("未检测到 LoRA 适配器，回退到 7B 基座")
        model = base_model
        report_name = "Qwen2.5-7B 基座"

    model.eval()
    print("模型加载完成！")

    test_samples = load_jsonl_testset(TESTSET_PATH)
    if not test_samples:
        print("无有效测试样本，终止测试")
        return

    correct_count = 0
    wrong_samples = []

    print("\n开始批量测试...")
    with torch.no_grad():
        for sample in tqdm(test_samples, desc="测试进度"):
            prompt = build_choice_prompt(sample["question"], sample["options_text"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
                model_answer_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].size(1) :],
                    skip_special_tokens=True,
                ).strip()
                model_answer_char = extract_answer_char(model_answer_text)
            except Exception as e:
                print(f"\n错误：第{sample['line_idx']}行样本预测失败 - {str(e)}")
                model_answer_char = ""
                model_answer_text = ""

            sample["model_answer"] = model_answer_char
            sample["model_raw_answer"] = model_answer_text
            if model_answer_char == sample["correct_answer"]:
                correct_count += 1
            else:
                wrong_samples.append(sample)

    total_count = len(test_samples)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    print("\n" + "=" * 80)
    print(f"{report_name} 牙科选择题测试报告")
    print("=" * 80)
    print(f"总测试样本数：{total_count}")
    print(f"正确数：{correct_count}")
    print(f"错误数：{len(wrong_samples)}")
    print(f"正确率：{accuracy:.2f}%")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"检测到GPU：{torch.cuda.get_device_name(0)}，使用GPU推理")
    else:
        print("未检测到GPU，使用CPU推理（速度较慢）")

    run_qwen_test()
