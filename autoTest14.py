from vllm import LLM, SamplingParams
import torch
import json
import os
from tqdm import tqdm

# ===================== 1. 基础配置 =====================
MODEL_PATH = "./dental_qwen2.5_14b_choice_lora_distill"
TOKENIZER_PATH = "./Qwen2.5-14B-Instruct"
GPU_MEMORY_UTILIZATION = 0.9
DTYPE = torch.bfloat16

SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=10,
    stop=["<|endoftext|>", "</s>"]
)

# ===================== 2. 工具函数 =====================
def extract_question_options(user_content):
    lines = user_content.strip().split("\n")
    question = ""
    options = {}
    is_option = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("请回答以下选择题"):
            continue
        elif line.startswith("问题："):
            question = line.replace("问题：", "").strip()
        elif line.startswith("选项："):
            is_option = True
        elif is_option and len(line) >= 2 and line[1] in [":", "："]:
            option_key = line[0].upper()
            option_value = line[2:].strip()
            options[option_key] = option_value
    return question, options


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
                user_content = ""
                correct_answer = ""
                for conv in data.get("conversations", []):
                    if conv.get("role") == "user":
                        user_content = conv.get("content", "")
                    elif conv.get("role") == "assistant":
                        correct_answer = conv.get("content", "")
                question, options = extract_question_options(user_content)
                correct_answer_char = extract_answer_char(correct_answer)
                if not question or not options or not correct_answer_char:
                    print(f"警告：第{line_idx+1}行数据解析失败，跳过")
                    continue
                test_samples.append({
                    "line_idx": line_idx + 1,
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer_char,
                })
            except Exception as e:
                print(f"错误：第{line_idx+1}行数据解析出错 - {str(e)}，跳过")
                continue
    print(f"成功加载{len(test_samples)}条有效测试样本（总行数：{len(lines)}）")
    return test_samples


# ===================== 3. 批量测试 =====================
def run_testset(llm, testset_path):
    test_samples = load_jsonl_testset(testset_path)
    if not test_samples:
        print("无有效测试样本，终止测试")
        return

    def build_test_prompt(question, options):
        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
        return f"""<|im_start|>system
你是一名专业的牙科医生，仅需输出正确选项的字母（如A、B、C、D、E），不要输出其他内容，无需额外解释。
<|im_end|>
<|im_start|>user
问题：{question}
选项：
{options_text}
<|im_end|>
<|im_start|>assistant
"""

    correct_count = 0
    wrong_samples = []

    print("\n开始批量测试...")
    for sample in tqdm(test_samples, desc="测试进度"):
        prompt = build_test_prompt(sample["question"], sample["options"])
        try:
            outputs = llm.generate([prompt], SAMPLING_PARAMS)
            model_answer_text = outputs[0].outputs[0].text.strip()
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
    print("测试结果统计报告")
    print("=" * 80)
    print(f"总测试样本数：{total_count}")
    print(f"正确数：{correct_count}")
    print(f"错误数：{len(wrong_samples)}")
    print(f"正确率：{accuracy:.2f}%")


# ===================== 4. 主函数 =====================
def main():
    print("正在加载Qwen2.5-14B-Instruct牙科模型（H100 vLLM加速）...")
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=TOKENIZER_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype=DTYPE,
    )
    print("模型加载完成！")

    TESTSET_PATH = "./data/dental/dental_sft_test.jsonl"
    run_testset(llm, TESTSET_PATH)


if __name__ == "__main__":
    main()
