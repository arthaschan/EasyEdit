import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# ===================== 1. 基础配置（核心修改：本地Qwen模型路径） =====================
# 替换为你本地Qwen2.5-1.5B-Instruct模型的实际路径
LOCAL_MODEL_PATH = "./Qwen2.5-7B-Instruct"  
# 替换为你测试集的实际路径
#TESTSET_PATH = "./dental_choice_testset.jsonl"  
TESTSET_PATH = "./data/dental/dental_sft_test.jsonl"  # 替换为你的jsonl测试集路径
# 生成配置（关闭随机性，仅输出字母）
GEN_CONFIG = GenerationConfig(
    temperature=0.0,  # 固定温度，确保回答稳定
    top_p=1.0,
    max_new_tokens=10,  # 仅需输出字母，限制生成长度
    pad_token_id=151643,  # Qwen的pad_token_id
    eos_token_id=151643,  # Qwen的eos_token_id
    do_sample=False,  # 关闭采样，确定性输出
)

# ===================== 2. 核心工具函数（适配你的数据格式） =====================
def extract_question_options(user_content):
    """
    从user的content中提取问题和选项（适配中文"问题：""选项："格式）
    输入：原始user content字符串
    输出：question（问题文本）、options（选项字典，如{"A":"xxx", "B":"xxx"}）
    """
    lines = user_content.strip().split("\n")
    question = ""
    options = {}
    is_option = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 跳过指令行，提取核心问题
        if line.startswith("请回答以下选择题"):
            continue
        # 提取问题（中文"问题："）
        elif line.startswith("问题："):
            question = line.replace("问题：", "").strip()
        # 识别选项开始（中文"选项："）
        elif line.startswith("选项："):
            is_option = True
        # 提取选项（A/B/C/D/E开头，兼容中英文冒号）
        elif is_option and len(line) >= 2 and line[1] in [":", "："]:
            option_key = line[0].upper()
            option_value = line[2:].strip()
            options[option_key] = option_value

    return question, options

def extract_answer_char(answer_text):
    """
    从文本中提取纯字母答案（A/B/C/D/E），处理模型输出多余内容的情况
    """
    for char in answer_text.strip().upper():
        if char in ["A", "B", "C", "D", "E"]:
            return char
    return ""

def load_jsonl_testset(file_path):
    """加载并解析jsonl测试集，返回标准化的样本列表"""
    test_samples = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"测试集文件不存在：{file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            try:
                data = json.loads(line.strip())
                # 提取user问题和assistant正确答案
                user_content = ""
                correct_answer = ""
                for conv in data.get("conversations", []):
                    if conv.get("role") == "user":
                        user_content = conv.get("content", "")
                    elif conv.get("role") == "assistant":
                        correct_answer = conv.get("content", "")

                # 解析问题和选项
                question, options = extract_question_options(user_content)
                correct_answer_char = extract_answer_char(correct_answer)

                # 过滤无效样本
                if not question or not options or not correct_answer_char:
                    print(f"警告：第{line_idx+1}行数据解析失败，跳过")
                    continue

                test_samples.append({
                    "line_idx": line_idx + 1,
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer_char
                })
            except Exception as e:
                print(f"错误：第{line_idx+1}行数据解析出错 - {str(e)}，跳过")
                continue

    print(f"成功加载{len(test_samples)}条有效测试样本（总行数：{len(lines)}）")
    return test_samples

# ===================== 3. 模型推理+正确率统计 =====================
def run_qwen_test():
    """加载本地Qwen模型，运行测试集并统计正确率"""
    # 1. 加载本地Qwen模型和tokenizer
    print("正在加载本地Qwen2.5-1.5B-Instruct模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # 适配GPU，若无BF16可用torch.float16
        device_map="auto",  # 自动分配到GPU/CPU
        load_in_4bit=False,  # 若显存不足可改为True（需安装bitsandbytes）
    )
    model.eval()  # 推理模式
    print("模型加载完成！")

    # 2. 加载测试集
    test_samples = load_jsonl_testset(TESTSET_PATH)
    if not test_samples:
        print("无有效测试样本，终止测试")
        return

    # 3. 构建Qwen专用prompt（遵循官方对话格式）
    def build_qwen_prompt(question, options):
        options_text = "\n".join([f"{k}：{v}" for k, v in options.items()])
        # Qwen2.5标准对话格式
        prompt = f"""<|im_start|>system
你是一名专业的牙科医生，仅需输出正确选项的字母（如A、B、C、D、E），不要输出其他内容，无需额外解释。
<|im_end|>
<|im_start|>user
问题：{question}
选项：
{options_text}
<|im_end|>
<|im_start|>assistant
"""
        return prompt

    # 4. 批量推理+答案对比
    correct_count = 0
    wrong_samples = []

    print("\n开始批量测试...")
    with torch.no_grad():  # 禁用梯度计算，节省显存
        for sample in tqdm(test_samples, desc="测试进度"):
            # 构建prompt并编码
            prompt = build_qwen_prompt(sample["question"], sample["options"])
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

            # 模型生成回答
            try:
                outputs = model.generate(
                    **inputs,
                    generation_config=GEN_CONFIG
                )
                # 解码生成结果（仅保留assistant部分）
                model_answer_text = tokenizer.decode(
                    outputs[0][len(inputs["input_ids"][0]):],
                    skip_special_tokens=True
                ).strip()
                # 提取纯字母答案
                model_answer_char = extract_answer_char(model_answer_text)
            except Exception as e:
                print(f"\n错误：第{sample['line_idx']}行样本预测失败 - {str(e)}")
                model_answer_char = ""
                model_answer_text = ""

            # 统计结果
            sample["model_answer"] = model_answer_char
            sample["model_raw_answer"] = model_answer_text
            if model_answer_char == sample["correct_answer"]:
                correct_count += 1
            else:
                wrong_samples.append(sample)

    # 5. 输出详细统计报告
    total_count = len(test_samples)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    print("\n" + "="*80)
    print("Qwen2.5-1.5B-Instruct 牙科选择题测试报告")
    print("="*80)
    print(f"总测试样本数：{total_count}")
    print(f"正确数：{correct_count}")
    print(f"错误数：{len(wrong_samples)}")
    print(f"正确率：{accuracy:.2f}%")

    # 输出错误样本详情
    # if wrong_samples:
    #     print("\n错误样本详情：")
    #     print("-"*80)
        # for idx, wrong_sample in enumerate(wrong_samples, 1):
        #     print(f"【错误{idx}】行号：{wrong_sample['line_idx']}")
        #     print(f"问题：{wrong_sample['question'][:60]}..." if len(wrong_sample['question'])>60 else f"问题：{wrong_sample['question']}")
        #     print(f"正确答案：{wrong_sample['correct_answer']}")
        #     print(f"模型答案：{wrong_sample['model_answer'] or '无有效回答'}")
        #     print(f"模型原始回答：{wrong_sample['model_raw_answer'] or '预测失败'}")
        #     print("-"*80)

# ===================== 4. 主函数 =====================
if __name__ == "__main__":
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"检测到GPU：{torch.cuda.get_device_name(0)}，使用GPU推理")
    else:
        print("未检测到GPU，使用CPU推理（速度较慢）")
    
    # 运行测试
    run_qwen_test()