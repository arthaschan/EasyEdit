from vllm import LLM, SamplingParams
import torch
import json
import os
from tqdm import tqdm  # 进度条，方便查看测试进度

# ===================== 1. 基础配置（与原代码保持一致） =====================
MODEL_PATH = "./dental_qwen2.5_7b_lora"
TOKENIZER_PATH = "./Qwen2.5-7B-Instruct"
GPU_MEMORY_UTILIZATION = 0.9
DTYPE = torch.bfloat16

# 采样参数：关闭随机性（temperature=0），确保回答稳定
SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,  # 固定温度为0，避免随机回答
    top_p=1.0,
    max_tokens=10,  # 仅需输出字母，限制最大生成长度
    stop=["<|endoftext|>", "</s>"]
)

# ===================== 2. 核心工具函数（关键调整：确保解析"选项："） =====================
def extract_question_options(user_content):
    """
    从user的content中提取问题和选项（适配中文"选项："格式）
    输入：原始user content字符串
    输出：question（问题文本）、options（选项字典，如{"A":"xxx", "B":"xxx"}）
    """
    # 分割问题和选项部分
    lines = user_content.strip().split("\n")
    question = ""
    options = {}
    
    # 标记是否开始读取选项（识别中文"选项："）
    is_option = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 跳过指令行，提取真正的问题
        if line.startswith("请回答以下选择题"):
            continue
        # 识别问题行（中文"问题："）
        elif line.startswith("问题："):
            question = line.replace("问题：", "").strip()
        # 识别选项开始行（中文"选项："，核心调整点）
        elif line.startswith("选项："):
            is_option = True
        # 提取选项（A/B/C/D/E开头，适配中文冒号/英文冒号）
        elif is_option and len(line) >= 2 and line[1] in [":", "："]:
            option_key = line[0].upper()  # 转为大写（防止小写a/b）
            option_value = line[2:].strip()
            options[option_key] = option_value
    
    return question, options

def extract_answer_char(answer_text):
    """
    从模型回答/正确答案中提取纯字母（A/B/C/D/E）
    处理边界情况：模型可能输出多余内容（如"C选项"、"答案是C"），仅提取第一个字母
    """
    # 遍历文本，找到第一个A-E的字母
    for char in answer_text.strip().upper():
        if char in ["A", "B", "C", "D", "E"]:
            return char
    # 若未找到，返回空字符串
    return ""

def load_jsonl_testset(file_path):
    """
    加载jsonl格式的测试集（适配中文"选项："格式）
    输入：文件路径
    输出：测试样本列表，每个样本包含question、options、correct_answer
    """
    test_samples = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"测试集文件不存在：{file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            try:
                # 解析单条json
                data = json.loads(line.strip())
                # 提取user和assistant内容
                user_content = ""
                correct_answer = ""
                for conv in data.get("conversations", []):
                    if conv.get("role") == "user":
                        user_content = conv.get("content", "")
                    elif conv.get("role") == "assistant":
                        correct_answer = conv.get("content", "")
                
                # 解析问题和选项（使用调整后的解析函数）
                question, options = extract_question_options(user_content)
                # 提取正确答案字母
                correct_answer_char = extract_answer_char(correct_answer)
                
                if not question or not options or not correct_answer_char:
                    print(f"警告：第{line_idx+1}行数据解析失败，跳过（内容：{line[:100]}...）")
                    continue
                
                test_samples.append({
                    "line_idx": line_idx + 1,  # 行号（方便定位错误）
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer_char
                })
            except Exception as e:
                print(f"错误：第{line_idx+1}行数据解析出错 - {str(e)}，跳过（内容：{line[:100]}...）")
                continue
    
    print(f"成功加载{len(test_samples)}条有效测试样本（总行数：{len(lines)}）")
    return test_samples

# ===================== 3. 批量测试+正确率统计（无调整） =====================
def run_testset(llm, testset_path):
    """
    运行测试集并输出统计结果
    """
    # 1. 加载测试集
    test_samples = load_jsonl_testset(testset_path)
    if not test_samples:
        print("无有效测试样本，终止测试")
        return
    
    # 2. 构建prompt模板（适配中文"选项："格式）
    def build_test_prompt(question, options):
        """构建测试用的prompt，要求仅输出字母"""
        options_text = "\n".join([f"{k}：{v}" for k, v in options.items()])
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
    
    # 3. 批量预测
    correct_count = 0
    wrong_samples = []  # 记录错误样本，用于分析
    
    print("\n开始批量测试...")
    for sample in tqdm(test_samples, desc="测试进度"):
        # 构建prompt
        prompt = build_test_prompt(sample["question"], sample["options"])
        # 模型生成回答
        try:
            outputs = llm.generate([prompt], SAMPLING_PARAMS)
            model_answer_text = outputs[0].outputs[0].text.strip()
            # 提取模型回答的字母
            model_answer_char = extract_answer_char(model_answer_text)
        except Exception as e:
            print(f"\n错误：第{sample['line_idx']}行样本预测失败 - {str(e)}")
            model_answer_char = ""
        
        # 对比答案
        sample["model_answer"] = model_answer_char
        sample["model_raw_answer"] = model_answer_text  # 保留原始回答（方便排查）
        if model_answer_char == sample["correct_answer"]:
            correct_count += 1
        else:
            wrong_samples.append(sample)
    
    # 4. 统计结果
    total_count = len(test_samples)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    # 5. 输出详细报告
    print("\n" + "="*80)
    print("测试结果统计报告")
    print("="*80)
    print(f"总测试样本数：{total_count}")
    print(f"正确数：{correct_count}")
    print(f"错误数：{len(wrong_samples)}")
    print(f"正确率：{accuracy:.2f}%")
    
    # 输出错误样本详情
    #if wrong_samples:
    #    print("\n错误样本详情（行号 | 问题 | 正确答案 | 模型答案 | 模型原始回答）：")
    #    print("-"*80)
    #    for wrong_sample in wrong_samples:
    #        print(f"行号：{wrong_sample['line_idx']}")
    #        print(f"问题：{wrong_sample['question'][:50]}..." if len(wrong_sample['question'])>50 else f"问题：{wrong_sample['question']}")
    #        print(f"正确答案：{wrong_sample['correct_answer']}")
     #       print(f"模型答案：{wrong_sample['model_answer'] or '无有效回答'}")
     #       print(f"模型原始回答：{wrong_sample['model_raw_answer'] or '预测失败'}")
      #      print("-"*80)

# ===================== 4. 主函数 =====================
def main():
    # 加载模型
    print("正在加载Qwen2.5-1.5B-Instruct牙科模型（H100 vLLM加速）...")
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=TOKENIZER_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype=DTYPE,
    )
    print("模型加载完成！")
    
    # 运行测试集（请修改为你的测试集路径）
    TESTSET_PATH = "./data/dental/dental_sft_test.jsonl"  # 替换为你的jsonl测试集路径
    run_testset(llm, TESTSET_PATH)

if __name__ == "__main__":
    main()