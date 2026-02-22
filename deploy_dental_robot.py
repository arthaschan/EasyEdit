from vllm import LLM, SamplingParams
import torch

# 1. 部署配置（H100专属优化）
MODEL_PATH = "./dental_qwen_1.8b_lora"  # 替换为你的微调模型保存路径
TOKENIZER_PATH = "Qwen/Qwen-1.8B-Chat"  # Qwen官方Tokenizer
GPU_MEMORY_UTILIZATION = 0.9  # H100显存利用率，0.9兼顾性能与稳定性
TORCH_DTYPE = "bfloat16"  # H100最优精度
QUANTIZATION = "awq"  # 开启AWQ 4bit量化，显存占用降低60%（可选，如需更高性能可关闭）

# 2. 采样参数配置（控制对话/选择题输出质量）
SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,  # 随机性，0.7兼顾流畅度与准确性
    top_p=0.95,  # 核采样，提升输出相关性
    max_tokens=2048,  # 最大输出长度
    stop=["<|endoftext|>", "</s>"]  # Qwen模型停止符
)

# 3. 标准化Prompt模板（中文牙科场景专属）
def build_qa_prompt(question):
    """构建牙科问答Prompt"""
    return f"""<|im_start|>system
你是一名专业的牙科医生，擅长解答各类口腔医学问题，回答需专业、准确、通俗易懂，符合中文表达习惯。
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""

def build_choice_prompt(question, options):
    """构建牙科选择题Prompt"""
    return f"""<|im_start|>system
你是一名专业的牙科医生，擅长解答口腔医学选择题，先直接给出答案选项，再简要说明解析理由。
<|im_end|>
<|im_start|>user
题干：{question}
选项：{options}
<|im_end|>
<|im_start|>assistant
"""

def main():
    # 4. 初始化vLLM模型（H100加速，支持高并发）
    print("正在加载Qwen-1.8B-Chat牙科模型（H100 vLLM加速）...")
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=TOKENIZER_PATH,
        tensor_parallel_size=1,  # 单卡H100（多卡可改为对应数量）
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        torch_dtype=TORCH_DTYPE,
        quantization=QUANTIZATION,
        trust_remote_code=True  # 信任Qwen自定义代码
    )
    print("模型加载完成！可开始进行牙科问答/选择题交互。")
    
    # 5. 交互演示（毕业设计演示用，可改为API接口）
    while True:
        task_type = input("\n请选择任务类型（1=问答，2=选择题，0=退出）：")
        if task_type == "0":
            print("退出牙科机器人，感谢使用！")
            break
        elif task_type == "1":
            question = input("请输入你的牙科问题：")
            prompt = build_qa_prompt(question)
            outputs = llm.generate([prompt], SAMPLING_PARAMS)
            # 提取并打印回答
            answer = outputs[0].outputs[0].text.strip()
            print(f"\n牙科医生回答：\n{answer}")
        elif task_type == "2":
            question = input("请输入选择题题干：")
            options = input("请输入选择题选项（格式：A.xxx B.xxx C.xxx D.xxx）：")
            prompt = build_choice_prompt(question, options)
            outputs = llm.generate([prompt], SAMPLING_PARAMS)
            # 提取并打印答案与解析
            result = outputs[0].outputs[0].text.strip()
            print(f"\n选择题解答结果：\n{result}")
        else:
            print("无效任务类型，请重新选择！")

if __name__ == "__main__":
    main()
