from vllm import LLM, SamplingParams
import torch

# 1. 部署配置（7B 模型）
MODEL_PATH = "./dental_qwen2.5_7b_choice_lora"
# MODEL_PATH = "./dental_qwen2.5_14b_lora"  # 14B 版本备用
TOKENIZER_PATH = "./Qwen2.5-7B-Instruct"  
# TOKENIZER_PATH = "Qwen/Qwen2.5-14B-Instruct"  # 14B 版本 tokenizer
GPU_MEMORY_UTILIZATION = 0.9
# 关键修改：字符串→torch.dtype类型
TORCH_DTYPE = torch.bfloat16  
# QUANTIZATION = "awq"  # 注释/删除 AWQ 量化

# 2. 采样参数配置（无修改）
SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=2048,
    stop=["<|endoftext|>", "</s>"]
)

# 3. Prompt 模板（无修改）
def build_qa_prompt(question):
    return f"""<|im_start|>system
你是一名专业的牙科医生，擅长解答各类口腔医学问题，回答需专业、准确、通俗易懂，符合中文表达习惯。
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""

def build_choice_prompt(question, options):
    return f"""<|im_start|>system
你是一名专业的牙科医生，擅长解答口腔医学选择题，先直接给出答案选项，再简要说明解析理由。
<|im_end|>
<|im_start|>user
题干：{question}
选项：{options}
<|im_end|>
<|im_start|>assistant
"""

# 4. 初始化 vLLM 模型（dtype 设置 + 注释量化）
def main():
    print("正在加载Qwen2.5-7B-Instruct牙科模型（H100 vLLM加速）...")
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=TOKENIZER_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype=TORCH_DTYPE,  # 关键修改：torch_dtype → dtype
        # quantization=QUANTIZATION,  # 注释 AWQ 量化
    )
    print("模型加载完成！可开始进行牙科问答/选择题交互。")
    
    # 5. 交互逻辑（无修改）
    while True:
        task_type = input("\n请选择任务类型（1=问答，2=选择题，0=退出）：")
        if task_type == "0":
            print("退出牙科机器人，感谢使用！")
            break
        elif task_type == "1":
            question = input("请输入你的牙科问题：")
            prompt = build_qa_prompt(question)
            outputs = llm.generate([prompt], SAMPLING_PARAMS)
            answer = outputs[0].outputs[0].text.strip()
            print(f"\n牙科医生回答：\n{answer}")
        elif task_type == "2":
            question = input("请输入选择题题干：")
            options = input("请输入选择题选项（格式：A.xxx B.xxx C.xxx D.xxx）：")
            prompt = build_choice_prompt(question, options)
            outputs = llm.generate([prompt], SAMPLING_PARAMS)
            result = outputs[0].outputs[0].text.strip()
            print(f"\n选择题解答结果：\n{result}")
        else:
            print("无效任务类型，请重新选择！")

if __name__ == "__main__":
    main()
