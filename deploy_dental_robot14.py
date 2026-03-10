import os
import torch

# 1. 部署配置
MODEL_PATH = "./dental_qwen2.5_14b_choice_lora_distill"
TOKENIZER_PATH = "./Qwen2.5-14B-Instruct"
GPU_MEMORY_UTILIZATION = 0.9
TORCH_DTYPE = torch.bfloat16


def is_adapter_only_model(path):
    """检测模型目录是否为纯 LoRA adapter（没有完整权重）"""
    if not os.path.isdir(path):
        return False
    files = os.listdir(path)
    has_adapter = "adapter_config.json" in files or "adapter_model.safetensors" in files
    has_full = any(f.startswith("model-") and f.endswith(".safetensors") for f in files)
    return has_adapter and not has_full


# 2. Prompt 模板（与训练一致）
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
你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。
<|im_end|>
<|im_start|>user
问题：{question}
选项：
{options}
<|im_end|>
<|im_start|>assistant
"""


# 3. vLLM 后端
def build_vllm_backend():
    from vllm import LLM, SamplingParams
    sampling_params_qa = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2048,
                                        stop=["<|endoftext|>", "</s>"])
    sampling_params_choice = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=10,
                                            stop=["<|endoftext|>", "</s>"])
    llm = LLM(
        model=MODEL_PATH, tokenizer=TOKENIZER_PATH,
        tensor_parallel_size=1, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype=TORCH_DTYPE,
    )
    return llm, sampling_params_qa, sampling_params_choice


# 4. transformers + PEFT 后端（当模型为纯 adapter 时自动使用）
def build_transformers_backend():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_PATH, torch_dtype=TORCH_DTYPE, trust_remote_code=True, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    return model, tokenizer


def main():
    print("正在加载Qwen2.5-14B-Instruct牙科模型...")

    use_peft = is_adapter_only_model(MODEL_PATH)
    if use_peft:
        print(f"检测到 {MODEL_PATH} 为纯 LoRA adapter，使用 transformers + PEFT 后端")
        model, tokenizer = build_transformers_backend()
    else:
        print(f"使用 vLLM 后端加载完整模型: {MODEL_PATH}")
        llm, sp_qa, sp_choice = build_vllm_backend()

    print("模型加载完成！可开始进行牙科问答/选择题交互。")

    while True:
        task_type = input("\n请选择任务类型（1=问答，2=选择题，0=退出）：")
        if task_type == "0":
            print("退出牙科机器人，感谢使用！")
            break
        elif task_type == "1":
            question = input("请输入你的牙科问题：")
            prompt = build_qa_prompt(question)
            if use_peft:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=2048, temperature=0.7, top_p=0.95,
                                         do_sample=True)
                answer = tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
            else:
                outputs = llm.generate([prompt], sp_qa)
                answer = outputs[0].outputs[0].text.strip()
            print(f"\n牙科医生回答：\n{answer}")
        elif task_type == "2":
            question = input("请输入选择题题干：")
            options = input("请输入选择题选项（格式：A.xxx B.xxx C.xxx D.xxx）：")
            prompt = build_choice_prompt(question, options)
            if use_peft:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                result = tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
            else:
                outputs = llm.generate([prompt], sp_choice)
                result = outputs[0].outputs[0].text.strip()
            print(f"\n选择题解答结果：\n{result}")
        else:
            print("无效任务类型，请重新选择！")


if __name__ == "__main__":
    main()