"""
CoT SFT 模型部署脚本 —— 交互式牙科问答/选择题
自动检测模型格式（LoRA adapter 或完整模型）
"""
import os
import torch

MODEL_DIR = "./dental_qwen2.5_7b_cot_sft"
BASE_MODEL = "./Qwen2.5-7B-Instruct"

SYSTEM_PROMPT_COT = (
    '你是一名专业的口腔医学专家。'
    '请按以下格式回答选择题：先给出"答案：X"（X为选项字母A-E），再给出"解析：..."说明理由。'
)
SYSTEM_PROMPT_QA = '你是一名专业的牙科医生，擅长解答各类口腔医学问题，回答需专业、准确、通俗易懂。'


def is_adapter_only(path):
    if not os.path.isdir(path):
        return False
    files = os.listdir(path)
    return "adapter_config.json" in files


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    if is_adapter_only(MODEL_DIR):
        print(f"检测到 LoRA adapter: {MODEL_DIR}，加载 base + adapter")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
        )
        model = PeftModel.from_pretrained(base, MODEL_DIR)
    else:
        print(f"加载完整模型: {MODEL_DIR}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
        )

    model.eval()
    return model, tokenizer


def build_qa_prompt(question):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT_QA}\n<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_choice_prompt(question, options):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT_COT}\n<|im_end|>\n"
        f"<|im_start|>user\n问题：{question}\n选项：\n{options}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    print("正在加载 CoT SFT 牙科模型...")
    model, tokenizer = load_model()
    print("模型加载完成！")

    while True:
        task_type = input("\n请选择任务类型（1=问答，2=选择题，0=退出）：")
        if task_type == "0":
            print("退出牙科机器人，感谢使用！")
            break
        elif task_type == "1":
            question = input("请输入你的牙科问题：")
            prompt = build_qa_prompt(question)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=True,
                                     temperature=0.7, top_p=0.95)
            answer = tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
            print(f"\n牙科医生回答：\n{answer}")
        elif task_type == "2":
            question = input("请输入选择题题干：")
            options = input("请输入选项（如 A.xxx B.xxx ...）：")
            prompt = build_choice_prompt(question, options)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            result = tokenizer.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
            print(f"\n选择题解答：\n{result}")
        else:
            print("无效任务类型，请重新选择！")


if __name__ == "__main__":
    main()
