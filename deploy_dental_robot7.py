import torch
import json
import os

# 尝试导入vLLM库；若不可用则回退到transformers
try:
    from vllm import LLM, SamplingParams
    _USE_VLLM = True
except ImportError:
    print("[警告] 未安装 vllm，脚本将使用 transformers 进行推理。"
          " 若需要高性能推理，请安装 vllm: `pip install vllm`.")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _USE_VLLM = False


# 1. 部署配置（7B LoRA 微调＋蒸馏模型）
# 使用标准LoRA微调后的模型路径（distill+LoRA 结果）
MODEL_PATH = "./dental_qwen2.5_7b_choice_lora_distill_easyedit"
# 若想切换到原始7B 训练模型可恢复为上面路径
# MODEL_PATH = "./dental_qwen2.5_7b_choice_lora"
TOKENIZER_PATH = "./Qwen2.5-7B-Instruct"
GPU_MEMORY_UTILIZATION = 0.9
# vLLM dtype 参数现在命名为 dtype
TORCH_DTYPE = torch.bfloat16


# 2. 采样参数配置（无修改）
if _USE_VLLM:
    SAMPLING_PARAMS = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        stop=["<|endoftext|>", "</s>"]
    )
else:
    # transformers 不需要采样参数，但保持接口兼容
    SAMPLING_PARAMS = None

# 3. Prompt 模板（与训练保持一致）
def build_qa_prompt(question):
    return f"""<|im_start|>system
你是一名专业的牙科医生，擅长解答各类口腔医学问题，回答需专业、准确、通俗易懂，符合中文表达习惯。
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""

# ====== 批量评估相关工具 ======

def extract_question_options(user_content: str):
    """从 user 字段提取题干和选项，适配训练/测试格式。"""
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


def extract_answer_char(answer_text: str) -> str:
    """从模型回答或正确答案中提取首个 A-E 字母"""
    for char in answer_text.strip().upper():
        if char in ["A", "B", "C", "D", "E"]:
            return char
    return ""


def load_cmexam_testset(file_path: str):
    """加载 cmexam_dental_choice 格式的测试数据"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            question = data.get("Question", "")
            options_text = data.get("Options", "")
            correct_answer = data.get("Answer", "")

            # 解析选项
            options = {}
            for opt_line in options_text.split('\n'):
                opt_line = opt_line.strip()
                if len(opt_line) >= 3 and opt_line[1] in ['.', '：', ':', ' ']:
                    option_key = opt_line[0].upper()
                    # 找到分隔符的位置
                    sep_pos = 1
                    if opt_line[1] in ['.', '：', ':']:
                        sep_pos = 2
                    elif opt_line[1] == ' ' and len(opt_line) > 2 and opt_line[2] in ['.', '：', ':']:
                        sep_pos = 3
                    option_value = opt_line[sep_pos:].strip()
                    options[option_key] = option_value

            if question and options and correct_answer:
                samples.append({
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer
                })
    print(f"成功加载{len(samples)}条测试样本（文件：{file_path}）")
    return samples


def evaluate_on_testset(llm, test_path: str):
    """使用当前模型对指定测试集进行批量预测并打印准确率"""
    samples = load_cmexam_testset(test_path)
    if not samples:
        print("无有效测试样本，跳出")
        return
    correct = 0
    print("\n开始批量测试...")
    for s in samples:
        prompt = build_choice_prompt(s["question"], "\n".join([f"{k}. {v}" for k, v in s["options"].items()]))
        out = llm.generate([prompt], SAMPLING_PARAMS)
        raw = out[0].outputs[0].text.strip()
        pred = extract_answer_char(raw)
        if pred == s["correct_answer"]:
            correct += 1
    acc = 100 * correct / len(samples)
    print(f"测试样本 {len(samples)} | 正确 {correct} | 准确率 {acc:.2f}%")


def is_adapter_only_model(model_path: str) -> bool:
    """判断给定目录是否为 PEFT LoRA adapter（非完整 HF 模型）。"""
    return os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")) and not os.path.exists(os.path.join(model_path, "config.json"))


def build_transformers_backend():
    """使用 transformers + PEFT 加载 LoRA 适配器作为推理后端。"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # load the base model (not the LoRA adapter directory)
    base = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_PATH,  # original model path with config
        torch_dtype=TORCH_DTYPE,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True
    )

    # patch config to drop any unsupported keys by comparing against LoraConfig signature
    import inspect
    from peft import LoraConfig
    cfg_path = os.path.join(MODEL_PATH, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        sig = inspect.signature(LoraConfig.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self"}
        cleaned = {k: v for k, v in cfg.items() if k in valid_keys}
        if cleaned != cfg:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f)
            print("已从adapter_config.json移除不受支持的字段:", set(cfg.keys()) - set(cleaned))

    llm_model = PeftModel.from_pretrained(base, MODEL_PATH)
    llm_model.eval()

    class TFWrapper2:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.device = next(model.parameters()).device

        def generate(self, prompts, sampling_params=None):
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            input_lengths = inputs["attention_mask"].sum(dim=1)
            outputs = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)
            results = []
            for i, out_ids in enumerate(outputs):
                gen_ids = out_ids[int(input_lengths[i]):]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                inner = type("Inner", (), {"text": text})
                wrapper = type("Wrapper", (), {"outputs": [inner]})
                results.append(wrapper)
            return results

    return TFWrapper2(llm_model, tokenizer)

def build_choice_prompt(question, options):
    # 与训练数据中使用的 prompt 完全一致
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

# 4. 初始化 vLLM 模型（dtype 设置 + 注释量化）
def main():
    # 解析命令行参数（可选择仅运行评估）
    import argparse
    parser = argparse.ArgumentParser(description="部署牙科模型并可选批量评估")
    parser.add_argument("--eval_path", type=str, help="指定测试集 jsonl 路径以计算准确率")
    args = parser.parse_args()

    # 要求 vLLM 以避免兼容性问题
    if not _USE_VLLM:
        print("[错误] 由于当前环境未安装 vllm 或 PEFT 版本不兼容，Transformer 回退无法加载 LoRA 模型。")
        print("请安装 vllm (`pip install vllm`) 或升级 peft/transformers 至兼容版本后重试。")
        return

    print("正在加载Qwen2.5-7B-Instruct牙科模型（H100推理）...")
    llm = None
    if is_adapter_only_model(MODEL_PATH):
        print("检测到 MODEL_PATH 为 LoRA adapter 目录（缺少 config.json），跳过 vLLM，直接使用 transformers+PEFT。")
        try:
            llm = build_transformers_backend()
            print("Transformer+PEFT推理后端已准备好。")
        except Exception as e:
            print(f"[错误] 载入LoRA权重失败: {e}")
            return
    else:
        try:
            llm = LLM(
                model=MODEL_PATH,
                tokenizer=TOKENIZER_PATH,
                tensor_parallel_size=1,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                dtype=TORCH_DTYPE,  # 关键修改：torch_dtype → dtype
            )
            print("vLLM模型加载完成！可开始进行牙科问答/选择题交互。")
        except Exception as e:
            print(f"[警告] vLLM加载失败: {e}")
            print("尝试使用transformers + PEFT载入LoRA权重进行推理。")
            try:
                llm = build_transformers_backend()
                print("Transformer+PEFT推理后端已准备好。")
            except Exception as e2:
                print(f"[错误] 载入LoRA权重失败: {e2}")
                return

    # 如果提供了评估路径，立即运行并退出
    if args.eval_path:
        evaluate_on_testset(llm, args.eval_path)
        return

    # 5. 交互/批量测试逻辑
    while True:
        task_type = input("\n请选择任务类型（1=问答，2=选择题，3=评估测试集，0=退出）：")
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
        elif task_type == "3":
            # 批量评估测试集
            test_path = "./data/cmexam_dental_choice_test.jsonl"
            evaluate_on_testset(llm, test_path)
        else:
            print("无效任务类型，请重新选择！")

if __name__ == "__main__":
    main()
