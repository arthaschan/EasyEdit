import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 本地模型路径（和上面的 MODEL_NAME 一致）
model_path = "."

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"  # 对应你的 DEVICE
)

print("✅ 本地模型加载成功！")