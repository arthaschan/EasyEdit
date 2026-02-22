import torch
import transformers
import peft
import vllm
import datasets
import bitsandbytes

# 验证PyTorch CUDA是否可用（H100关键验证）
print(f"PyTorch版本：{torch.__version__}")
print(f"CUDA是否可用：{torch.cuda.is_available()}")
print(f"CUDA版本：{torch.version.cuda}")

# 验证核心依赖版本
print(f"Transformers版本：{transformers.__version__}")
print(f"PEFT版本：{peft.__version__}")
print(f"VLLM版本：{vllm.__version__}")
print(f"Datasets版本：{datasets.__version__}")

# 验证Qwen2.5模型是否可正常加载（贴合你的项目需求）
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.8B-Instruct", torch_dtype=torch.bfloat16).to("cuda")
print("Qwen2.5模型加载成功，无版本冲突！")
