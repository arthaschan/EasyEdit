from easyedit import LoRAEditing
import torch

model_name = "Qwen/Qwen2.5-1.8B-Instruct"
editor = LoRAEditing(
    model_name=model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)
print("模型加载成功，兼容性正常！")
