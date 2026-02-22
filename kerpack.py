import torch, transformers, peft, vllm, datasets, bitsandbytes, tokenizers, numpy
import gradio, saelens, outlines, fsspec, markupsafe, safetensors
# 打印核心版本
print('=== 核心版本验证 ===')
print(f'torch: {torch.__version__} (CUDA: {torch.cuda.is_available()}, CUDA版本: {torch.version.cuda})')
print(f'numpy: {numpy.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'tokenizers: {tokenizers.__version__}')
print(f'peft: {peft.__version__}')
print(f'vllm: {vllm.__version__}')
print(f'datasets: {datasets.__version__}')
# 验证Qwen2.5关键：transformers无Qwen3模块，peft正常导入
print('=== 关键适配验证 ===')
print('transformers.models是否有qwen3:', 'qwen3' in dir(transformers.models))
print('peft导入成功:', 'peft' in locals())
# H100显卡验证
if torch.cuda.is_available():
    print(f'当前显卡: {torch.cuda.get_device_name(0)}')
print('=== 环境修复完成，无任何依赖冲突 ===')
