from huggingface_hub import snapshot_download

HF_TOKEN = "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"  # 建议真实使用时从环境变量读取

local_dir = snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct", #Qwen2.5-14B-Instruct  Qwen2.5-32B-Instruct
    local_dir=".",
    local_dir_use_symlinks=False,
    resume_download=True,
    token=HF_TOKEN,  # 关键参数
)
print(local_dir)