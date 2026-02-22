#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linux 环境下专用：下载 Qwen1.5-1.8B-Chat 模型并验证
"""
import os
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope.utils.constant import DownloadMode

def main():
    # 1. 配置参数（可根据需求修改）
    MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"  # 模型固定ID
    # 自定义模型保存路径（Linux 下建议绝对路径，如 "./qwen_model" 或 "/home/xxx/models/qwen"）
    CACHE_DIR = "./qwen1.5-1.8B-Chat"
    # 强制重新下载（False：缓存优先，已下载则不重复下载；True：强制重新下载）
    FORCE_REDOWNLOAD = False

    # 2. 创建自定义缓存目录（Linux 下自动创建多级目录）
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"✅ 已创建自定义模型目录：{CACHE_DIR}")
    else:
        print(f"ℹ️  自定义模型目录已存在：{CACHE_DIR}")

    try:
        # 3. 加载分词器（适配 Qwen 模型，Linux 环境无编码兼容问题）
        print("\n🚀 开始加载分词器（若未下载，将自动下载相关文件）...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=False,
            use_fast=False,
            cache_dir=CACHE_DIR  # 指定模型保存到自定义目录
        )
        print("✅ 分词器加载/下载完成")

        # 4. 加载模型（自动下载权重文件，Linux 下支持 GPU/CPU 自动分配）
        print("\n🚀 开始加载模型（模型约 3.7GB，下载速度取决于网络，请耐心等待）...")
        download_mode = DownloadMode.FORCE_REDOWNLOAD if FORCE_REDOWNLOAD else DownloadMode.CACHE_FIRST
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=False,
            torch_dtype="auto",  # 自动匹配精度，Linux 下 GPU 会自动使用 float16
            device_map="auto",   # 自动分配设备（优先 GPU，无 GPU 则使用 CPU）
            auto_download=True,
            download_mode=download_mode,
            cache_dir=CACHE_DIR  # 指定模型保存到自定义目录
        )
        print("✅ 模型加载/下载完成")

        # 5. 简单对话验证（确认模型可用）
        print("\n📝 开始进行模型对话验证...")
        messages = [
            {"role": "user", "content": "你好，介绍一下你自己"}
        ]
        # 构建对话模板
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # 生成回复
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        # 解码并打印回复
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        print(f"\n✅ 模型验证成功，回复如下：\n{response}")

        # 6. 打印模型文件路径（方便用户查找）
        print(f"\n📂 模型文件已保存至：{os.path.abspath(CACHE_DIR)}")
        print("ℹ️  后续加载模型可直接指定该目录，无需重复下载")

    except Exception as e:
        print(f"\n❌ 执行失败，错误信息：{str(e)}")
        print("ℹ️  常见解决方案：")
        print("   1. 确认依赖已安装：pip3 install modelscope>=1.10.0 transformers>=4.37.0 torch safetensors")
        print("   2. 确认磁盘空间充足（至少 5GB 剩余）")
        print("   3. 网络不佳可配置 ModelScope 镜像源")

if __name__ == "__main__":
    main()
