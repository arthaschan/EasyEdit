#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linux 环境下专用：下载 Qwen2.5-0.5B-Instruct 模型并验证
"""
import os
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope.utils.constant import DownloadMode

def main():
    # 1. 核心配置（可按需修改）
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Qwen2.5-0.5B-Instruct 固定模型ID
    CACHE_DIR = "./qwen2.5-0.5B-Instruct"    # 自定义模型保存目录（Linux 绝对/相对路径均可）
    FORCE_REDOWNLOAD = False  # False：缓存优先（推荐），True：强制重新下载

    # 2. 自动创建自定义模型目录（Linux 支持多级目录自动创建）
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"✅ 已创建自定义模型目录：{CACHE_DIR}")
    else:
        print(f"ℹ️  自定义模型目录已存在：{CACHE_DIR}")

    try:
        # 3. 加载分词器（适配 Qwen2.5 模型特性，关闭 fast 分词器）
        print("\n🚀 开始加载/下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=False,
            use_fast=False,  # 必须关闭 fast 分词器，适配 Qwen2.5 格式
            cache_dir=CACHE_DIR  # 指定模型保存到自定义目录，避免分散在系统缓存
        )
        print("✅ 分词器加载/下载完成")

        # 4. 加载模型（自动下载权重文件，Linux 下自动适配 CPU/GPU）
        print("\n🚀 开始加载/下载 Qwen2.5-0.5B-Instruct 模型（约 1GB，耐心等待）...")
        download_mode = DownloadMode.FORCE_REDOWNLOAD if FORCE_REDOWNLOAD else DownloadMode.CACHE_FIRST
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=False,
            torch_dtype="auto",  # 自动匹配精度，CPU 下为 float32，GPU 下自动优化
            device_map="auto",   # 自动分配设备（优先 GPU，无 GPU 则用 CPU，适配 Linux 环境）
            auto_download=True,
            download_mode=download_mode,
            cache_dir=CACHE_DIR
        )
        print("✅ 模型加载/下载完成")

        # 5. 简单对话验证（确认模型可用，排除下载损坏问题）
        print("\n📝 开始模型功能验证...")
        messages = [
            {"role": "user", "content": "你好，简要介绍一下你自己"}
        ]
        # 构建 Qwen2.5 标准对话模板
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # 生成回复（适配小模型，调整生成参数避免冗余）
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,  # 小模型无需过长生成，节省 CPU 资源
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

        # 解码并打印回复
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        print(f"\n✅ 模型验证成功，回复如下：\n{response}")

        # 6. 打印模型保存路径（方便后续直接调用）
        print(f"\n📂 模型文件已完整保存至：{os.path.abspath(CACHE_DIR)}")
        print("ℹ️  后续加载模型可直接指定该目录，无需重复下载")

    except Exception as e:
        print(f"\n❌ 执行失败，错误信息：{str(e)}")
        print("ℹ️  Linux 环境专属解决方案：")
        print("   1. 升级依赖至要求版本：pip3 install --upgrade transformers>=4.40.0 modelscope>=1.10.0 torch safetensors")
        print("   2. 确认磁盘空间充足（至少 2GB 剩余）")
        print("   3. 网络超时可配置镜像：export MODEL_SCOPE_HUB_ENDPOINT=https://modelscope.cn/api/v1")
        print("   4. 权限不足可添加 sudo 运行：sudo python3 download_qwen2.5_0.5b.py")

if __name__ == "__main__":
    main()
