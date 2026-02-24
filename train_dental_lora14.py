import os
import json
import torch
import numpy as np
from easyeditor import BaseEditor
from easyeditor.models.lora import LoRAHyperParams
import shutil
from transformers import BitsAndBytesConfig
import gc

# ===================== 全局配置 & 显存优化 =====================
# 解决显存碎片问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 清空GPU缓存
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 修复PyTorch 2.6+ weights_only安全加载问题
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar,
    np.ndarray,
    np.float32,
    np.float64
])

# 设备配置
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ===================== 工具函数 =====================
def read_jsonl(file_path):
    """逐行读取JSONL文件，避免一次性加载占用显存"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
                # 每200行清理一次显存
                if (line_idx + 1) % 200 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            except json.JSONDecodeError as e:
                print(f"⚠️ 第 {line_idx+1} 行 JSON 解析失败，跳过：{e}")
                continue
    return data

def json_serialize_hparams(obj):
    """自定义JSON序列化，处理特殊类型（torch dtype/BNB配置）"""
    if isinstance(obj, torch.dtype):
        return str(obj).split(".")[-1]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, BitsAndBytesConfig):
        return obj.to_dict()
    raise TypeError(f"无法序列化类型: {type(obj)}")

def count_total_valid_samples():
    """动态统计所有JSONL文件中的有效样本数（避免写死）"""
    train_data_paths = [
        "./data/easyedit_dental_qa.jsonl",
        "./data/easyedit_dental_choice.jsonl"
    ]
    total_valid = 0
    total_invalid = 0

    for path in train_data_paths:
        if not os.path.exists(path):
            print(f"⚠️ 数据文件不存在，跳过：{path}")
            continue
        
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    total_invalid += 1
                    continue
                try:
                    item = json.loads(line)
                    # 有效样本判断：必须包含instruction/input/output字段
                    if "instruction" in item and "input" in item and "output" in item:
                        total_valid += 1
                    else:
                        total_invalid += 1
                except json.JSONDecodeError:
                    total_invalid += 1
                    continue
        
        # 统计完一个文件清理显存
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n📊 数据统计完成：")
    print(f"总有效样本数：{total_valid}")
    print(f"总无效样本数：{total_invalid}")
    return total_valid

def load_dental_data_chunk(chunk_start, chunk_end):
    """分批加载数据（仅加载指定区间的样本，控制显存）"""
    train_data_paths = [
        "./data/easyedit_dental_qa.jsonl",
        "./data/easyedit_dental_choice.jsonl"
    ]
    all_data = []
    current_global_idx = 0  # 跨文件的全局样本索引

    for path in train_data_paths:
        if not os.path.exists(path):
            print(f"⚠️ 数据文件不存在，跳过：{path}")
            continue
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if current_global_idx >= chunk_end:
                    break
                if current_global_idx < chunk_start:
                    current_global_idx += 1
                    continue
                
                line = line.strip()
                if not line:
                    current_global_idx += 1
                    continue
                
                try:
                    item = json.loads(line)
                    # 仅保留有效样本
                    if "instruction" in item and "input" in item and "output" in item:
                        all_data.append(item)
                    current_global_idx += 1
                except json.JSONDecodeError:
                    current_global_idx += 1
                    continue
        
        # 加载完一个文件清理显存
        gc.collect()
        torch.cuda.empty_cache()

    # 适配Qwen2.5的Chat模板（解决Loss无变化的关键）
    prompts = []
    target_new = []
    subject = []
    for item in all_data:
        # Qwen2.5标准Chat模板：<|im_start|>user/assistant 标识
        full_prompt = f"<|im_start|>user\n{item['instruction']}\n{item['input']}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(full_prompt)
        # Target补充结束标识，保证格式对齐
        target_new.append(item['output'] + "<|im_end|>")
        subject.append("牙科")

    ground_truth = target_new
    print(f"✅ 加载批次数据完成：{chunk_start}-{chunk_end}（有效：{len(prompts)}）")
    return prompts, target_new, subject, ground_truth

def save_checkpoint(editor, optimizer, current_step, metrics, hparams, checkpoint_path):
    """保存检查点（保存前清理显存，避免溢出）"""
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint = {
        "step": current_step,
        "lora_state_dict": editor.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "hparams": hparams.__dict__,
        "device": DEVICE,
        "bnb_config": BNB_CONFIG.to_dict()
    }
    # 保存到CPU，减少GPU占用
    torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
    print(f"✅ 检查点已保存至：{checkpoint_path}（步数：{current_step}）")

def load_latest_checkpoint(editor, checkpoint_dir):
    """加载最新检查点（加载前清理显存）"""
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
    if not checkpoint_files:
        print("ℹ️ 未找到任何检查点，将从头开始训练")
        return False, 0, None
    
    checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
    latest_ckpt = checkpoint_files[0]
    ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
    
    try:
        # 先加载到CPU，再移到GPU，降低显存峰值
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        editor.model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
        editor.model.to(DEVICE)
        resume_step = checkpoint["step"]
        optimizer_state = checkpoint["optimizer_state_dict"]
        
    except Exception as e:
        print(f"⚠️ 加载检查点失败：{e}")
        return False, 0, None
    
    print(f"✅ 加载最新检查点成功：{ckpt_path}（恢复步数：{resume_step}）")
    return True, resume_step, optimizer_state

# ===================== 核心配置 =====================
# 4bit量化配置（提升精度，解决梯度截断问题）
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,    # 双重量化，省显存
    bnb_4bit_quant_type="nf4",         # NF4量化，精度更高
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,  # 优先bfloat16
    bnb_4bit_quant_storage=torch.uint8
)

# 模型/输出路径
MODEL_NAME = "./Qwen2.5-14B-Instruct"
OUTPUT_DIR = "./dental_qwen2.5_14b_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 断点续训配置
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_STEP = 5  # 每5步保存一次检查点
RESUME_TRAINING = True

# 分批配置（控制显存，批次内逐条训练）
CHUNK_SIZE = 50  # 每次加载50条，可根据显存调整
TOTAL_EPOCHS = 3  # 总训练轮数

# ===================== LoRA超参数配置（优化后解决Loss无变化）=====================
def get_lora_hparams(resume_step=0, total_valid_samples=0):
    """
    LoRA超参数配置：
    - 调整lr/weight_decay解决Loss无变化
    - 扩展target_modules覆盖Qwen2.5核心层
    - 显式设置batch_size=1（EasyEdit强制要求）
    """
    # 计算总步数（逐条训练，每样本1步）
    total_steps = int(total_valid_samples * TOTAL_EPOCHS) if total_valid_samples > 0 else 0
    # 计算剩余步数（续训时使用）
    remaining_steps = max(0, total_steps - resume_step)
    
    hparams = LoRAHyperParams(
        lora_type="lora",                
        layers=[],                      # 空列表表示训练所有层
        num_steps=remaining_steps,       # 剩余训练步数
        lr=5e-4,                         # 提升学习率（解决Loss无变化）
        weight_decay=0.0001,             # 降低权重衰减（减少梯度压制）
        kl_factor=0.0,                   # 无KL约束
        norm_constraint=None,           
        # 扩展target_modules，覆盖Qwen2.5-14B核心层（解决Loss无变化）
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  
        rank=16,                         # 提升LoRA秩，增加学习容量
        lora_alpha=32,                   # 同步提升alpha，保持rank/alpha比例
        lora_dropout=0.05,              
        device=DEVICE.split(":")[-1],   
        alg_name="LoRA",                
        model_name=MODEL_NAME,
        batch_size=1                     # EasyEdit Single Editing强制要求
    )
    
    hparams.resume_step = resume_step
    return hparams

# ===================== 训练主逻辑 =====================
def main():
    # 1. 动态统计总有效样本数
    total_valid_samples = count_total_valid_samples()
    if total_valid_samples == 0:
        print("❌ 无有效训练样本，终止训练")
        return
    
    # 2. 初始化LoRA编辑器
    hparams = get_lora_hparams(total_valid_samples=total_valid_samples)
    editor = BaseEditor.from_hparams(hparams)
       # 🚨 新增：给Qwen模型添加空的unload()方法，规避EasyEdit兼容性问题
    def dummy_unload():
        pass
    if not hasattr(editor.model, 'unload'):
        editor.model.unload = dummy_unload
    
    # 开启梯度检查点（省显存）+ 修复梯度回传
    editor.model.gradient_checkpointing_enable()
    editor.model.config.use_cache = True  # 解决梯度中断问题
    
    # 3. 加载检查点（断点续训）
    resume_step = 0
    if RESUME_TRAINING:
        load_success, resume_step, _ = load_latest_checkpoint(editor, CHECKPOINT_DIR)
        # 续训时重新计算剩余步数
        hparams = get_lora_hparams(resume_step=resume_step, total_valid_samples=total_valid_samples)
    
    # 4. 打印训练配置（方便核对）
    total_chunks = (total_valid_samples + CHUNK_SIZE - 1) // CHUNK_SIZE  # 向上取整
    print(f"\n🚀 最终训练配置：")
    print(f"总有效样本数：{total_valid_samples}")
    print(f"分批加载大小：{CHUNK_SIZE}")
    print(f"总加载批次/轮：{total_chunks} × {TOTAL_EPOCHS} 轮")
    print(f"总训练步数：{total_valid_samples * TOTAL_EPOCHS}")
    print(f"LoRA学习率：{hparams.lr} | Batch Size：{hparams.batch_size}")
    print(f"量化精度：{BNB_CONFIG.bnb_4bit_compute_dtype}")
    
    # 5. 分批训练 + 批次内逐条训练（核心逻辑）
    global_step = resume_step
    for epoch in range(TOTAL_EPOCHS):
        print(f"\n========== 第 {epoch+1}/{TOTAL_EPOCHS} 轮训练 ==========")
        epoch_loss = []  # 记录本轮所有样本的Loss，验证是否下降
        
        for chunk_idx in range(total_chunks):
            # 计算当前批次的样本区间
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, total_valid_samples)
            if chunk_start >= chunk_end:
                break
            
            # 加载当前批次数据
            prompts, target_new, subject, ground_truth = load_dental_data_chunk(chunk_start, chunk_end)
            if len(prompts) == 0:
                print(f"⚠️ 批次 {chunk_idx+1} 无有效数据，跳过")
                continue
            
            # 批次内逐条训练（适配EasyEdit batch_size=1要求）
            print(f"\n训练批次 {chunk_idx+1}/{total_chunks}（样本：{chunk_start}-{chunk_end}），共{len(prompts)}条样本")
            for idx in range(len(prompts)):
                try:
                    # 混合精度训练，降低显存占用
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        metrics, edited_model, optimizer = editor.edit(
                            prompts=[prompts[idx]],                # 单样本传入
                            target_new=[target_new[idx]],          # 单样本传入
                            subject=[subject[idx]],                # 单样本传入
                            ground_truth=[ground_truth[idx]],      # 单样本传入
                            keep_original_weight=False             # 允许LoRA参数更新（关键）
                        )
                    global_step += 1
                    batch_loss = metrics.get("loss", 0.0)
                    epoch_loss.append(batch_loss)
                    
                    # 打印单条样本训练结果
                    print(f"✅ 单条样本 {idx+1}/{len(prompts)} 训练完成 | 全局步数：{global_step} | Batch Loss：{batch_loss:.6f}")
                    
                    # 验证LoRA梯度是否有效（排查Loss无变化）
                    has_gradient = False
                    for name, param in editor.model.named_parameters():
                        if "lora" in name.lower() and param.grad is not None:
                            grad_norm = torch.norm(param.grad).item()
                            if grad_norm > 0:
                                has_gradient = True
                                # 每10步打印一次梯度（避免日志刷屏）
                                if global_step % 10 == 0:
                                    print(f"📌 LoRA参数 {name[:20]}... 梯度范数：{grad_norm:.6f}")
                                break
                    if not has_gradient and global_step % 10 == 0:
                        print(f"⚠️ 单条样本 {idx+1} 无有效梯度（LoRA未更新）")
                    
                    # 保存检查点（每5步/最后一条样本）
                    if global_step % CHECKPOINT_STEP == 0 or (epoch == TOTAL_EPOCHS-1 and chunk_idx == total_chunks-1 and idx == len(prompts)-1):
                        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{global_step}.pth")
                        save_checkpoint(editor, optimizer, global_step, metrics, hparams, ckpt_path)
                    
                    # 单条训练后清理显存
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"⚠️ 单条样本 {idx+1}/{len(prompts)} 训练失败：{e}")
                    torch.cuda.empty_cache()
                    continue
            
            # 释放当前批次显存
            del prompts, target_new, subject, ground_truth
            gc.collect()
            torch.cuda.empty_cache()
        
        # 打印本轮Loss统计（验证是否下降）
        if epoch_loss:
            avg_epoch_loss = np.mean(epoch_loss)
            print(f"\n📈 第 {epoch+1} 轮训练完成 | 平均Loss：{avg_epoch_loss:.6f} | 本轮最小Loss：{np.min(epoch_loss):.6f}")
    
    # 6. 保存最终模型
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n正在保存最终 LoRA 模型至：{OUTPUT_DIR}")
    editor.model.save_pretrained(OUTPUT_DIR)
    
    # 保存超参数（方便后续复现）
    hparams_path = os.path.join(OUTPUT_DIR, "lora_hparams.json")
    with open(hparams_path, "w", encoding="utf-8") as f:
        json.dump(hparams.__dict__, f, ensure_ascii=False, indent=4, default=json_serialize_hparams)
    
    # 打印最终显存使用情况
    print(f"\n✅ 所有训练完成！模型保存至：{OUTPUT_DIR}")
    print(f"\n最终显存使用：")
    print(f"GPU 已用：{torch.cuda.memory_allocated(0)/1024**3:.2f} GiB")
    print(f"GPU 剩余：{torch.cuda.memory_free(0)/1024**3:.2f} GiB")

if __name__ == "__main__":
    main()