import os
import json
import torch
import numpy as np  # 新增：处理numpy相关序列化
from easyeditor import BaseEditor  # 核心编辑器基类
from easyeditor.models.lora import LoRAHyperParams  # LoRA 超参数类
import shutil
# 有断点续训的功能

# ===================== 修复PyTorch 2.6+ weights_only问题（全局配置）=====================
# 允许加载可信的numpy全局变量（解决UnpicklingError）
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar,
    np.ndarray,
    np.float32,
    np.float64
])

# ===================== 自定义依赖（修复 read_jsonl + JSON序列化）=====================
def read_jsonl(file_path):
    """读取 JSONL 文件（EasyEdit 无内置，自定义实现）"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ 第 {line_idx+1} 行 JSON 解析失败，跳过：{e}")
                continue
    return data

def json_serialize_hparams(obj):
    """自定义JSON序列化函数，处理torch dtype等特殊类型"""
    # 处理torch dtype
    if isinstance(obj, torch.dtype):
        if obj == torch.bfloat16:
            return "bfloat16"
        elif obj == torch.float16:
            return "float16"
        elif obj == torch.float32:
            return "float32"
        elif obj == torch.float64:
            return "float64"
        else:
            return str(obj)
    # 处理numpy类型
    if isinstance(obj, np.generic):
        return obj.item()
    # 处理其他可能的非序列化对象
    raise TypeError(f"无法序列化类型: {type(obj)}")

# ===================== 1. 基础配置（新增断点续训相关）=====================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16
# 本地模型路径（替换为你实际的路径）
MODEL_NAME = "./Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./dental_qwen2.5_7b_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 断点续训核心配置 =====
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")  # 检查点保存目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_STEP = 10  # 每10步保存一次检查点
RESUME_TRAINING = True  # 是否开启断点续训（True=开启，False=从头训练）

# ===================== 2. 字段映射配置（适配你的数据源格式）=====================
FIELD_MAPPING = {
    "instruction": "instruction",  # 指令字段
    "input": "input",              # 具体输入字段
    "output": "output"             # 输出答案字段
}

# ===================== 2. 加载牙科数据（返回单独的参数列表，而非requests）=====================
def load_dental_data():
    """加载并预处理牙科数据，返回prompts/target_new/subject/ground_truth（适配edit方法）"""
    # 读取数据文件
    train_data_paths = [
        "./data/easyedit_dental_qa.jsonl",
        "./data/easyedit_dental_choice.jsonl"
    ]
    all_data = []
    for path in train_data_paths:
        if not os.path.exists(path):
            print(f"⚠️ 数据文件不存在，跳过：{path}")
            continue
        file_data = read_jsonl(path)
        all_data.extend(file_data)
    print(f"原始数据加载完成，共 {len(all_data)} 条")

    # 提取核心字段列表（直接返回，不转换为requests）
    prompts = []
    target_new = []
    subject = []
    invalid_count = 0

    for idx, item in enumerate(all_data):
        try:
            # 1. 合并 instruction + input 作为 prompt
            instr_field = FIELD_MAPPING["instruction"]
            input_field = FIELD_MAPPING["input"]
            output_field = FIELD_MAPPING["output"]
            
            if instr_field not in item or input_field not in item:
                raise KeyError(f"缺少 instruction/input 字段")
            if output_field not in item:
                raise KeyError(f"缺少 output 字段")
            
            full_prompt = f"{item[instr_field]}\n{item[input_field]}"
            prompts.append(full_prompt)
            target_new.append(item[output_field])
            subject.append("牙科")  # 默认主题

        except Exception as e:
            invalid_count += 1
            print(f"⚠️ 第 {idx+1} 条数据无效，跳过：{e}")
            continue

    # 数据校验
    if len(prompts) == 0:
        raise ValueError("❌ 有效数据为 0，请检查数据文件或字段映射！")
    print(f"有效数据筛选完成，共 {len(prompts)} 条（无效 {invalid_count} 条）")
    
    # 返回单独的列表（而非requests），适配edit方法参数要求
    ground_truth = target_new  # ground_truth与target_new一致
    return prompts, target_new, subject, ground_truth

# ===================== 3. 检查点工具函数（核心：保存/加载）=====================
def save_checkpoint(editor, optimizer, current_step, metrics, hparams, checkpoint_path):
    """保存检查点（包含模型、优化器、步数、超参数等）"""
    checkpoint = {
        "step": current_step,
        "lora_state_dict": editor.model.state_dict(),  # LoRA模型权重
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,  # 优化器状态
        "metrics": metrics,  # 训练指标
        "hparams": hparams.__dict__,  # 超参数
        "torch_dtype": TORCH_DTYPE,
        "device": DEVICE
    }
    # 保存检查点（先存临时文件，避免写入中断损坏）
    temp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, temp_path)
    shutil.move(temp_path, checkpoint_path)
    print(f"✅ 检查点已保存至：{checkpoint_path}（步数：{current_step}）")

def load_latest_checkpoint(editor, checkpoint_dir):
    """加载最新的检查点，返回 (是否加载成功, 恢复的步数, 优化器状态)"""
    # 查找所有检查点文件
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
    if not checkpoint_files:
        print("ℹ️ 未找到任何检查点，将从头开始训练")
        return False, 0, None
    
    # 按步数排序，取最新的检查点
    checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
    latest_ckpt = checkpoint_files[0]
    ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
    
    # 修复：加载检查点时关闭weights_only（可信源），并指定map_location
    try:
        checkpoint = torch.load(
            ckpt_path, 
            map_location=DEVICE,
            weights_only=False  # 关键：关闭权重仅加载（自己的检查点可信）
        )
    except Exception as e:
        print(f"⚠️ 加载检查点失败：{e}")
        return False, 0, None
    
    # 恢复模型权重
    editor.model.load_state_dict(checkpoint["lora_state_dict"])
    # 恢复步数和优化器状态
    resume_step = checkpoint["step"]
    optimizer_state = checkpoint["optimizer_state_dict"]
    
    print(f"✅ 加载最新检查点成功：{ckpt_path}（恢复步数：{resume_step}）")
    return True, resume_step, optimizer_state

# ===================== 3. LoRA 超参数配置（适配断点续训）=====================
def get_lora_hparams(resume_step=0):
    """构建 LoRA 超参数（适配断点续训：调整剩余训练步数）"""
    # 原始总步数
    total_steps = 60
    # 若续训，调整剩余步数
    remaining_steps = max(0, total_steps - resume_step)
    
    hparams = LoRAHyperParams(
        lora_type="lora",                
        layers=[],                     
        num_steps=remaining_steps,       # 续训时为剩余步数
        lr=1e-4,                         
        weight_decay=0.01,               
        kl_factor=0.0,                   
        norm_constraint=None,            
        target_modules=["q_proj", "v_proj"],  
        rank=8,                         
        lora_alpha=16,                   
        lora_dropout=0.05,               
        device=DEVICE.split(":")[-1],    
        alg_name="LoRA",                 
        model_name=MODEL_NAME            
    )
    
    # 赋值额外参数
    hparams.torch_dtype = TORCH_DTYPE   
    hparams.batch_size = 1              
    hparams.max_length = 1024           
    hparams.use_chat_template = True    
    # 记录续训起始步数
    hparams.resume_step = resume_step
    
    return hparams

# ===================== 4. 训练主逻辑（带断点续训）=====================
def main():
    # 1. 加载数据（返回单独的参数列表）
    try:
        prompts, target_new, subject, ground_truth = load_dental_data()
    except ValueError as e:
        print(f"❌ 数据加载失败：{e}")
        return
    print(f"加载牙科数据完成，共 {len(prompts)} 条样本")
    
    # 2. 初始化检查点相关
    resume_step = 0
    optimizer_state = None
    if RESUME_TRAINING:
        # 先实例化基础编辑器（用于加载检查点）
        temp_hparams = get_lora_hparams()
        temp_editor = BaseEditor.from_hparams(temp_hparams)
        # 加载最新检查点
        load_success, resume_step, optimizer_state = load_latest_checkpoint(temp_editor, CHECKPOINT_DIR)
        if load_success:
            # 加载成功后，用剩余步数重新构建hparams
            hparams = get_lora_hparams(resume_step)
            # 替换编辑器为带剩余步数的hparams
            editor = BaseEditor.from_hparams(hparams)
            # 恢复模型权重（从temp_editor迁移）
            editor.model.load_state_dict(temp_editor.model.state_dict())
        else:
            # 无检查点，从头训练
            hparams = get_lora_hparams()
            editor = BaseEditor.from_hparams(hparams)
    else:
        # 不续训，从头训练
        hparams = get_lora_hparams()
        editor = BaseEditor.from_hparams(hparams)
    
 
    # 开启梯度检查点（牺牲速度换显存）
    #editor.model.gradient_checkpointing_enable()

    # 打印edit方法参数（调试用）
    import inspect
    print("=== BaseEditor.edit() 方法参数 ===")
    print(inspect.signature(editor.edit))
    
    # 3. 启动 LoRA 微调（核心：带检查点保存）
    print(f"开始 Qwen2.5-1.5B-Instruct LoRA 微调（牙科数据）...")
    print(f"ℹ️ 续训起始步数：{resume_step}，剩余训练步数：{hparams.num_steps}")
    
    try:
        # 执行训练（EasyEdit封装的edit方法）
        metrics, edited_model, optimizer = editor.edit(
            prompts=prompts,                # 必填：模型输入提示
            target_new=target_new,          # 必填：目标输出
            subject=subject,                # 可选：数据主题
            ground_truth=ground_truth,      # 可选：真实标签
            keep_original_weight=True       # 保留原模型权重
        )
        
        # 训练完成后，保存最终检查点
        final_ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{resume_step + hparams.num_steps}.pth")
        save_checkpoint(editor, optimizer, resume_step + hparams.num_steps, metrics, hparams, final_ckpt_path)
        
    except KeyboardInterrupt:
        # 捕获手动中断（Ctrl+C），保存中断时的检查点
        print("\n⚠️ 检测到手动中断，保存当前训练状态...")
        interrupt_step = resume_step + (hparams.num_steps // 2)  # 估算当前步数（EasyEdit封装较深，可根据实际调整）
        interrupt_ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{interrupt_step}.pth")
        save_checkpoint(editor, optimizer, interrupt_step, metrics if 'metrics' in locals() else {}, hparams, interrupt_ckpt_path)
        raise
    
    # 4. 保存最终 LoRA 模型（修复：替换不存在的save_model）
    print(f"正在保存最终 LoRA 模型至：{OUTPUT_DIR}")
    editor.model.save_pretrained(OUTPUT_DIR)  # PEFT原生保存方法
    
    # 保存超参数（修复：处理dtype序列化）
    hparams_path = os.path.join(OUTPUT_DIR, "lora_hparams.json")
    with open(hparams_path, "w", encoding="utf-8") as f:
        json.dump(hparams.__dict__, f, ensure_ascii=False, indent=4, default=json_serialize_hparams)
    
    print(f"微调完成！模型保存至：{OUTPUT_DIR}")
    print(f"训练指标：{metrics}")

if __name__ == "__main__":
    main()