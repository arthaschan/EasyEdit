import os
import json
import torch
from easyeditor import BaseEditor
from easyeditor.models.lora import LoRAHyperParams
import shutil
from torch.utils.data import Dataset, DataLoader
from transformers import BitsAndBytesConfig  # 引入量化优化
import accelerate  # 加速库

# ===================== 1. H100 专属配置（核心优化）=====================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# H100 原生支持 BF16，强制开启
TORCH_DTYPE = torch.bfloat16
# 量化配置（减少显存占用，提升吞吐量）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4bit量化，H100支持无损
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# 数据加载多线程（H100 算力高，避免数据加载成为瓶颈）
NUM_WORKERS = 32  # 等于CPU核心数（H100服务器通常≥8核）
PIN_MEMORY = True  # 锁页内存，加速GPU数据传输

# 基础路径配置
MODEL_NAME = "./Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./dental_qwen2.5_1.5b_lora"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 训练性能优化配置
CHECKPOINT_STEP = 10
RESUME_TRAINING = True
BATCH_SIZE = 1  # 单H100可大幅提升批次（原1→8）
GRADIENT_ACCUMULATION_STEPS = 64  # 梯度累积，模拟更大批次（8*4=32）
MAX_LENGTH = 2048
LEARNING_RATE = 2e-4  # H100大批次可适当调大学习率

# ===================== 2. 自定义数据集（适配DataLoader多线程）=====================
class DentalDataset(Dataset):
    """牙科数据集类，支持多线程加载"""
    def __init__(self, prompts, target_new, subject, ground_truth):
        self.prompts = prompts
        self.target_new = target_new
        self.subject = subject
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "target_new": self.target_new[idx],
            "subject": self.subject[idx],
            "ground_truth": self.ground_truth[idx]
        }

# ===================== 3. 数据加载（多线程优化）=====================
def read_jsonl(file_path):
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

def load_dental_data():
    """加载数据并返回多线程DataLoader"""
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

    prompts = []
    target_new = []
    subject = []
    invalid_count = 0
    FIELD_MAPPING = {"instruction": "instruction", "input": "input", "output": "output"}

    for idx, item in enumerate(all_data):
        try:
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
            subject.append("牙科")

        except Exception as e:
            invalid_count += 1
            print(f"⚠️ 第 {idx+1} 条数据无效，跳过：{e}")
            continue

    if len(prompts) == 0:
        raise ValueError("❌ 有效数据为 0，请检查数据文件或字段映射！")
    print(f"有效数据筛选完成，共 {len(prompts)} 条（无效 {invalid_count} 条）")
    
    ground_truth = target_new
    # 构建多线程DataLoader
    dataset = DentalDataset(prompts, target_new, subject, ground_truth)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 打乱数据，提升训练效果
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True  # 丢弃最后不完整批次，避免训练波动
    )
    return dataloader, prompts, target_new, subject, ground_truth

# ===================== 4. 检查点工具（适配大批次）=====================
def save_checkpoint(editor, optimizer, current_step, metrics, hparams, checkpoint_path):
    checkpoint = {
        "step": current_step,
        "lora_state_dict": editor.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "hparams": hparams.__dict__,
        "torch_dtype": TORCH_DTYPE,
        "device": DEVICE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS
    }
    temp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, temp_path)
    shutil.move(temp_path, checkpoint_path)
    print(f"✅ 检查点已保存至：{checkpoint_path}（步数：{current_step}）")

def load_latest_checkpoint(editor, checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
    if not checkpoint_files:
        print("ℹ️ 未找到任何检查点，将从头开始训练")
        return False, 0, None
    
    checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
    latest_ckpt = checkpoint_files[0]
    ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    editor.model.load_state_dict(checkpoint["lora_state_dict"])
    resume_step = checkpoint["step"]
    optimizer_state = checkpoint["optimizer_state_dict"]
    
    print(f"✅ 加载最新检查点成功：{ckpt_path}（恢复步数：{resume_step}）")
    return True, resume_step, optimizer_state

# ===================== 5. LoRA 超参数（H100 优化版）=====================
def get_lora_hparams(resume_step=0):
    total_steps = 60
    remaining_steps = max(0, total_steps - resume_step)
    
    hparams = LoRAHyperParams(
        lora_type="lora",                
        layers=None,                     
        num_steps=remaining_steps,       
        lr=LEARNING_RATE,                         
        weight_decay=0.01,               
        kl_factor=0.0,                   
        norm_constraint=None,            
        # H100适配：增加更多目标模块，提升微调效果
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  
        rank=32,  # H100显存充足，提升rank至32（原16）
        lora_alpha=64,  # 对应rank调整
        lora_dropout=0.05,               
        device=DEVICE.split(":")[-1],    
        alg_name="LoRA",                 
        model_name=MODEL_NAME            
    )
    
    # H100 关键优化配置
    hparams.torch_dtype = TORCH_DTYPE   
    hparams.batch_size = BATCH_SIZE              
    hparams.max_length = MAX_LENGTH           
    hparams.use_chat_template = True    
    hparams.resume_step = resume_step
    # 开启混合精度训练（H100原生支持）
    hparams.fp16 = True
    hparams.bf16 = True
    # 梯度累积
    hparams.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    # 量化配置
    hparams.bnb_config = bnb_config
    
    return hparams

# ===================== 6. 训练主逻辑（H100 性能优化）=====================
def main():
    # 1. 加载多线程数据加载器
    try:
        dataloader, prompts, target_new, subject, ground_truth = load_dental_data()
    except ValueError as e:
        print(f"❌ 数据加载失败：{e}")
        return
    print(f"加载牙科数据完成，共 {len(prompts)} 条样本，批次大小：{BATCH_SIZE}（梯度累积后等效 {BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS}）")
    
    # 2. 初始化编辑器（适配H100）
    resume_step = 0
    optimizer_state = None
    if RESUME_TRAINING:
        temp_hparams = get_lora_hparams()
        temp_editor = BaseEditor.from_hparams(temp_hparams)
        load_success, resume_step, optimizer_state = load_latest_checkpoint(temp_editor, CHECKPOINT_DIR)
        if load_success:
            hparams = get_lora_hparams(resume_step)
            editor = BaseEditor.from_hparams(hparams)
            editor.model.load_state_dict(temp_editor.model.state_dict())
        else:
            hparams = get_lora_hparams()
            editor = BaseEditor.from_hparams(hparams)
    else:
        hparams = get_lora_hparams()
        editor = BaseEditor.from_hparams(hparams)
    
    # H100 核心优化：编译模型（torch.compile）
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:  # H100是9.0架构
        print("✅ 开启H100专属torch.compile优化...")
        editor.model = torch.compile(editor.model, mode="max-autotune", fullgraph=True)
    
    # 3. 启动训练（利用Accelerate加速）
    print(f"开始 Qwen2.5-1.5B-Instruct LoRA 微调（H100优化版）...")
    print(f"ℹ️ 续训起始步数：{resume_step}，剩余训练步数：{hparams.num_steps}")
    print(f"ℹ️ H100配置：BF16混合精度 + 4bit量化 + 梯度累积{GRADIENT_ACCUMULATION_STEPS}步")
    
    try:
        metrics, edited_model, optimizer = editor.edit(
            prompts=prompts,                
            target_new=target_new,          
            subject=subject,                
            ground_truth=ground_truth,      
            keep_original_weight=True,
            # 传递H100优化参数
            dataloader=dataloader,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        )
        
        # 保存最终检查点
        final_ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{resume_step + hparams.num_steps}.pth")
        save_checkpoint(editor, optimizer, resume_step + hparams.num_steps, metrics, hparams, final_ckpt_path)
        
    except KeyboardInterrupt:
        print("\n⚠️ 检测到手动中断，保存当前训练状态...")
        interrupt_step = resume_step + (hparams.num_steps // 2)
        interrupt_ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{interrupt_step}.pth")
        save_checkpoint(editor, optimizer, interrupt_step, metrics if 'metrics' in locals() else {}, hparams, interrupt_ckpt_path)
        raise
    
    # 保存最终模型
    editor.save_model(OUTPUT_DIR)
    print(f"微调完成！模型保存至：{OUTPUT_DIR}")
    print(f"训练指标：{metrics}")

if __name__ == "__main__":
    # 初始化Accelerate（H100分布式必备）
    accelerator = accelerate.Accelerator()
    main()