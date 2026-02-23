import os
import json
import torch
from easyeditor import BaseEditor
from easyeditor.models.lora import LoRAHyperParams
import shutil
from torch.utils.data import Dataset, DataLoader
from transformers import BitsAndBytesConfig
import accelerate

# ===================== 1. H100 专属配置（核心优化）=====================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
NUM_WORKERS = 32
PIN_MEMORY = True

# 基础路径配置
MODEL_NAME = "./Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./dental_qwen2.5_1.5b_lora"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 训练性能优化配置
CHECKPOINT_STEP = 10
RESUME_TRAINING = True
BATCH_SIZE = 1  # 必须为1（EasyEdit Single Editing强制要求）
GRADIENT_ACCUMULATION_STEPS = 64
MAX_LENGTH = 2048
LEARNING_RATE = 2e-4

# ===================== 2. 自定义数据集 =====================
class DentalDataset(Dataset):
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

# ===================== 3. 数据加载 =====================
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
    dataset = DentalDataset(prompts, target_new, subject, ground_truth)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    return dataloader, prompts, target_new, subject, ground_truth

# ===================== 4. 检查点工具（彻底移除hparams非原生字段）=====================
def save_checkpoint(editor, optimizer, current_step, metrics, checkpoint_path):
    """保存检查点（仅保留LoRAHyperParams原生字段）"""
    # 只提取LoRAHyperParams原生初始化参数
    native_hparam_keys = LoRAHyperParams.__init__.__code__.co_varnames
    native_hparams = {
        k: v for k, v in editor.hparams.__dict__.items()
        if k in native_hparam_keys and k != 'self'
    }
    checkpoint = {
        "step": current_step,
        "lora_state_dict": editor.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "hparams": native_hparams,
        # 自定义配置单独存储，不混入hparams
        "train_config": {
            "torch_dtype": TORCH_DTYPE,
            "device": DEVICE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "bnb_config": bnb_config
        }
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

# ===================== 5. LoRA 超参数（仅保留原生字段！）=====================
def get_lora_hparams(resume_step=0):
    total_steps = 60
    remaining_steps = max(0, total_steps - resume_step)
    
    # 仅初始化LoRAHyperParams原生支持的字段，不附加任何自定义属性！
    hparams = LoRAHyperParams(
        lora_type="lora",                
        layers=[],                   
        num_steps=remaining_steps,       
        lr=LEARNING_RATE,                         
        weight_decay=0.01,               
        kl_factor=0.0,                   
        norm_constraint=None,            
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  
        rank=32,
        lora_alpha=64,
        lora_dropout=0.05,               
        device=DEVICE.split(":")[-1],    
        alg_name="LoRA",                 
        model_name=MODEL_NAME            
    )
    
    # 【核心修正】彻底删除所有手动附加的属性（batch_size/torch_dtype等全部删掉）
    # 以下代码全部移除，不要保留任何hparams.xxx = xxx的写法
    return hparams

# ===================== 6. 训练主逻辑（Accelerate正确配置）=====================
def main():
    # 初始化Accelerate（配置混合精度+梯度累积，替代hparams附加属性）
    accelerator = accelerate.Accelerator(
        mixed_precision="bf16",  # H100原生支持BF16
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        device_placement=True
    )
    
    # 1. 加载数据
    try:
        dataloader, prompts, target_new, subject, ground_truth = load_dental_data()
    except ValueError as e:
        accelerator.print(f"❌ 数据加载失败：{e}")
        return
    accelerator.print(f"加载牙科数据完成，共 {len(prompts)} 条样本，批次大小：{BATCH_SIZE}（梯度累积后等效 {BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS}）")
    
    # 2. 初始化编辑器（仅用原生hparams）
    resume_step = 0
    optimizer_state = None
    editor = None
    
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
    
    # H100 模型编译优化
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
        accelerator.print("✅ 开启H100专属torch.compile优化...")
        editor.model = torch.compile(editor.model, mode="max-autotune", fullgraph=True)
    
    # 用Accelerate包装模型和DataLoader
    editor.model, dataloader = accelerator.prepare(editor.model, dataloader)
    
    # 3. 启动训练
    accelerator.print(f"开始 Qwen2.5-1.5B-Instruct LoRA 微调（H100优化版）...")
    accelerator.print(f"ℹ️ 续训起始步数：{resume_step}，剩余训练步数：{hparams.num_steps}")
    accelerator.print(f"ℹ️ H100配置：BF16混合精度 + 4bit量化 + 梯度累积{GRADIENT_ACCUMULATION_STEPS}步")
    
    try:
        metrics, edited_model, optimizer = editor.edit(
            prompts=prompts,                
            target_new=target_new,          
            subject=subject,                
            ground_truth=ground_truth,      
            keep_original_weight=True,
            dataloader=dataloader,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        )
        
        # 保存最终检查点（移除hparams参数）
        final_step = resume_step + hparams.num_steps
        final_ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{final_step}.pth")
        save_checkpoint(editor, optimizer, final_step, metrics, final_ckpt_path)
        
    except KeyboardInterrupt:
        accelerator.print("\n⚠️ 检测到手动中断，保存当前训练状态...")
        interrupt_step = resume_step + (hparams.num_steps // 2) if hparams.num_steps > 0 else resume_step
        interrupt_ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{interrupt_step}.pth")
        save_checkpoint(editor, optimizer, interrupt_step, metrics if 'metrics' in locals() else {}, interrupt_ckpt_path)
        raise
    
    # 保存最终模型
    editor.save_model(OUTPUT_DIR)
    accelerator.print(f"微调完成！模型保存至：{OUTPUT_DIR}")
    accelerator.print(f"训练指标：{metrics}")

if __name__ == "__main__":
    main()