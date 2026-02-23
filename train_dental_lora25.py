import os
import json
import torch
from easyeditor import BaseEditor  # 核心编辑器基类
from easyeditor.models.lora import LoRAHyperParams  # LoRA 超参数类
# 基础能正常的执行的版本
# ===================== 自定义依赖（修复 read_jsonl）=====================
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

# ===================== 1. 基础配置 =====================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16
# 本地模型路径（替换为你实际的路径）
MODEL_NAME = "./Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./dental_qwen2.5_1.8b_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ===================== 3. LoRA 超参数配置（最终版）=====================
def get_lora_hparams():
    """构建 LoRA 超参数（仅传入构造函数必填参数）"""
    hparams = LoRAHyperParams(
        lora_type="lora",                
        layers=[],                    
        num_steps=60,                    
        lr=2e-4,                         
        weight_decay=0.01,               
        kl_factor=0.0,                   
        norm_constraint=None,            
        target_modules=["q_proj", "v_proj"],  
        rank=16,                         
        lora_alpha=32,                   
        lora_dropout=0.05,               
        device=DEVICE.split(":")[-1],    
        alg_name="LoRA",                 
        model_name=MODEL_NAME            
    )
    
    # 赋值额外参数
    hparams.torch_dtype = TORCH_DTYPE   
    hparams.batch_size = 1              
    hparams.max_length = 2048           
    hparams.use_chat_template = True    
    
    return hparams

# ===================== 4. 训练主逻辑（修复edit方法参数传递）=====================
def main():
    # 1. 加载数据（返回单独的参数列表）
    try:
        prompts, target_new, subject, ground_truth = load_dental_data()
    except ValueError as e:
        print(f"❌ 数据加载失败：{e}")
        return
    print(f"加载牙科数据完成，共 {len(prompts)} 条样本")
    
    # 2. 加载 LoRA 超参数
    hparams = get_lora_hparams()
    
    # 3. 实例化 LoRA 编辑器
    editor = BaseEditor.from_hparams(hparams)
    import inspect
    print("=== BaseEditor.edit() 方法参数 ===")
    print(inspect.signature(editor.edit))
    # 4. 启动 LoRA 微调（核心修复：传入正确的必填参数）
    print("开始 Qwen2.5-1.8B-Instruct LoRA 微调（牙科数据）...")
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,                # 必填：模型输入提示
        target_new=target_new,          # 必填：目标输出
        subject=subject,                # 可选：数据主题
        ground_truth=ground_truth,      # 可选：真实标签
        keep_original_weight=True       # 保留原模型权重
    )
    
    # 5. 保存 LoRA 模型
    editor.save_model(OUTPUT_DIR)
    print(f"微调完成！模型保存至：{OUTPUT_DIR}")
    print(f"训练指标：{metrics}")

if __name__ == "__main__":
    main()