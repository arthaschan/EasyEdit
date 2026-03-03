import os
import json
import random
import re
import torch
import numpy as np
from easyeditor import BaseEditor
from easyeditor.models.lora import LoRAHyperParams
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# ===================== 全局配置 =====================
# PyTorch 2.6+ 反序列化修复
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar,
    np.ndarray,
    np.float32,
    np.float64
])

# 基础配置
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16
MODEL_NAME = "./Qwen2.5-7B-Instruct"  # 7B模型路径
OUTPUT_DIR = "./dental_qwen2.5_7b_choice_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 断点续训配置
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_STEP = 5
RESUME_TRAINING = False  # 首次训练关闭续训

# 牙科术语同义词典（数据增强用）
DENTAL_SYNONYM = {
    "牙髓炎": ["牙髓炎症", "牙髓感染"],
    "牙周病": ["牙周炎症", "牙周病变"],
    "龋齿": ["蛀牙", "虫牙"],
    "智齿": ["第三磨牙", "立事牙"],
    "多生牙": ["额外牙", "多余牙"],
    "乳牙滞留": ["乳牙迟脱", "乳牙未掉"],
    "地图舌": ["剥脱性舌炎", "游走性舌炎"],
    "沟纹舌": ["脑回舌", "皱褶舌"]
}

# ===================== 自定义Logits处理器（替代allowed_tokens）=====================
class AllowedTokensLogitsProcessor:
    """自定义Logits处理器，限制仅输出指定token（A/B/C/D/E）"""
    def __init__(self, tokenizer, allowed_chars=["A", "B", "C", "D", "E"]):
        self.allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_chars)
        # 过滤无效token ID（避免-100）
        self.allowed_token_ids = [tid for tid in self.allowed_token_ids if tid != tokenizer.unk_token_id]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 创建掩码，将非允许token的概率设为负无穷
        mask = torch.ones_like(scores) * -float("inf")
        mask[:, self.allowed_token_ids] = 0
        scores = scores + mask
        return scores

# ===================== 工具函数 =====================
def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    if not os.path.exists(file_path):
        print(f"⚠️ 数据文件不存在：{file_path}")
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ 第{line_idx+1}行JSON解析失败：{e}")
                continue
    print(f"📄 成功读取 {file_path}：{len(data)} 条数据")
    return data

def json_serialize_hparams(obj):
    """自定义JSON序列化（处理dtype）"""
    if isinstance(obj, torch.dtype):
        return str(obj).split(".")[-1]
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"无法序列化类型: {type(obj)}")

def parse_options(options_text):
    """解析Options字段为(A, 内容)格式的列表"""
    options = []
    # 按换行分割选项行
    option_lines = [line.strip() for line in options_text.split("\n") if line.strip()]
    for line in option_lines:
        # 匹配 A xxx / A.xxx / A：xxx 等格式
        match = re.match(r"([A-Z])[\s.：:、]*(.*)", line)
        if match:
            opt_char = match.group(1).strip()
            opt_content = match.group(2).strip()
            if opt_char and opt_content:
                options.append((opt_char, opt_content))
    return options

def get_teacher_answer(prompt, teacher_model, teacher_tokenizer):
    """调用老师模型获取选择题答案（修复allowed_tokens警告）"""
    inputs = teacher_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # 创建Logits处理器，限制仅输出A/B/C/D/E
    logits_processor = LogitsProcessorList([
        AllowedTokensLogitsProcessor(teacher_tokenizer, ["A", "B", "C", "D", "E"])
    ])
    
    with torch.no_grad():
        outputs = teacher_model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.0,
            top_k=1,
            pad_token_id=teacher_tokenizer.eos_token_id,
            logits_processor=logits_processor,  # 替代allowed_tokens
            do_sample=False  # 确保确定性输出
        )
    answer = teacher_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # 兜底处理
    final_ans = ""
    for char in answer[::-1]:  # 从后往前找选项字母
        if char in ["A", "B", "C", "D", "E"]:
            final_ans = char
            break
    if not final_ans:
        final_ans = "A"
    return final_ans

def infer_choice(model, tokenizer, prompt):
    """定制化选择题推理函数（修复allowed_tokens警告）"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # 创建Logits处理器，限制仅输出A/B/C/D/E
    logits_processor = LogitsProcessorList([
        AllowedTokensLogitsProcessor(tokenizer, ["A", "B", "C", "D", "E"])
    ])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.0,
            top_k=1,
            top_p=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor  # 替代allowed_tokens
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # 概率兜底
    final_ans = ""
    for char in answer[::-1]:
        if char in ["A", "B", "C", "D", "E"]:
            final_ans = char
            break
    if not final_ans:
        logits = model(**inputs).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        choice_ids = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D", "E"])
        choice_probs = [probs[0][idx] if idx < probs.shape[-1] else 0.0 for idx in choice_ids]
        final_ans = ["A", "B", "C", "D", "E"][torch.argmax(torch.tensor(choice_probs))]
    return final_ans

# ===================== 数据加载（适配新格式）=====================
def load_dental_choice_data():
    """加载并优化牙科选择题数据（适配cmexam_dental_choice.jsonl格式）"""
    # 1. 加载基础数据（仅读取新文件）
    train_data_paths = ["./data/cmexam_dental_choice.jsonl"]
    all_data = []
    for path in train_data_paths:
        all_data.extend(read_jsonl(path))
    if len(all_data) == 0:
        raise ValueError("❌ 未加载到任何数据，请检查文件路径和格式")

    # 2. 加载老师模型（用于双目标蒸馏）
    print("📌 加载老师模型（Qwen2.5-7B-Instruct）...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    teacher_model.eval()

    # 3. 数据处理+增强
    prompts = []
    target_new = []
    subject = []
    invalid_count = 0

    for idx, item in enumerate(all_data):
        try:
            # 基础字段校验（适配新格式）
            required_fields = ["Question", "Options", "Answer"]
            if not all(f in item for f in required_fields):
                raise KeyError(f"缺少字段：{[f for f in required_fields if f not in item]}")
            
            # 提取核心信息
            title = item["Question"].strip()
            options_text = item["Options"].strip()
            original_answer = item["Answer"].strip().upper()
            
            # 解析选项
            options = parse_options(options_text)
            if len(options) < 2:
                raise ValueError(f"有效选项数不足（仅{len(options)}个）")
            if original_answer not in [opt[0] for opt in options]:
                raise ValueError(f"答案{original_answer}不在选项列表中")
            
            # 术语同义替换（数据增强）
            for term, synonyms in DENTAL_SYNONYM.items():
                if term in title and random.random() > 0.5:
                    title = title.replace(term, random.choice(synonyms))
            
            # 选项随机打乱（增强鲁棒性）
            shuffled_options = options.copy()
            random.shuffle(shuffled_options)
            
            # 重建选项文本和答案映射
            option_text = "\n".join([f"{opt[0]}. {opt[1]}" for opt in shuffled_options])
            # 找到打乱后的答案
            new_answer = ""
            for opt_char, _ in shuffled_options:
                if opt_char == original_answer:
                    new_answer = opt_char
                    break
            if not new_answer:
                new_answer = original_answer

            # 结构化Prompt（选择题专属）
            prompt = f"""指令：请回答以下牙科选择题，仅输出选项字母（如A/B/C/D/E），无需解释。
问题：{title}
选项：
{option_text}"""

            # 双目标蒸馏（老师答案+人工答案）
            teacher_ans = get_teacher_answer(prompt, teacher_model, teacher_tokenizer)
            target = f"{teacher_ans}|{new_answer}"  # 双目标格式

            prompts.append(prompt)
            target_new.append(target)
            subject.append("牙科选择题")

            # 难题增强（包含"最正确""以上均正确/不正确"的视为难题，权重×2）
            if "最正确" in title or "以上均正确" in options_text or "以上均不正确" in options_text:
                prompts.append(prompt)
                target_new.append(target)
                subject.append("牙科选择题")

        except Exception as e:
            invalid_count += 1
            print(f"⚠️ 第{idx+1}条数据处理失败：{e}")
            continue

    # 清理老师模型释放显存
    del teacher_model
    del teacher_tokenizer
    torch.cuda.empty_cache()

    print(f"✅ 数据处理完成：有效{len(prompts)}条 | 无效{invalid_count}条")
    if len(prompts) == 0:
        raise ValueError("❌ 无有效训练数据，请检查数据源格式")
    ground_truth = target_new
    return prompts, target_new, subject, ground_truth

# ===================== 检查点工具 =====================
def save_checkpoint(editor, optimizer, current_step, metrics, hparams, checkpoint_path):
    """保存检查点（适配7B模型）"""
    checkpoint = {
        "step": current_step,
        "lora_state_dict": editor.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "hparams": hparams.__dict__,
        "torch_dtype": TORCH_DTYPE,
        "device": DEVICE
    }
    temp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, temp_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    shutil.move(temp_path, checkpoint_path)
    print(f"✅ 检查点保存：{checkpoint_path}（步数：{current_step}）")

def load_latest_checkpoint(editor, checkpoint_dir):
    """加载最新检查点"""
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
    if not checkpoint_files:
        print("ℹ️ 无检查点，从头训练")
        return False, 0, None
    
    checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
    latest_ckpt = checkpoint_files[0]
    ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
    
    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        editor.model.load_state_dict(checkpoint["lora_state_dict"])
        resume_step = checkpoint["step"]
        optimizer_state = checkpoint["optimizer_state_dict"]
        print(f"✅ 加载检查点：{ckpt_path}（恢复步数：{resume_step}）")
        return True, resume_step, optimizer_state
    except Exception as e:
        print(f"⚠️ 加载检查点失败：{e}")
        return False, 0, None

# ===================== LoRA超参数（适配Qwen2.5-7B）=====================
def get_lora_hparams(resume_step=0):
    """定制化LoRA超参数（适配Qwen2.5-7B实际结构）"""
    total_steps = 20  # 选择题专属训练步数
    remaining_steps = max(0, total_steps - resume_step)
    
    hparams = LoRAHyperParams(
        lora_type="lora",
        layers=[],  # 空列表让PEFT自动搜索整个模型中的target_modules
        num_steps=remaining_steps,
        lr=1e-4,  # 降低学习率避免过拟合
        weight_decay=0.001,  # 增强泛化
        kl_factor=0.1,  # KL散度平衡
        norm_constraint=1.0,
        # Qwen2.5-7B官方配置（来自hparams/LoRA/qwen2.5-7b.yaml）
        target_modules=["q_proj", "v_proj"],
        rank=8,  # 平衡精度与显存
        lora_alpha=16,
        lora_dropout=0.02,  # 低dropout提升精准度
        device=DEVICE.split(":")[-1],
        alg_name="LoRA",
        model_name=MODEL_NAME
    )
    
    # 选择题专属配置（batch_size=1）
    hparams.torch_dtype = TORCH_DTYPE
    hparams.batch_size = 1  # 按要求设置为1
    hparams.max_length = 1024
    hparams.use_chat_template = False  # 关闭对话模板
    hparams.resume_step = resume_step
    
    return hparams

# ===================== 训练主逻辑 =====================
def main():
    # 1. 显存清理
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 2. 加载数据
    try:
        prompts, target_new, subject, ground_truth = load_dental_choice_data()
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        return
    
    # 3. 初始化编辑器
    resume_step = 0
    optimizer_state = None
    hparams = get_lora_hparams()
    
    if RESUME_TRAINING:
        temp_editor = BaseEditor.from_hparams(hparams)
        load_success, resume_step, optimizer_state = load_latest_checkpoint(temp_editor, CHECKPOINT_DIR)
        if load_success:
            hparams = get_lora_hparams(resume_step)
            editor = BaseEditor.from_hparams(hparams)
            editor.model.load_state_dict(temp_editor.model.state_dict())
        else:
            editor = BaseEditor.from_hparams(hparams)
    else:
        editor = BaseEditor.from_hparams(hparams)
    
    # 4. 通用优化
    editor.model.gradient_checkpointing_enable()  # 梯度检查点（通用显存优化）
    # 冻结非LoRA层（仅训练LoRA）
    for name, param in editor.model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    
    # 5. 开始训练
    print(f"\n🚀 开始训练Qwen2.5-7B牙科选择题蒸馏")
    print(f"ℹ️ 训练步数：{hparams.num_steps} | Batch：{hparams.batch_size} | 精度：{TORCH_DTYPE}")
    
    try:
        # 执行蒸馏训练
        metrics, edited_model, optimizer = editor.edit(
            prompts=prompts,
            target_new=target_new,
            subject=subject,
            ground_truth=ground_truth,
            keep_original_weight=True
        )
        
        # 保存最终检查点
        final_ckpt = os.path.join(CHECKPOINT_DIR, f"checkpoint_{resume_step + hparams.num_steps}.pth")
        save_checkpoint(editor, optimizer, resume_step + hparams.num_steps, metrics, hparams, final_ckpt)
        
    except KeyboardInterrupt:
        print("\n⚠️ 手动中断，保存当前状态...")
        interrupt_step = resume_step + (hparams.num_steps // 2)
        interrupt_ckpt = os.path.join(CHECKPOINT_DIR, f"checkpoint_{interrupt_step}.pth")
        save_checkpoint(editor, optimizer, interrupt_step, metrics if 'metrics' in locals() else {}, hparams, interrupt_ckpt)
        raise
    
    # 6. 保存最终模型
    print(f"\n💾 保存最终模型到：{OUTPUT_DIR}")
    editor.model.save_pretrained(OUTPUT_DIR)
    
    # 保存超参数
    hparams_path = os.path.join(OUTPUT_DIR, "lora_hparams.json")
    with open(hparams_path, "w", encoding="utf-8") as f:
        json.dump(hparams.__dict__, f, ensure_ascii=False, indent=4, default=json_serialize_hparams)
    
    # 7. 验证效果
    print("\n📊 验证模型效果...")
    test_prompt = """指令：请回答以下牙科选择题，仅输出选项字母（如A/B/C/D/E），无需解释。
问题：地图舌的舌背黏膜表现为
选项：
A.片状白斑
B.菌状乳头不明显
C.光滑的红色剥脱区
D.沟纹如脑回状
E.以上均不正确"""
    
    # 加载tokenizer验证
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    student_ans = infer_choice(editor.model, tokenizer, test_prompt)
    print(f"学生模型答案：{student_ans}（正确答案应为C）")
    print(f"\n✅ 训练完成！模型路径：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
# target_modules=["q_proj", "v_proj"]
