import os
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path

import os
# 禁用 wandb 避免API key问题（之前用于绕过缺失问题，现在可以安全注释）
# os.environ['WANDB_API_KEY'] = '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'
# 导入EasyEdit框架组件
# from easyeditor.dataset.counterfact import CounterFactDataset  # 复用数据集基类
# from easyeditor.util.hparams import HyperParams
# from easyeditor.models.lora.lora_hparams import LoRAHyperParams

# ===================== 自定义数据集类（基于EasyEdit） =====================
class DentalQADataset(Dataset):
    """
    基于EasyEdit CounterFactDataset的牙科QA数据集
    """
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024, augment: bool = False):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment  # whether to apply simple text augmentations

        # 加载JSONL格式的数据
        self.data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        print(f"加载牙科QA数据集完成，共 {len(self.data)} 条记录")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建对话格式的prompt（与deploy_dental_robot7.py保持一致）
        question = item.get("Question", "")
        options = item.get("Options", "")
        answer = item.get("Answer", "")

        # 简单数据增强：在 question 前后添加随机短语
        if self.augment and random.random() < 0.3:
            prefixes = ["请回答：", "以下问题：", "问题是："]
            suffixes = ["。", "?", ""]
            question = random.choice(prefixes) + question + random.choice(suffixes)

        prompt = f"<|im_start|>system\n你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。\n<|im_end|>\n<|im_start|>user\n问题：{question}\n选项：\n{options}\n<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": inputs["input_ids"].squeeze()  # Causal LM的labels与input_ids相同
        }

# ===================== 蒸馏损失函数 =====================
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """
    计算蒸馏损失：alpha * KL(teacher_logits, student_logits) + (1-alpha) * CE(student_logits, labels)
    """
    # 学生模型的交叉熵损失
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=-100)

    # 蒸馏损失：KL散度
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    # 组合损失
    loss = alpha * kl_loss + (1 - alpha) * ce_loss
    return loss

# ===================== 实用函数 =====================
def extract_answer_char(text: str) -> str:
    """从生成文本中提取第一个 A-E 字母"""
    for ch in text.strip().upper():
        if ch in ["A", "B", "C", "D", "E"]:
            return ch
    return ""


def evaluate_generation(model, tokenizer, file_path, device, max_new_tokens=4):
    """对指定 jsonl 文件进行批量生成评估，返回准确率和错误列表"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            q = data.get("Question", "")
            opts = data.get("Options", "")
            ans = data.get("Answer", "")
            if q and opts and ans:
                samples.append((q, opts, ans))
    correct = 0
    wrongs = []
    model.eval()
    for q, opts, ans in samples:
        prompt = f"<|im_start|>system\n你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。\n<|im_end|>\n<|im_start|>user\n问题：{q}\n选项：\n{opts}\n<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        pred = extract_answer_char(gen)
        if pred == ans:
            correct += 1
        else:
            wrongs.append({"question": q, "options": opts, "gt": ans, "pred": pred, "gen": gen})
    acc = 100 * correct / len(samples) if samples else 0.0
    return acc, wrongs


# ===================== 自定义训练循环 =====================
def train_with_distillation(student_model, teacher_model, tokenizer, train_dataloader, optimizer, scheduler, device, num_epochs=3, hparams=None):
    """使用蒸馏的自定义训练循环，可以在每轮后执行验证
    若出现OOM会捕获并给出建议。
    教师模型可能驻留在CPU，运行时会将输入转到CPU。
    """
    student_model.train()
    teacher_model.eval()
    best_acc = hparams.best_val_acc if hparams is not None else 0.0

    for epoch in range(num_epochs):
        try:
            print(f"\n🚀 开始训练第 {epoch + 1} 轮")
            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # 学生模型前向传播
                student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits

                # 教师模型前向传播 (在CPU)
                with torch.no_grad():
                    t_input_ids = input_ids.to(teacher_model.device)
                    t_attn = attention_mask.to(teacher_model.device)
                    teacher_outputs = teacher_model(input_ids=t_input_ids, attention_mask=t_attn)
                    teacher_logits = teacher_outputs.logits.to(device)

                # 计算蒸馏损失，并按梯度累积比例缩放
                accum_steps = hparams.gradient_accumulation_steps if hparams is not None else 1
                loss = distillation_loss(student_logits, teacher_logits, labels) / accum_steps

                # 反向传播 + 累积
                loss.backward()
                step = progress_bar.n  # current batch index
                if (step + 1) % accum_steps == 0:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / len(train_dataloader)
            print(f"第 {epoch + 1} 轮平均损失: {avg_loss:.4f}")

            # 如果提供了验证集则在每轮后评估
            if hparams is not None and hparams.val_path:
                val_acc, wrongs = evaluate_generation(student_model, tokenizer, hparams.val_path, device)
                print(f"第 {epoch + 1} 轮验证准确率: {val_acc:.2f}%")
                # 比较并保存最优模型
                if val_acc > best_acc:
                    best_acc = val_acc
                    hparams.best_val_acc = val_acc
                    save_dir = os.path.join(hparams.output_dir, "best")
                    os.makedirs(save_dir, exist_ok=True)
                    student_model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    print(f"保存当前最佳模型到 {save_dir}")
                    # 记录错误样本
                    with open(os.path.join(hparams.output_dir, f"val_wrong_epoch{epoch+1}.jsonl"), "w", encoding="utf-8") as wf:
                        for w in wrongs:
                            wf.write(json.dumps(w, ensure_ascii=False) + "\n")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("\n[OOM] CUDA 内存不足。尝试减小 --batch_size 或使用 --gradient_accumulation_steps。")
                torch.cuda.empty_cache()
                raise
            else:
                raise

    return student_model 
 
# ===================== 主函数 =====================
def main():
    print("🚀 开始基于EasyEdit框架的蒸馏+LoRA微调牙科选择题模型")

    # 使用简化的配置类（避免EasyEdit依赖）
    class DentalLoRAHyperParams:
        """牙科QA训练配置"""
        # LoRA配置
        lora_type: str = "lora"
        layers: list = []
        rank: int = 16              # 增大秩
        lora_alpha: int = 32        # 增大 alpha
        lora_dropout: float = 0.05
        target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # 训练配置
        num_epochs: int = 5         # 提高训练轮次
        batch_size: int = 4         # 默认batch较小以降低显存占用
        gradient_accumulation_steps: int = 1  # 可选梯度累积
        learning_rate: float = 2e-4
        weight_decay: float = 0.01

        # 蒸馏配置
        temperature: float = 2.0
        alpha: float = 0.5  # 蒸馏权重

        # 路径配置
        model_name: str = "./Qwen2.5-7B-Instruct"
        data_path: str = "./data/cmexam_dental_choice_train.jsonl"  # 使用训练集
        output_dir: str = "./dental_qwen2.5_7b_choice_lora_distill_easyedit"

        # 设备配置
        device: int = 0
        model_parallel: bool = False
        augment: bool = False  # 是否执行简易数据增强

        # 可选的验证/测试集路径
        val_path: str = ""  # 训练时提供可进行每轮验证
        test_path: str = ""

        # 内部跟踪
        best_val_acc: float = 0.0

    # 解析命令行参数并创建配置实例
    parser = argparse.ArgumentParser(description="训练牙科QA模型（蒸馏 + LoRA）")
    parser.add_argument("--num_epochs", type=int, help="训练轮数")
    parser.add_argument("--batch_size", type=int, help="批量大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--rank", type=int, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--augment", action="store_true", help="启用简单的数据增强")
    parser.add_argument("--data_path", type=str, help="数据路径")
    parser.add_argument("--val_path", type=str, help="验证集 jsonl 路径")
    parser.add_argument("--test_path", type=str, help="测试集 jsonl 路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    args = parser.parse_args()

    hparams = DentalLoRAHyperParams()
    # 用命令行参数覆盖默认值
    if args.num_epochs is not None:
        hparams.num_epochs = args.num_epochs
    if args.batch_size is not None:
        hparams.batch_size = args.batch_size
    if args.learning_rate is not None:
        hparams.learning_rate = args.learning_rate
    if args.rank is not None:
        hparams.rank = args.rank
    if args.lora_alpha is not None:
        hparams.lora_alpha = args.lora_alpha
    if args.data_path is not None:
        hparams.data_path = args.data_path
    if args.output_dir is not None:
        hparams.output_dir = args.output_dir
    if args.augment:
        hparams.augment = True
    if args.batch_size is not None and args.batch_size>0:
        hparams.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps>1:
        hparams.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.val_path is not None:
        hparams.val_path = args.val_path
    if args.test_path is not None:
        hparams.test_path = args.test_path

    # 打印配置以便调试
    print("训练配置:", vars(hparams))

    device = torch.device(f"cuda:{hparams.device}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载tokenizer（复用EasyEdit的tokenizer设置）
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 按照EasyEdit的Qwen设置
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.model_name,
        eos_token='<|endoftext|>',
        pad_token='<|endoftext|>',
        unk_token='<|endoftext|>',
        trust_remote_code=True
    )

    # 2. 加载和处理数据（使用自定义数据集类）
    print("加载数据...")
    dataset = DentalQADataset(hparams.data_path, tokenizer, augment=hparams.augment)
    train_dataloader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True)

    # 3. 加载教师模型（放到CPU以节省GPU显存）
    print("加载教师模型到CPU...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        hparams.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu"
    )
    teacher_model.eval()  # 教师模型设为评估模式

    # 4. 加载学生模型并应用LoRA（复用EasyEdit的LoRA配置）
    print("加载学生模型并应用LoRA...")
    student_model = AutoModelForCausalLM.from_pretrained(
        hparams.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 使用EasyEdit风格的LoRA配置
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=hparams.rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        target_modules=hparams.target_modules
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model = student_model.to(device)
    student_model.print_trainable_parameters()  # 显示可训练参数

    # 5. 优化器和调度器
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay
    )
    scheduler = None  # 可以添加学习率调度器

    # 6. 开始蒸馏训练
    print("开始蒸馏训练...")
    trained_model = train_with_distillation(
        student_model, teacher_model, tokenizer, train_dataloader,
        optimizer, scheduler, device, num_epochs=hparams.num_epochs
    )

    # 7. 保存模型
    print("保存模型...")
    os.makedirs(hparams.output_dir, exist_ok=True)
    trained_model.save_pretrained(hparams.output_dir)
    tokenizer.save_pretrained(hparams.output_dir)

    print(f"✅ 基于EasyEdit框架的蒸馏训练完成！模型保存至：{hparams.output_dir}")

    # 如果有测试集路径，则用最终模型进行一次评估
    if hparams.test_path:
        print("在测试集上进行评估...")
        test_acc, test_wrongs = evaluate_generation(trained_model, tokenizer, hparams.test_path, device)
        print(f"测试集准确率: {test_acc:.2f}%")
        with open(os.path.join(hparams.output_dir, "test_wrong.jsonl"), "w", encoding="utf-8") as wf:
            for w in test_wrongs:
                wf.write(json.dumps(w, ensure_ascii=False) + "\n")
        print(f"测试集错误样本已记录在 {hparams.output_dir}/test_wrong.jsonl")

if __name__ == "__main__":
    main()