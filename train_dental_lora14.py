import os
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ===================== 自定义数据集类 =====================
class DentalQADataset(Dataset):
    """牙科QA数据集（选择题格式）"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024, augment: bool = False):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

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
        question = item.get("Question", "")
        options = item.get("Options", "")
        answer = item.get("Answer", "")

        if self.augment and random.random() < 0.3:
            prefixes = ["请回答：", "以下问题：", "问题是："]
            suffixes = ["。", "?", ""]
            question = random.choice(prefixes) + question + random.choice(suffixes)

        prompt_prefix = f"<|im_start|>system\n你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。\n<|im_end|>\n<|im_start|>user\n问题：{question}\n选项：\n{options}\n<|im_end|>\n<|im_start|>assistant\n"
        full_text = prompt_prefix + f"{answer}<|im_end|>"

        inputs = self.tokenizer(
            full_text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        prefix_enc = self.tokenizer(prompt_prefix, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])

        labels = inputs["input_ids"].squeeze().clone()
        labels[:prefix_len] = -100
        labels[inputs["attention_mask"].squeeze() == 0] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
        }


# ===================== 蒸馏损失函数 =====================
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """蒸馏损失（含 causal LM label shift）"""
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    return alpha * kl_loss + (1 - alpha) * ce_loss


# ===================== 实用函数 =====================
def extract_answer_char(text: str) -> str:
    for ch in text.strip().upper():
        if ch in ["A", "B", "C", "D", "E"]:
            return ch
    return ""


def evaluate_generation(model, tokenizer, file_path, device, max_new_tokens=4):
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
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        pred = extract_answer_char(gen)
        if pred == ans:
            correct += 1
        else:
            wrongs.append({"question": q, "options": opts, "gt": ans, "pred": pred, "gen": gen})
    acc = 100 * correct / len(samples) if samples else 0.0
    return acc, wrongs


# ===================== 训练循环 =====================
def train_with_distillation(student_model, teacher_model, tokenizer, train_dataloader,
                            optimizer, scheduler, device, num_epochs=3, hparams=None):
    student_model.train()
    teacher_model.eval()
    best_acc = hparams.best_val_acc if hparams is not None else 0.0
    accum_steps = max(1, hparams.gradient_accumulation_steps) if hparams is not None else 1

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        try:
            print(f"\n🚀 开始训练第 {epoch + 1} 轮")
            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits

                # 教师模型在 CPU 上推理（14B×2 无法同时放入单卡 GPU）
                with torch.no_grad():
                    t_input_ids = input_ids.to(teacher_model.device)
                    t_attn = attention_mask.to(teacher_model.device)
                    teacher_outputs = teacher_model(input_ids=t_input_ids, attention_mask=t_attn)
                    teacher_logits = teacher_outputs.logits.to(device)

                loss = distillation_loss(
                    student_logits, teacher_logits, labels,
                    temperature=(hparams.temperature if hparams is not None else 2.0),
                    alpha=(hparams.alpha if hparams is not None else 0.5),
                ) / accum_steps

                loss.backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_dataloader):
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / len(train_dataloader)
            print(f"第 {epoch + 1} 轮平均损失: {avg_loss:.4f}")

            if hparams is not None and hparams.val_path:
                val_acc, wrongs = evaluate_generation(student_model, tokenizer, hparams.val_path, device)
                print(f"第 {epoch + 1} 轮验证准确率: {val_acc:.2f}%")
                if val_acc > best_acc or (not hparams.best_ckpt_path):
                    best_acc = val_acc
                    hparams.best_val_acc = val_acc
                    save_dir = os.path.join(hparams.output_dir, "best")
                    os.makedirs(save_dir, exist_ok=True)
                    student_model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    hparams.best_ckpt_path = save_dir
                    print(f"保存当前最佳模型到 {save_dir}")
                    with open(os.path.join(hparams.output_dir, f"val_wrong_epoch{epoch+1}.jsonl"), "w", encoding="utf-8") as wf:
                        for w in wrongs:
                            wf.write(json.dumps(w, ensure_ascii=False) + "\n")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("\n[OOM] CUDA 内存不足。尝试减小 --batch_size 或增大 --gradient_accumulation_steps。")
                torch.cuda.empty_cache()
                raise
            else:
                raise

    return student_model


# ===================== 主函数 =====================
def main():
    print("🚀 开始 Qwen2.5-14B-Instruct 蒸馏+LoRA 微调牙科选择题模型")

    class DentalLoRAHyperParams:
        # LoRA
        lora_type: str = "lora"
        layers: list = []
        rank: int = 16
        lora_alpha: int = 32
        lora_dropout: float = 0.05
        target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # 训练（14B 教师在CPU，batch=1 + grad_acc=16 等效 batch=16）
        num_epochs: int = 4
        batch_size: int = 1
        gradient_accumulation_steps: int = 16
        learning_rate: float = 1e-4
        weight_decay: float = 0.01

        # 蒸馏
        temperature: float = 2.0
        alpha: float = 0.5

        # 路径 —— 14B
        model_name: str = "./Qwen2.5-14B-Instruct"
        data_path: str = "./data/cmexam_dental_choice_train.jsonl"
        output_dir: str = "./dental_qwen2.5_14b_choice_lora_distill"

        device: int = 0
        model_parallel: bool = False
        augment: bool = False

        val_path: str = ""
        test_path: str = ""

        best_val_acc: float = 0.0
        best_ckpt_path: str = ""

    parser = argparse.ArgumentParser(description="训练 14B 牙科QA模型（蒸馏 + LoRA）")
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--val_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    hparams = DentalLoRAHyperParams()
    if args.num_epochs is not None:      hparams.num_epochs = args.num_epochs
    if args.batch_size is not None and args.batch_size > 0:
        hparams.batch_size = args.batch_size
    if args.learning_rate is not None:   hparams.learning_rate = args.learning_rate
    if args.rank is not None:            hparams.rank = args.rank
    if args.lora_alpha is not None:      hparams.lora_alpha = args.lora_alpha
    if args.temperature is not None and args.temperature > 0:
        hparams.temperature = args.temperature
    if args.alpha is not None and 0.0 <= args.alpha <= 1.0:
        hparams.alpha = args.alpha
    if args.data_path is not None:       hparams.data_path = args.data_path
    if args.output_dir is not None:      hparams.output_dir = args.output_dir
    if args.augment:                     hparams.augment = True
    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps > 0:
        hparams.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.val_path is not None:        hparams.val_path = args.val_path
    if args.test_path is not None:       hparams.test_path = args.test_path

    print("训练配置:", vars(hparams))

    device = torch.device(f"cuda:{hparams.device}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.model_name,
        eos_token='<|endoftext|>', pad_token='<|endoftext|>', unk_token='<|endoftext|>',
        trust_remote_code=True,
    )

    # 2. 数据
    print("加载数据...")
    dataset = DentalQADataset(hparams.data_path, tokenizer, augment=hparams.augment)
    train_dataloader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True)

    # 3. 教师模型 —— 放 CPU（14B bf16 约28GB，两个14B无法同时放入96GB GPU）
    print("加载教师模型到CPU...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        hparams.model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="cpu",
    )
    teacher_model.eval()

    # 4. 学生模型 + LoRA → GPU
    print("加载学生模型并应用LoRA...")
    student_model = AutoModelForCausalLM.from_pretrained(
        hparams.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM", inference_mode=False,
        r=hparams.rank, lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout, target_modules=hparams.target_modules,
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model = student_model.to(device)
    student_model.print_trainable_parameters()

    # 5. 优化器 + cosine warmup
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    accum_for_sched = max(1, hparams.gradient_accumulation_steps)
    total_steps = (len(train_dataloader) + accum_for_sched - 1) // accum_for_sched * hparams.num_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"调度器: cosine warmup, total_steps={total_steps}, warmup={warmup_steps}")

    # 6. 训练
    print("开始蒸馏训练...")
    trained_model = train_with_distillation(
        student_model, teacher_model, tokenizer, train_dataloader,
        optimizer, scheduler, device, num_epochs=hparams.num_epochs, hparams=hparams,
    )

    # 7. 保存
    print("保存模型...")
    os.makedirs(hparams.output_dir, exist_ok=True)
    trained_model.save_pretrained(hparams.output_dir)
    tokenizer.save_pretrained(hparams.output_dir)
    print(f"✅ 14B 蒸馏训练完成！模型保存至：{hparams.output_dir}")

    # 8. 测试集评估
    if hparams.test_path:
        print("在测试集上进行评估...")
        model_for_test = trained_model
        if hparams.best_ckpt_path and os.path.isdir(hparams.best_ckpt_path):
            print(f"检测到最佳检查点: {hparams.best_ckpt_path}，使用最佳模型进行测试评估")
            from peft import PeftModel
            base_for_eval = AutoModelForCausalLM.from_pretrained(
                hparams.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            model_for_test = PeftModel.from_pretrained(base_for_eval, hparams.best_ckpt_path).to(device)
            model_for_test.eval()

        test_acc, test_wrongs = evaluate_generation(model_for_test, tokenizer, hparams.test_path, device)
        print(f"测试集准确率: {test_acc:.2f}%")
        with open(os.path.join(hparams.output_dir, "test_wrong.jsonl"), "w", encoding="utf-8") as wf:
            for w in test_wrongs:
                wf.write(json.dumps(w, ensure_ascii=False) + "\n")
        print(f"测试集错误样本已记录在 {hparams.output_dir}/test_wrong.jsonl")


if __name__ == "__main__":
    main()
