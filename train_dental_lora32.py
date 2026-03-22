import os
import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel


SYSTEM_PROMPT = (
    "你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，"
    "不要附带任何解释或空格。"
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DentalQADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        augment: bool = False,
        hard_set=None,
        hard_upsample: int = 1,
    ):
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

        # 对困难样本做上采样：仅复制训练题，不引入新数据。
        hard_set = hard_set or set()
        if hard_set and hard_upsample > 1:
            hard_samples = [x for x in self.data if x.get("Question", "") in hard_set]
            extra = hard_samples * (hard_upsample - 1)
            self.data.extend(extra)
            print(
                f"困难样本上采样: {len(hard_samples)} 条 x{hard_upsample} "
                f"=> 额外 +{len(extra)} 条"
            )

        print(f"加载牙科选择题数据完成，共 {len(self.data)} 条")

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

        prompt_prefix = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n问题：{question}\n选项：\n{options}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = prompt_prefix + f"{answer}<|im_end|>"

        inputs = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
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


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.1):
    """仅在有效答案 token 上蒸馏，避免全序列 KL 爆炸。"""
    shift_student = student_logits[:, :-1, :].contiguous()
    shift_teacher = teacher_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    ce_loss = F.cross_entropy(
        shift_student.view(-1, shift_student.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    # token-level KL: [B, T]
    teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)
    student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

    valid_mask = (shift_labels != -100).float()
    valid_count = valid_mask.sum().clamp_min(1.0)
    kl_loss = (token_kl * valid_mask).sum() / valid_count
    kl_loss = kl_loss * (temperature ** 2)

    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
    return total_loss, ce_loss.detach(), kl_loss.detach()


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

    model.eval()
    correct = 0
    wrongs = []

    for q, opts, ans in samples:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n问题：{q}\n选项：\n{opts}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        pred = extract_answer_char(gen)
        if pred == ans:
            correct += 1
        else:
            wrongs.append({"question": q, "gt": ans, "pred": pred, "gen": gen})

    acc = 100 * correct / len(samples) if samples else 0.0
    return acc, wrongs


def mine_hard_examples(teacher_model, student_model, tokenizer, data_path, device):
    """挖掘困难样本: teacher 对且 student 错 的题目，用于上采样。"""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    hard_q = set()
    teacher_model.eval()
    student_model.eval()

    for item in tqdm(samples, desc="挖掘困难样本"):
        q = item.get("Question", "")
        opts = item.get("Options", "")
        ans = item.get("Answer", "")
        if not q or not opts or not ans:
            continue

        prompt = (
            f"<|im_start|>system\\n{SYSTEM_PROMPT}\\n<|im_end|>\\n"
            f"<|im_start|>user\\n问题：{q}\\n选项：\\n{opts}\\n<|im_end|>\\n"
            f"<|im_start|>assistant\\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_t = teacher_model.generate(**inputs, max_new_tokens=4, do_sample=False)
            out_s = student_model.generate(**inputs, max_new_tokens=4, do_sample=False)

        pred_t = extract_answer_char(tokenizer.decode(out_t[0][inputs["input_ids"].size(1):], skip_special_tokens=True))
        pred_s = extract_answer_char(tokenizer.decode(out_s[0][inputs["input_ids"].size(1):], skip_special_tokens=True))
        if pred_t == ans and pred_s != ans:
            hard_q.add(q)

    print(f"困难样本挖掘完成: teacher对且student错 = {len(hard_q)} 条")
    teacher_model.train(False)
    student_model.train(False)
    return hard_q


def train_with_distillation(student_model, teacher_model, tokenizer, train_dataloader, optimizer, scheduler, device, hparams):
    student_model.train()
    teacher_model.eval()
    best_acc = hparams.best_val_acc
    accum_steps = max(1, hparams.gradient_accumulation_steps)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(hparams.num_epochs):
        print(f"\n🚀 开始训练第 {epoch + 1} 轮")
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        # alpha warmup: 先纯CE，再逐步打开KL
        if epoch < hparams.alpha_warmup_epochs:
            alpha_now = 0.0
        else:
            alpha_now = hparams.alpha
        print(f"当前蒸馏权重 alpha={alpha_now:.3f}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            loss, ce_val, kl_val = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=hparams.temperature,
                alpha=alpha_now,
            )
            loss = loss / accum_steps

            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{ce_val.item():.4f}",
                "kl": f"{kl_val.item():.4f}",
            })

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"第 {epoch + 1} 轮平均损失: {avg_loss:.4f}")

        if hparams.val_path:
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
                with open(os.path.join(hparams.output_dir, f"val_wrong_epoch{epoch + 1}.jsonl"), "w", encoding="utf-8") as wf:
                    for w in wrongs:
                        wf.write(json.dumps(w, ensure_ascii=False) + "\n")

    return student_model


def main():
    class DistillHyperParams:
        teacher_model_name: str = "./Qwen2.5-32B-Instruct"
        student_model_name: str = "./Qwen2.5-7B-Instruct"

        data_path: str = "./data/cmexam_dental_choice_train.jsonl"
        val_path: str = "./data/cmexam_dental_choice_val.jsonl"
        test_path: str = "./data/cmexam_dental_choice_test.jsonl"
        output_dir: str = "./dental_qwen2.5_7b_choice_lora_distill_from32"

        num_epochs: int = 4
        batch_size: int = 1
        gradient_accumulation_steps: int = 8
        learning_rate: float = 1e-4
        weight_decay: float = 0.01

        rank: int = 16
        lora_alpha: int = 32
        lora_dropout: float = 0.05
        target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"]

        temperature: float = 2.0
        alpha: float = 0.1
        alpha_warmup_epochs: int = 1

        max_length: int = 768
        device: int = 0
        augment: bool = False
        hard_upsample: int = 2
        seed: int = 42

        best_val_acc: float = 0.0
        best_ckpt_path: str = ""

    parser = argparse.ArgumentParser(description="Qwen2.5-32B(teacher) -> 7B(student) 蒸馏+LoRA")
    parser.add_argument("--teacher_model", type=str, help="教师模型路径")
    parser.add_argument("--student_model", type=str, help="学生模型路径")
    parser.add_argument("--data_path", type=str, help="训练集路径")
    parser.add_argument("--val_path", type=str, help="验证集路径")
    parser.add_argument("--test_path", type=str, help="测试集路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--num_epochs", type=int, help="训练轮数")
    parser.add_argument("--batch_size", type=int, help="批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="梯度累积")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--rank", type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--temperature", type=float, help="蒸馏温度")
    parser.add_argument("--alpha", type=float, help="KL权重")
    parser.add_argument("--alpha_warmup_epochs", type=int, help="前多少轮仅CE(不使用KL)")
    parser.add_argument("--hard_upsample", type=int, help="困难样本上采样倍数")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--max_length", type=int, help="序列最大长度")
    parser.add_argument("--augment", action="store_true", help="启用简单增强")
    args = parser.parse_args()

    hparams = DistillHyperParams()
    if args.teacher_model:
        hparams.teacher_model_name = args.teacher_model
    if args.student_model:
        hparams.student_model_name = args.student_model
    if args.data_path:
        hparams.data_path = args.data_path
    if args.val_path is not None:
        hparams.val_path = args.val_path
    if args.test_path is not None:
        hparams.test_path = args.test_path
    if args.output_dir:
        hparams.output_dir = args.output_dir
    if args.num_epochs is not None:
        hparams.num_epochs = args.num_epochs
    if args.batch_size is not None and args.batch_size > 0:
        hparams.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps > 0:
        hparams.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        hparams.learning_rate = args.learning_rate
    if args.rank is not None:
        hparams.rank = args.rank
    if args.lora_alpha is not None:
        hparams.lora_alpha = args.lora_alpha
    if args.temperature is not None and args.temperature > 0:
        hparams.temperature = args.temperature
    if args.alpha is not None and 0 <= args.alpha <= 1:
        hparams.alpha = args.alpha
    if args.alpha_warmup_epochs is not None and args.alpha_warmup_epochs >= 0:
        hparams.alpha_warmup_epochs = args.alpha_warmup_epochs
    if args.hard_upsample is not None and args.hard_upsample >= 1:
        hparams.hard_upsample = args.hard_upsample
    if args.seed is not None:
        hparams.seed = args.seed
    if args.max_length is not None and args.max_length > 0:
        hparams.max_length = args.max_length
    if args.augment:
        hparams.augment = True

    print("训练配置:", vars(hparams))
    os.makedirs(hparams.output_dir, exist_ok=True)
    set_seed(hparams.seed)

    device = torch.device(f"cuda:{hparams.device}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        hparams.student_model_name,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        trust_remote_code=True,
    )

    dataset = DentalQADataset(
        hparams.data_path,
        tokenizer,
        max_length=hparams.max_length,
        augment=hparams.augment,
    )
    train_dataloader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True)

    print("加载32B教师模型...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        hparams.teacher_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    teacher_model.eval()

    print("加载7B学生模型并挂载LoRA...")
    student_model = AutoModelForCausalLM.from_pretrained(
        hparams.student_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    student_model.config.use_cache = False

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=hparams.rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        target_modules=hparams.target_modules,
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model = student_model.to(device)
    student_model.print_trainable_parameters()

    # 先挖掘困难样本，再构建训练集（teacher对且student错）
    hard_set = mine_hard_examples(
        teacher_model, student_model, tokenizer, hparams.data_path, device
    )
    dataset = DentalQADataset(
        hparams.data_path,
        tokenizer,
        max_length=hparams.max_length,
        augment=hparams.augment,
        hard_set=hard_set,
        hard_upsample=hparams.hard_upsample,
    )
    train_dataloader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )

    accum_steps_for_sched = max(1, hparams.gradient_accumulation_steps)
    total_steps = (len(train_dataloader) + accum_steps_for_sched - 1) // accum_steps_for_sched * hparams.num_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"调度器: cosine warmup, total_steps={total_steps}, warmup={warmup_steps}")

    trained_model = train_with_distillation(
        student_model,
        teacher_model,
        tokenizer,
        train_dataloader,
        optimizer,
        scheduler,
        device,
        hparams,
    )

    print("保存最终模型...")
    trained_model.save_pretrained(hparams.output_dir)
    tokenizer.save_pretrained(hparams.output_dir)
    print(f"✅ 训练完成，模型保存到: {hparams.output_dir}")

    if hparams.test_path:
        print("在测试集上评估...")
        model_for_test = trained_model
        if hparams.best_ckpt_path and os.path.isdir(hparams.best_ckpt_path):
            print(f"使用最佳检查点评估: {hparams.best_ckpt_path}")
            base_for_eval = AutoModelForCausalLM.from_pretrained(
                hparams.student_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)
            model_for_test = PeftModel.from_pretrained(base_for_eval, hparams.best_ckpt_path).to(device)
            model_for_test.eval()

        test_acc, test_wrongs = evaluate_generation(model_for_test, tokenizer, hparams.test_path, device)
        print(f"测试集准确率: {test_acc:.2f}%")

        with open(os.path.join(hparams.output_dir, "test_wrong.jsonl"), "w", encoding="utf-8") as wf:
            for w in test_wrongs:
                wf.write(json.dumps(w, ensure_ascii=False) + "\n")
        print(f"测试集错误样本已保存: {hparams.output_dir}/test_wrong.jsonl")


if __name__ == "__main__":
    main()
