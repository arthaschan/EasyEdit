"""
Qwen2.5-14B-Instruct 靶向加权 SFT + LoRA 训练脚本

策略：全量 SFT + 错题加权
- 用全部训练题做 SFT（保持模型一致性）
- 对 14B 答错的题目做上采样（重点修补薄弱知识）
- 保持原始 prompt 格式（仅输出选项字母）
- 纯 CE 损失，无 KL 蒸馏
"""
import os
import gc
import json
import random
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SYSTEM_PROMPT = '你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。'


def extract_answer_char(text: str) -> str:
    for ch in text.strip().upper():
        if ch in "ABCDE":
            return ch
    return ""


# ==================== Phase 1: 诊断薄弱点 ====================

def diagnose_model(model_path, data_path, device, exclude_stems=None):
    """用模型跑训练集，分出正确/错误两组"""
    print(f"\n{'='*60}")
    print(f"Phase 1: 诊断模型薄弱点")
    print(f"  模型: {model_path}")
    print(f"  数据: {data_path}")
    print(f"{'='*60}")

    exclude = exclude_stems or set()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device,
    )
    model.eval()

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                if item.get("Question", "").strip() not in exclude:
                    samples.append(item)
    print(f"加载 {len(samples)} 条训练题目（已排除测试/验证题）")

    correct_items = []
    wrong_items = []

    for item in tqdm(samples, desc="模型推理"):
        q, opts, gt = item["Question"], item["Options"], item["Answer"]
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n问题：{q}\n选项：\n{opts}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        pred = extract_answer_char(gen)

        if pred == gt:
            correct_items.append(item)
        else:
            wrong_items.append(item)

    print(f"\n诊断结果: 正确 {len(correct_items)}, 错误 {len(wrong_items)}, "
          f"准确率 {100*len(correct_items)/len(samples):.1f}%")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("诊断模型已卸载\n")
    return correct_items, wrong_items


# ==================== Dataset ====================

class WeightedSFTDataset(Dataset):
    """全量 SFT + 错题加权数据集"""

    def __init__(self, correct_items, wrong_items, tokenizer, max_length=512,
                 wrong_upsample=3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # 正确题：各加1次
        for item in correct_items:
            self.data.append(item)
        print(f"正确题: {len(correct_items)} 条 × 1")

        # 错题：上采样
        for _ in range(wrong_upsample):
            for item in wrong_items:
                self.data.append(item)
        print(f"错误题: {len(wrong_items)} 条 × {wrong_upsample} = {len(wrong_items) * wrong_upsample} 条")

        random.shuffle(self.data)
        print(f"训练集总计: {len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        q, opts, ans = item["Question"], item["Options"], item["Answer"]

        prompt_prefix = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n问题：{q}\n选项：\n{opts}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = prompt_prefix + f"{ans}<|im_end|>"

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


# ==================== 评估 ====================

def evaluate(model, tokenizer, file_path, device):
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line.strip())
                q, opts, ans = d.get("Question", ""), d.get("Options", ""), d.get("Answer", "")
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
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        pred = extract_answer_char(gen)
        if pred == ans:
            correct += 1
        else:
            wrongs.append({"question": q, "gt": ans, "pred": pred, "gen": gen[:80]})

    acc = 100 * correct / len(samples) if samples else 0.0
    return acc, wrongs


# ==================== 训练 ====================

def train(args, correct_items, wrong_items):
    print(f"\n{'='*60}")
    print(f"Phase 2: 全量 SFT + 错题加权 LoRA 训练")
    print(f"  模型: {args.model_path}")
    print(f"  正确题: {len(correct_items)}, 错题: {len(wrong_items)}")
    print(f"  输出: {args.output_dir}")
    print(f"{'='*60}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        eos_token='<|endoftext|>', pad_token='<|endoftext|>', unk_token='<|endoftext|>',
        trust_remote_code=True,
    )

    dataset = WeightedSFTDataset(
        correct_items, wrong_items, tokenizer,
        max_length=args.max_length, wrong_upsample=args.wrong_upsample,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM", inference_mode=False,
        r=args.rank, lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    accum = max(1, args.gradient_accumulation_steps)
    total_steps = (len(dataloader) + accum - 1) // accum * args.num_epochs
    warmup = max(1, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps,
    )
    print(f"调度器: cosine warmup, total={total_steps}, warmup={warmup}")

    best_acc = 0.0
    best_ckpt = ""

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accum
            loss.backward()

            if (step + 1) % accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item() * accum:.4f}"})

        avg_loss = epoch_loss / len(dataloader) * accum
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")

        # 验证
        if args.val_path:
            val_acc, val_wrongs = evaluate(model, tokenizer, args.val_path, device)
            print(f"Epoch {epoch+1} 验证准确率: {val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                save_dir = os.path.join(args.output_dir, "best")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                best_ckpt = save_dir
                print(f"  → 保存最佳模型到 {save_dir} (acc={val_acc:.2f}%)")
                with open(os.path.join(args.output_dir, f"val_wrong_epoch{epoch+1}.jsonl"), "w", encoding="utf-8") as wf:
                    for w in val_wrongs:
                        wf.write(json.dumps(w, ensure_ascii=False) + "\n")
            # 每个epoch都做测试集评估（方便观察趋势）
            if args.test_path:
                test_acc, _ = evaluate(model, tokenizer, args.test_path, device)
                print(f"Epoch {epoch+1} 测试集准确率: {test_acc:.2f}%")

    # 保存最终模型
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n最终模型保存至: {args.output_dir}")

    # 最终测试
    if args.test_path:
        print("\n最终测试集评估...")
        if best_ckpt:
            print(f"使用最佳检查点: {best_ckpt}")
            base = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            test_model = PeftModel.from_pretrained(base, best_ckpt).to(device)
            test_model.eval()
        else:
            test_model = model

        test_acc, test_wrongs = evaluate(test_model, tokenizer, args.test_path, device)
        print(f"✅ 测试集准确率: {test_acc:.2f}%")

        with open(os.path.join(args.output_dir, "test_wrong.jsonl"), "w", encoding="utf-8") as wf:
            for w in test_wrongs:
                wf.write(json.dumps(w, ensure_ascii=False) + "\n")
        with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as rf:
            json.dump({"test_accuracy": test_acc, "best_val_accuracy": best_acc,
                        "test_wrong_count": len(test_wrongs),
                        "correct_items": len(correct_items),
                        "wrong_items": len(wrong_items)}, rf, ensure_ascii=False, indent=2)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="14B 全量SFT + 错题加权 LoRA 训练")
    parser.add_argument("--model_path", type=str, default="./Qwen2.5-14B-Instruct")
    parser.add_argument("--train_data", type=str, default="./data/cmexam_dental_choice_train.jsonl")
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--skip_diagnose", action="store_true", help="跳过诊断，直接用全量数据训练")
    # 训练超参
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--wrong_upsample", type=int, default=3, help="错题上采样倍数")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./dental_qwen2.5_14b_targeted_sft")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen2.5-14B-Instruct 全量SFT + 错题加权 LoRA")
    print(f"  模型: {args.model_path}")
    print("=" * 60)

    # 收集测试/验证题干，防泄露
    exclude_stems = set()
    for p in [args.test_path, args.val_path]:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        stem = item.get("Question", "").strip()
                        if stem:
                            exclude_stems.add(stem)

    if not args.skip_diagnose:
        dev = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        correct_items, wrong_items = diagnose_model(
            args.model_path, args.train_data, dev, exclude_stems=exclude_stems
        )
    else:
        # 跳过诊断，全部当正确题训练（upsample=1生效）
        correct_items = []
        wrong_items = []
        with open(args.train_data, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    if item.get("Question", "").strip() not in exclude_stems:
                        correct_items.append(item)
        print(f"跳过诊断，全部 {len(correct_items)} 条作为正确题训练")

    train(args, correct_items, wrong_items)


if __name__ == "__main__":
    main()
