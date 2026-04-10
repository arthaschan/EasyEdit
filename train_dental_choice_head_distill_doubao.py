import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def set_global_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_answer_char(text: str) -> str:
    for ch in text.strip().upper():
        if ch in OPTION_LETTERS:
            return ch
    return ""


def evaluate_generation(model, tokenizer, file_path, device, max_new_tokens=4):
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get("Question", "")
            opts = row.get("Options", "")
            ans = row.get("Answer", "")
            if q and opts and ans:
                samples.append((q, opts, ans))

    correct = 0
    model.eval()
    for q, opts, ans in samples:
        prompt = (
            "<|im_start|>system\n"
            "你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"问题：{q}\n选项：\n{opts}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        pred = extract_answer_char(gen)
        if pred == ans:
            correct += 1
    return 100.0 * correct / len(samples) if samples else 0.0


class DentalChoiceHeadDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024, default_distill_mask: int = 1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.default_distill_mask = 1 if int(default_distill_mask) != 0 else 0
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        print(f"[DATA] loaded={len(self.data)} from {data_path}")

    def __len__(self):
        return len(self.data)

    def _build_teacher_dist(self, row, gt_letter: str):
        raw = row.get("TeacherDist", None)
        probs = torch.zeros(len(OPTION_LETTERS), dtype=torch.float32)
        if isinstance(raw, dict):
            for i, ch in enumerate(OPTION_LETTERS):
                try:
                    probs[i] = float(raw.get(ch, 0.0))
                except Exception:
                    probs[i] = 0.0
        if float(probs.sum().item()) <= 0.0 and gt_letter in OPTION_LETTERS:
            probs[OPTION_LETTERS.index(gt_letter)] = 1.0
        probs = torch.clamp(probs, min=0.0)
        s = float(probs.sum().item())
        if s <= 0.0:
            probs[0] = 1.0
            s = 1.0
        return probs / s

    def __getitem__(self, idx):
        row = self.data[idx]
        q = str(row.get("Question", ""))
        opts = row.get("Options", "")
        ans = str(row.get("Answer", "")).strip().upper()
        if ans not in OPTION_LETTERS:
            ans = "A"

        prompt_prefix = (
            "<|im_start|>system\n"
            "你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"问题：{q}\n选项：\n{opts}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        text = prompt_prefix + f"{ans}<|im_end|>"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        prefix_enc = self.tokenizer(prompt_prefix, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])

        labels = enc["input_ids"].squeeze().clone()
        labels[:prefix_len] = -100
        labels[enc["attention_mask"].squeeze() == 0] = -100

        distill_mask = 1 if str(row.get("SelectiveSource", "")).strip() == "clean_teacher" else self.default_distill_mask

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels,
            "teacher_dist": self._build_teacher_dist(row, ans),
            "gt_option": torch.tensor(OPTION_LETTERS.index(ans), dtype=torch.long),
            "distill_mask": torch.tensor(distill_mask, dtype=torch.float32),
        }


def choice_head_distill_loss(student_logits, labels, teacher_dist, gt_option, option_token_ids, alpha, distill_mask):
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_positions = shift_labels != -100

    ce_terms = []
    kl_terms = []
    ce_weights = []
    kl_weights = []

    for b in range(shift_logits.size(0)):
        pos_idx = torch.nonzero(valid_positions[b], as_tuple=False)
        if pos_idx.numel() == 0:
            continue
        pos = int(pos_idx[0].item())
        opt_logits = shift_logits[b, pos, option_token_ids]

        ce_b = F.cross_entropy(opt_logits.view(1, -1), gt_option[b].view(1), reduction="sum")
        ce_terms.append(ce_b)
        ce_weights.append(1.0)

        tdist = teacher_dist[b].to(student_logits.device).float()
        tdist = torch.clamp(tdist, min=0.0)
        tsum = float(tdist.sum().item())
        if tsum > 0:
            tdist = tdist / tsum
            slogp = F.log_softmax(opt_logits, dim=-1)
            kl_b = F.kl_div(slogp, tdist, reduction="sum")
            w = float(distill_mask[b].item())
            kl_terms.append(kl_b * w)
            kl_weights.append(w)

    if ce_terms:
        ce_loss = torch.stack(ce_terms).sum() / max(1.0, float(sum(ce_weights)))
    else:
        ce_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    if kl_terms and sum(kl_weights) > 0:
        kl_loss = torch.stack(kl_terms).sum() / float(sum(kl_weights))
    else:
        kl_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    return alpha * kl_loss + (1.0 - alpha) * ce_loss


def main():
    parser = argparse.ArgumentParser(description="Choice-head distillation for dental MCQ")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--default_distill_mask", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume_from", type=str, default="")
    args = parser.parse_args()

    set_global_seed(args.seed, args.deterministic)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        trust_remote_code=True,
    )

    option_token_ids = []
    for ch in OPTION_LETTERS:
        tids = tokenizer.encode(ch, add_special_tokens=False)
        option_token_ids.append(tids[0])
    option_token_ids = torch.tensor(option_token_ids, dtype=torch.long, device=device)

    ds = DentalChoiceHeadDataset(
        args.data_path,
        tokenizer,
        default_distill_mask=args.default_distill_mask,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=g)

    base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if args.resume_from and os.path.isdir(args.resume_from):
        model = PeftModel.from_pretrained(base, args.resume_from, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(base, lora_cfg)

    model = model.to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    accum_steps = max(1, args.gradient_accumulation_steps)

    global_step = 0
    for ep in range(args.num_epochs):
        pbar = tqdm(dl, desc=f"HeadDistill Epoch {ep + 1}/{args.num_epochs}")
        optim.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tdist = batch["teacher_dist"].to(device)
            gt_opt = batch["gt_option"].to(device)
            dmask = batch["distill_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn)
            loss = choice_head_distill_loss(
                out.logits,
                labels,
                tdist,
                gt_opt,
                option_token_ids,
                alpha=args.alpha,
                distill_mask=dmask,
            )
            loss = loss / accum_steps
            loss.backward()

            if ((i + 1) % accum_steps == 0) or (i + 1 == len(dl)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_postfix(loss=f"{float(loss.item() * accum_steps):.4f}")

        ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"epoch_{ep + 1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        if args.val_path:
            val_acc = evaluate_generation(model, tokenizer, args.val_path, device)
            print(f"[VAL] epoch={ep + 1} acc={val_acc:.2f}%")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.test_path:
        test_acc = evaluate_generation(model, tokenizer, args.test_path, device)
        print(f"测试集准确率: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
