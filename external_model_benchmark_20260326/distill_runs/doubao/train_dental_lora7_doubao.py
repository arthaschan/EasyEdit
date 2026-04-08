import os
import json
import random
import argparse
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path


def set_global_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# 禁用 wandb 避免API key问题（之前用于绕过缺失问题，现在可以安全注释）
# os.environ['WANDB_API_KEY'] = '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'
# ===================== 自定义数据集类（基于EasyEdit） =====================
class DentalQADataset(Dataset):
    """
    基于EasyEdit CounterFactDataset的牙科QA数据集
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        augment: bool = False,
        default_distill_mask: int = 1,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment  # whether to apply simple text augmentations
        self.default_distill_mask = 1 if int(default_distill_mask) != 0 else 0
        self.option_letters = ["A", "B", "C", "D", "E"]

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
        teacher_answer = str(item.get("TeacherAnswer", answer)).strip().upper()
        if teacher_answer not in self.option_letters:
            teacher_answer = str(answer).strip().upper()

        # 简单数据增强：在 question 前后添加随机短语
        if self.augment and random.random() < 0.3:
            prefixes = ["请回答：", "以下问题：", "问题是："]
            suffixes = ["。", "?", ""]
            question = random.choice(prefixes) + question + random.choice(suffixes)

        # 分拆 prompt 前缀和答案，用于 label masking
        prompt_prefix = f"<|im_start|>system\n你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。\n<|im_end|>\n<|im_start|>user\n问题：{question}\n选项：\n{options}\n<|im_end|>\n<|im_start|>assistant\n"
        full_text = prompt_prefix + f"{answer}<|im_end|>"

        # Tokenize 完整序列
        inputs = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Tokenize 前缀以确定 prompt 长度（使用相同的 add_special_tokens 设置）
        prefix_enc = self.tokenizer(prompt_prefix, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_enc["input_ids"])

        # labels: prompt 部分设为 -100（不参与损失计算），仅答案 token 计算损失
        labels = inputs["input_ids"].squeeze().clone()
        labels[:prefix_len] = -100
        labels[inputs["attention_mask"].squeeze() == 0] = -100  # padding 也忽略

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
            "teacher_dist": self._build_teacher_dist(item, teacher_answer),
            "distill_mask": torch.tensor(
                1
                if str(item.get("SelectiveSource", "")).strip() == "clean_teacher"
                else self.default_distill_mask,
                dtype=torch.float32,
            ),
        }

    def _build_teacher_dist(self, item, teacher_answer: str):
        raw_dist = item.get("TeacherDist", None)
        probs = torch.zeros(len(self.option_letters), dtype=torch.float32)

        if isinstance(raw_dist, dict):
            for i, ch in enumerate(self.option_letters):
                try:
                    probs[i] = float(raw_dist.get(ch, 0.0))
                except Exception:
                    probs[i] = 0.0
        elif isinstance(raw_dist, list) and len(raw_dist) == len(self.option_letters):
            for i, v in enumerate(raw_dist):
                try:
                    probs[i] = float(v)
                except Exception:
                    probs[i] = 0.0

        if float(probs.sum().item()) <= 0.0:
            if teacher_answer in self.option_letters:
                probs[self.option_letters.index(teacher_answer)] = 1.0
            else:
                probs[0] = 1.0

        probs = torch.clamp(probs, min=0.0)
        s = float(probs.sum().item())
        if s <= 0.0:
            probs[0] = 1.0
            s = 1.0
        probs = probs / s
        return probs

# ===================== 蒸馏损失函数 =====================
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5, distill_mask=None):
    """
    计算蒸馏损失：alpha * KL(teacher_logits, student_logits) + (1-alpha) * CE(student_logits, labels)
    注意: Causal LM 中 logits[t] 预测 position t+1，因此 CE 需要做 label shift。
    """
    # Causal LM shift: logits[t] 预测 labels[t+1]
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # 学生模型的交叉熵损失（shift 后）
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    # 蒸馏损失：KL散度（student/teacher logits 对齐，无需 shift）
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    if distill_mask is None:
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    else:
        per_token_kl = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
        per_sample_kl = per_token_kl.mean(dim=-1)
        dm = distill_mask.float().view(-1)
        if float(dm.sum().item()) > 0:
            kl_loss = ((per_sample_kl * dm).sum() / dm.sum()) * (temperature ** 2)
        else:
            kl_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    # 组合损失
    loss = alpha * kl_loss + (1 - alpha) * ce_loss
    return loss


def distillation_loss_with_teacher_dist(
    student_logits,
    labels,
    teacher_dist,
    option_token_ids,
    alpha=0.5,
    distill_mask=None,
):
    """使用教师软标签分布(仅A-E)进行蒸馏：alpha*KL + (1-alpha)*CE。"""
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    option_token_ids = option_token_ids.to(student_logits.device)
    valid_positions = (shift_labels != -100)
    kl_terms = []
    weights = []
    for b in range(shift_logits.size(0)):
        valid_idx = torch.nonzero(valid_positions[b], as_tuple=False)
        if valid_idx.numel() == 0:
            continue

        pos = int(valid_idx[0].item())
        student_option_logits = shift_logits[b, pos, option_token_ids]
        student_log_probs = F.log_softmax(student_option_logits, dim=-1)

        target = teacher_dist[b].to(student_logits.device).float()
        target = torch.clamp(target, min=0.0)
        target_sum = float(target.sum().item())
        if target_sum <= 0.0:
            continue
        target = target / target_sum

        kl_b = F.kl_div(student_log_probs, target, reduction="sum")
        w = 1.0 if distill_mask is None else float(distill_mask[b].item())
        kl_terms.append(kl_b * w)
        weights.append(w)

    if len(kl_terms) > 0 and sum(weights) > 0:
        kl_loss = torch.stack(kl_terms).sum() / float(sum(weights))
    else:
        kl_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    return alpha * kl_loss + (1 - alpha) * ce_loss

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


def find_latest_epoch_checkpoint(output_dir: str):
    ckpt_root = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return 0, ""

    latest_epoch = 0
    latest_dir = ""
    for name in os.listdir(ckpt_root):
        m = re.fullmatch(r"epoch_(\d+)", name)
        if not m:
            continue
        ep = int(m.group(1))
        p = os.path.join(ckpt_root, name)
        if ep > latest_epoch and os.path.isdir(p):
            latest_epoch = ep
            latest_dir = p
    return latest_epoch, latest_dir


def save_train_state(output_dir: str, epoch: int, best_val_acc: float, best_ckpt_path: str):
    state = {
        "last_completed_epoch": int(epoch),
        "best_val_acc": float(best_val_acc),
        "best_ckpt_path": best_ckpt_path or "",
    }
    with open(os.path.join(output_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_train_state(output_dir: str):
    path = os.path.join(output_dir, "training_state.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ===================== 自定义训练循环 =====================
def train_with_distillation(
    student_model,
    teacher_model,
    tokenizer,
    option_token_ids,
    train_dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs=3,
    hparams=None,
    start_epoch=0,
    use_teacher_dist=False,
):
    """使用蒸馏的自定义训练循环，可以在每轮后执行验证。
    若出现OOM会捕获并给出建议。
    """
    student_model.train()
    if teacher_model is not None:
        teacher_model.eval()
    best_acc = hparams.best_val_acc if hparams is not None else 0.0
    accum_steps = max(1, hparams.gradient_accumulation_steps) if hparams is not None else 1

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, num_epochs):
        try:
            print(f"\n🚀 开始训练第 {epoch + 1} 轮")
            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                distill_mask = batch.get("distill_mask", None)
                teacher_dist = batch.get("teacher_dist", None)
                if distill_mask is not None:
                    distill_mask = distill_mask.to(device)
                if teacher_dist is not None:
                    teacher_dist = teacher_dist.to(device)

                # 学生模型前向传播
                student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits

                if use_teacher_dist and teacher_dist is not None:
                    loss = distillation_loss_with_teacher_dist(
                        student_logits,
                        labels,
                        teacher_dist,
                        option_token_ids=option_token_ids,
                        alpha=(hparams.alpha if hparams is not None else 0.5),
                        distill_mask=distill_mask,
                    ) / accum_steps
                else:
                    # 教师模型前向传播（与学生在同一设备）
                    with torch.no_grad():
                        teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                        teacher_logits = teacher_outputs.logits

                    # 计算蒸馏损失，并按梯度累积比例缩放
                    loss = distillation_loss(
                        student_logits,
                        teacher_logits,
                        labels,
                        temperature=(hparams.temperature if hparams is not None else 2.0),
                        alpha=(hparams.alpha if hparams is not None else 0.5),
                        distill_mask=distill_mask,
                    ) / accum_steps

                # 反向传播 + 累积
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

            # 如果提供了验证集则在每轮后评估
            if hparams is not None and hparams.val_path:
                val_acc, wrongs = evaluate_generation(student_model, tokenizer, hparams.val_path, device)
                print(f"第 {epoch + 1} 轮验证准确率: {val_acc:.2f}%")
                # 比较并保存最优模型
                if val_acc > best_acc or (not hparams.best_ckpt_path):
                    best_acc = val_acc
                    hparams.best_val_acc = val_acc
                    save_dir = os.path.join(hparams.output_dir, "best")
                    os.makedirs(save_dir, exist_ok=True)
                    student_model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    hparams.best_ckpt_path = save_dir
                    print(f"保存当前最佳模型到 {save_dir}")
                    # 记录错误样本
                    with open(os.path.join(hparams.output_dir, f"val_wrong_epoch{epoch+1}.jsonl"), "w", encoding="utf-8") as wf:
                        for w in wrongs:
                            wf.write(json.dumps(w, ensure_ascii=False) + "\n")

            # 每个epoch保存断点，支持后续继续训练
            if hparams is not None:
                ckpt_dir = os.path.join(hparams.output_dir, "checkpoints", f"epoch_{epoch+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                student_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                if scheduler is not None:
                    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
                save_train_state(hparams.output_dir, epoch + 1, hparams.best_val_acc, hparams.best_ckpt_path)
                print(f"保存训练断点到 {ckpt_dir}")
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
        default_distill_mask: int = 1
        use_teacher_dist: bool = False

        # 路径配置
        model_name: str = "/home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct"
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
        best_ckpt_path: str = ""
        resume: bool = False
        resume_from: str = ""
        seed: int = 42
        deterministic: bool = False

    # 解析命令行参数并创建配置实例
    parser = argparse.ArgumentParser(description="训练牙科QA模型（蒸馏 + LoRA）")
    parser.add_argument("--num_epochs", type=int, help="训练轮数")
    parser.add_argument("--batch_size", type=int, help="批量大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--model_name", type=str, help="学生/教师基础模型路径或HF仓库")
    parser.add_argument("--rank", type=int, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--temperature", type=float, help="蒸馏温度")
    parser.add_argument("--alpha", type=float, help="蒸馏损失中的KL权重")
    parser.add_argument("--augment", action="store_true", help="启用简单的数据增强")
    parser.add_argument("--data_path", type=str, help="数据路径")
    parser.add_argument("--val_path", type=str, help="验证集 jsonl 路径")
    parser.add_argument("--test_path", type=str, help="测试集 jsonl 路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--resume", action="store_true", help="从output_dir中的最新断点继续训练")
    parser.add_argument("--resume_from", type=str, help="指定断点目录继续训练，例如 checkpoints/epoch_2")
    parser.add_argument("--default_distill_mask", type=int, choices=[0, 1], help="无SelectiveSource字段样本的蒸馏mask默认值")
    parser.add_argument("--use_teacher_dist", action="store_true", help="使用样本中的TeacherDist软标签蒸馏（无需加载本地教师模型）")
    parser.add_argument("--seed", type=int, help="全局随机种子")
    parser.add_argument("--deterministic", action="store_true", help="启用确定性模式（降低吞吐，提升可复现性）")
    args = parser.parse_args()

    hparams = DentalLoRAHyperParams()
    # 用命令行参数覆盖默认值
    if args.num_epochs is not None:
        hparams.num_epochs = args.num_epochs
    if args.batch_size is not None:
        hparams.batch_size = args.batch_size
    if args.learning_rate is not None:
        hparams.learning_rate = args.learning_rate
    if args.model_name is not None:
        hparams.model_name = args.model_name
    if args.rank is not None:
        hparams.rank = args.rank
    if args.lora_alpha is not None:
        hparams.lora_alpha = args.lora_alpha
    if args.temperature is not None and args.temperature > 0:
        hparams.temperature = args.temperature
    if args.alpha is not None and 0.0 <= args.alpha <= 1.0:
        hparams.alpha = args.alpha
    if args.model_name is not None and str(args.model_name).strip():
        hparams.model_name = str(args.model_name).strip()
    if args.data_path is not None:
        hparams.data_path = args.data_path
    if args.output_dir is not None:
        hparams.output_dir = args.output_dir
    if args.augment:
        hparams.augment = True
    if args.batch_size is not None and args.batch_size>0:
        hparams.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps > 0:
        hparams.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.val_path is not None:
        hparams.val_path = args.val_path
    if args.test_path is not None:
        hparams.test_path = args.test_path
    if args.resume:
        hparams.resume = True
    if args.resume_from is not None:
        hparams.resume_from = args.resume_from
    if args.default_distill_mask is not None:
        hparams.default_distill_mask = int(args.default_distill_mask)
    if args.use_teacher_dist:
        hparams.use_teacher_dist = True
    if args.seed is not None:
        hparams.seed = int(args.seed)
    if args.deterministic:
        hparams.deterministic = True

    set_global_seed(hparams.seed, hparams.deterministic)

    # 打印配置以便调试
    print("训练配置:", vars(hparams))
    os.makedirs(hparams.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{hparams.device}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载tokenizer（复用EasyEdit的Qwen设置）
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.model_name,
        eos_token='<|endoftext|>',
        pad_token='<|endoftext|>',
        unk_token='<|endoftext|>',
        trust_remote_code=True
    )
    option_token_ids = []
    for ch in ["A", "B", "C", "D", "E"]:
        token_ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(token_ids) == 0:
            raise ValueError(f"无法为选项 {ch} 编码token")
        if len(token_ids) > 1:
            print(f"[警告] 选项 {ch} 编码为多token，默认使用第一个token: {token_ids}")
        option_token_ids.append(token_ids[0])
    option_token_ids = torch.tensor(option_token_ids, dtype=torch.long, device=device)

    # 2. 加载和处理数据（使用自定义数据集类）
    print("加载数据...")
    dataset = DentalQADataset(
        hparams.data_path,
        tokenizer,
        augment=hparams.augment,
        default_distill_mask=hparams.default_distill_mask,
    )
    loader_gen = torch.Generator()
    loader_gen.manual_seed(hparams.seed)
    train_dataloader = DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        generator=loader_gen,
    )

    # 3. 可选加载教师模型到GPU
    teacher_model = None
    if not hparams.use_teacher_dist:
        print("加载教师模型到GPU...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            hparams.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        teacher_model.eval()  # 教师模型设为评估模式
    else:
        print("启用TeacherDist软标签蒸馏，跳过本地教师模型加载")

    # 4. 加载学生模型并应用LoRA（复用EasyEdit的LoRA配置）
    print("加载学生模型并应用LoRA...")
    student_base = AutoModelForCausalLM.from_pretrained(
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
    start_epoch = 0
    resume_dir = ""
    if hparams.resume_from:
        resume_dir = hparams.resume_from
    elif hparams.resume:
        _, latest_dir = find_latest_epoch_checkpoint(hparams.output_dir)
        resume_dir = latest_dir

    if resume_dir and os.path.isdir(resume_dir):
        print(f"检测到断点，加载LoRA权重: {resume_dir}")
        student_model = PeftModel.from_pretrained(student_base, resume_dir, is_trainable=True)

        m = re.search(r"epoch_(\d+)", resume_dir)
        if m:
            start_epoch = int(m.group(1))
        state = load_train_state(hparams.output_dir)
        if state:
            hparams.best_val_acc = float(state.get("best_val_acc", hparams.best_val_acc))
            hparams.best_ckpt_path = str(state.get("best_ckpt_path", hparams.best_ckpt_path))
        print(f"将从 epoch={start_epoch + 1} 开始继续训练")
    else:
        student_model = get_peft_model(student_base, lora_config)

    student_model = student_model.to(device)
    student_model.print_trainable_parameters()  # 显示可训练参数

    # 5. 优化器和调度器
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay
    )
    # Cosine warmup 学习率调度器
    accum_steps_for_sched = max(1, hparams.gradient_accumulation_steps)
    total_steps = (len(train_dataloader) + accum_steps_for_sched - 1) // accum_steps_for_sched * hparams.num_epochs
    warmup_steps = max(1, total_steps // 10)  # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"调度器: cosine warmup, total_steps={total_steps}, warmup={warmup_steps}")

    if resume_dir and os.path.isdir(resume_dir):
        opt_path = os.path.join(resume_dir, "optimizer.pt")
        sch_path = os.path.join(resume_dir, "scheduler.pt")
        if os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"已恢复优化器状态: {opt_path}")
        if os.path.isfile(sch_path):
            scheduler.load_state_dict(torch.load(sch_path, map_location=device))
            print(f"已恢复调度器状态: {sch_path}")

    # 6. 开始蒸馏训练
    print("开始蒸馏训练...")
    if start_epoch >= hparams.num_epochs:
        print(f"检测到训练已完成: start_epoch={start_epoch}, num_epochs={hparams.num_epochs}，跳过训练阶段")
        trained_model = student_model
    else:
        trained_model = train_with_distillation(
            student_model, teacher_model, tokenizer, option_token_ids, train_dataloader,
            optimizer, scheduler, device, num_epochs=hparams.num_epochs, hparams=hparams,
            start_epoch=start_epoch, use_teacher_dist=hparams.use_teacher_dist
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
        model_for_test = trained_model
        if hparams.best_ckpt_path and os.path.isdir(hparams.best_ckpt_path):
            print(f"检测到最佳检查点: {hparams.best_ckpt_path}，使用最佳模型进行测试评估")
            base_for_eval = AutoModelForCausalLM.from_pretrained(
                hparams.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
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