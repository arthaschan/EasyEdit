#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_answer_char(text: str) -> str:
    for ch in text.strip().upper():
        if ch in ["A", "B", "C", "D", "E"]:
            return ch
    return ""


def build_prompt(item: dict) -> str:
    q = item.get("Question", "")
    opts = item.get("Options", "")
    return (
        "<|im_start|>system\n"
        "你是一名专业的牙科医生，只需输出一个字母（A、B、C、D、E）作为结果，不要附带任何解释或空格。\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"问题：{q}\n"
        f"选项：\n{opts}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def evaluate(adapter_dir: Path, base_model: str, test_data: Path, device: str):
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    model.eval()

    samples = load_jsonl(test_data)
    total = 0
    correct = 0

    for item in samples:
        ans = str(item.get("Answer", "")).strip().upper()
        if ans not in {"A", "B", "C", "D", "E"}:
            continue
        total += 1

        prompt = build_prompt(item)
        inputs = tok(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4)
        gen = tok.decode(out[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        pred = extract_answer_char(gen)
        if pred == ans:
            correct += 1

    acc = 100.0 * correct / total if total else 0.0
    return {"total": total, "correct": correct, "accuracy": round(acc, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--distill_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    root = Path(args.distill_root)
    test_data = Path(args.test_data)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    mapping = [
        ("DeepSeek-V3", root / "deepseek_v3" / "outputs" / "student_qwen25_7b_from_deepseek_v3" / "best"),
        ("moonshot-v1-32k", root / "moonshot_v1_32k" / "outputs" / "student_qwen25_7b_from_moonshot_v1_32k" / "best"),
        ("doubao", root / "doubao" / "outputs" / "student_qwen25_7b_from_doubao" / "best"),
    ]

    results = []
    for teacher, adapter in mapping:
        if not adapter.exists():
            results.append({"teacher": teacher, "status": "missing_adapter", "adapter": str(adapter)})
            continue
        try:
            r = evaluate(adapter, args.base_model, test_data, device)
            r.update({"teacher": teacher, "status": "ok", "adapter": str(adapter)})
            results.append(r)
        except Exception as e:
            results.append({"teacher": teacher, "status": "failed", "error": str(e), "adapter": str(adapter)})

    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"device": device, "results": results}, f, ensure_ascii=False, indent=2)

    lines = [
        "# Distillation Accuracy Report",
        "",
        f"- device: {device}",
        f"- base_model: {args.base_model}",
        f"- test_data: {args.test_data}",
        "",
        "| Teacher | Accuracy(%) | Correct | Total | Status |",
        "|---|---:|---:|---:|---|",
    ]

    ok_rows = [x for x in results if x.get("status") == "ok"]
    ok_rows.sort(key=lambda x: x.get("accuracy", 0.0), reverse=True)
    for r in ok_rows:
        lines.append(f"| {r['teacher']} | {r['accuracy']:.2f} | {r['correct']} | {r['total']} | ok |")

    for r in results:
        if r.get("status") != "ok":
            lines.append(f"| {r['teacher']} | - | - | - | {r.get('status')} |")

    if ok_rows:
        best = ok_rows[0]
        lines += [
            "",
            "## Improvement Suggestions",
            "",
            f"- Best current teacher: {best['teacher']} ({best['accuracy']:.2f}%).",
            "- For lower-accuracy teachers, increase label quality filtering: keep only samples where teacher output is a strict single A-E letter.",
            "- Tune distillation weight: try alpha in [0.3, 0.7] and temperature in [1.5, 3.0].",
            "- Use curriculum: first train on high-confidence teacher labels, then full teacher set.",
            "- Add 5-10% ground-truth supervised anchors to reduce teacher-specific bias drift.",
            "- If one teacher underperforms, use weighted teacher mixture rather than pure single-teacher distillation.",
        ]

    with output_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OUT] {output_json}")
    print(f"[OUT] {output_md}")


if __name__ == "__main__":
    main()
