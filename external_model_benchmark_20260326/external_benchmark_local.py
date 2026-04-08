#!/usr/bin/env python3
import argparse
import gc
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

ANSWER_RE = re.compile(r"\b([A-E])\b")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_question_text(item):
    q = str(item.get("question") or item.get("Question") or "").strip()
    options = item.get("options") or item.get("Options") or {}
    lines = [q]
    if isinstance(options, dict):
        for k in ["A", "B", "C", "D", "E"]:
            if k in options:
                lines.append(f"{k}. {str(options[k]).strip()}")
    else:
        text = str(options).strip()
        if text:
            lines.append(text)
    lines.append("请只输出一个大写字母（A/B/C/D/E）。")
    return "\n".join(lines)


def extract_answer_letter(text):
    if not text:
        return ""
    t = text.strip().upper()
    if len(t) == 1 and t in "ABCDE":
        return t
    m = ANSWER_RE.search(t)
    return m.group(1) if m else ""


def to_torch_dtype(name):
    n = str(name or "").lower().strip()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16"}:
        return torch.float16
    if n in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16


def load_local_model(candidate):
    model_path = candidate["model_path"]
    tokenizer_path = candidate.get("tokenizer_path") or model_path
    torch_dtype = to_torch_dtype(candidate.get("torch_dtype", "bfloat16"))
    device_map = candidate.get("device_map", "auto")

    quantization_config = None
    if bool(candidate.get("load_in_4bit", False)):
        if BitsAndBytesConfig is None:
            raise RuntimeError("load_in_4bit requested but BitsAndBytesConfig is unavailable")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return tok, model


def build_input_text(tok, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


def evaluate_local_model(candidate, samples, system_prompt, max_new_tokens):
    tok = None
    model = None
    try:
        tok, model = load_local_model(candidate)
    except Exception as e:
        return {
            "name": candidate["name"],
            "model_path": candidate.get("model_path", ""),
            "status": "failed",
            "reason": f"load_error: {e}",
        }

    total = 0
    correct = 0
    parsed = 0
    errors = 0
    latencies = []
    out_token_counts = []
    details = []

    for i, item in enumerate(samples, start=1):
        gt = str(item.get("answer") or item.get("Answer") or "").strip().upper()
        if gt not in {"A", "B", "C", "D", "E"}:
            continue
        total += 1

        prompt = build_question_text(item)
        text = build_input_text(tok, system_prompt, prompt)

        try:
            inputs = tok(text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            start = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tok.eos_token_id,
                )
            elapsed = time.perf_counter() - start

            gen = out[0][inputs["input_ids"].shape[1] :]
            raw = tok.decode(gen, skip_special_tokens=True)
            pred = extract_answer_letter(raw)

            if pred:
                parsed += 1
            hit = pred == gt
            if hit:
                correct += 1

            latencies.append(elapsed)
            out_token_counts.append(int(gen.shape[0]))
            details.append(
                {
                    "index": i,
                    "answer": gt,
                    "prediction": pred,
                    "hit": hit,
                    "raw": raw,
                    "latency_sec": round(elapsed, 4),
                }
            )
        except Exception as e:
            errors += 1
            details.append(
                {
                    "index": i,
                    "answer": gt,
                    "prediction": "",
                    "hit": False,
                    "raw": "",
                    "error": str(e),
                }
            )

    acc = (100.0 * correct / total) if total else 0.0
    parsed_rate = (100.0 * parsed / total) if total else 0.0
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    total_out_tokens = sum(out_token_counts)
    total_time = sum(latencies)
    tps = (total_out_tokens / total_time) if total_time > 1e-9 else 0.0

    status = "ok"
    reason = ""
    if total > 0 and errors == total and parsed == 0:
        status = "failed"
        reason = "all requests failed"

    result = {
        "name": candidate["name"],
        "provider": "local",
        "model": candidate.get("model_path", ""),
        "status": status,
        "reason": reason,
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 2),
        "parsed": parsed,
        "parsed_rate": round(parsed_rate, 2),
        "errors": errors,
        "avg_latency_sec": round(avg_latency, 4),
        "output_tokens": total_out_tokens,
        "tokens_per_sec": round(tps, 2),
        "details": details,
    }

    del model
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def write_markdown(path, run_meta, results):
    lines = [
        "# Local H100 Benchmark Leaderboard",
        "",
        f"- timestamp: {run_meta['timestamp']}",
        f"- dataset: `{run_meta['dataset']}`",
        f"- sample_size: {run_meta['sample_size']}",
        f"- seed: {run_meta['seed']}",
        "",
        "| Rank | Name | Model Path | Accuracy(%) | Parsed(%) | Total | Correct | Avg Latency(s) | Tokens/s | Status |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    ok = [r for r in results if r.get("status") == "ok"]
    ok.sort(key=lambda x: x.get("accuracy", 0.0), reverse=True)

    rank = 1
    for r in ok:
        lines.append(
            f"| {rank} | {r['name']} | {r['model']} | {r['accuracy']:.2f} | {r['parsed_rate']:.2f} | {r['total']} | {r['correct']} | {r['avg_latency_sec']:.4f} | {r['tokens_per_sec']:.2f} | ok |"
        )
        rank += 1

    for r in results:
        if r.get("status") != "ok":
            lines.append(
                f"| - | {r.get('name','')} | {r.get('model','')} | - | - | - | - | - | - | {r.get('status','failed')}: {r.get('reason','')} |"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--system_prompt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.system_prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    with open(args.candidates, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    data = load_jsonl(args.dataset)
    random.seed(args.seed)
    if args.sample_size and args.sample_size > 0 and args.sample_size < len(data):
        data = random.sample(data, args.sample_size)

    enabled = [c for c in candidates if c.get("enabled", True)]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_meta = {
        "timestamp": ts,
        "dataset": args.dataset,
        "sample_size": len(data),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }

    results = []
    for c in enabled:
        print(f"[RUN] {c['name']} model_path={c.get('model_path','')}", flush=True)
        r = evaluate_local_model(c, data, system_prompt, args.max_new_tokens)
        results.append(r)
        if r.get("status") == "ok":
            print(
                f"[DONE] {c['name']} acc={r['accuracy']:.2f}% parsed={r['parsed_rate']:.2f}% tps={r['tokens_per_sec']:.2f}",
                flush=True,
            )
        else:
            print(f"[SKIP] {c['name']} reason={r.get('reason','')}", flush=True)

    j = out_dir / f"leaderboard_local_{ts}.json"
    m = out_dir / f"leaderboard_local_{ts}.md"
    lj = out_dir / "leaderboard_local_latest.json"
    lm = out_dir / "leaderboard_local_latest.md"

    with open(j, "w", encoding="utf-8") as f:
        json.dump({"run": run_meta, "results": results}, f, ensure_ascii=False, indent=2)
    write_markdown(m, run_meta, results)

    with open(lj, "w", encoding="utf-8") as f:
        json.dump({"run": run_meta, "results": results}, f, ensure_ascii=False, indent=2)
    write_markdown(lm, run_meta, results)

    print(f"[OUT] {j}")
    print(f"[OUT] {m}")
    print(f"[OUT] {lj}")
    print(f"[OUT] {lm}")


if __name__ == "__main__":
    main()
