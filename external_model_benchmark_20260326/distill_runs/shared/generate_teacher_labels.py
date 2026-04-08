#!/usr/bin/env python3
import argparse
import sys
import json
import os
import random
import re
import hashlib
import time
from pathlib import Path

import requests

ANSWER_RE = re.compile(r"\b([A-E])\b")


def resolve_api_key(candidate):
    direct_key = str(candidate.get("api_key", "")).strip()
    if direct_key:
        return direct_key

    key_env = str(candidate.get("api_key_env", "")).strip()
    if not key_env:
        return ""

    if key_env.startswith("sk-") or key_env.startswith("eyJ"):
        return key_env

    if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key_env):
        return os.getenv(key_env, "").strip()

    return key_env


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_question_text(item):
    q = str(item.get("Question") or item.get("question") or "").strip()
    options = item.get("Options") or item.get("options") or {}
    lines = [q]

    if isinstance(options, dict):
        for k in ["A", "B", "C", "D", "E"]:
            if k in options:
                lines.append(f"{k}. {str(options[k]).strip()}")
    else:
        opt_text = str(options).strip()
        if opt_text:
            lines.append(opt_text)

    lines.append("请只输出一个大写字母（A/B/C/D/E）。")
    return "\n".join(lines)


def sample_key(item):
    q = str(item.get("Question") or item.get("question") or "").strip()
    options = item.get("Options") or item.get("options") or {}
    if isinstance(options, dict):
        opt_text = "\n".join(f"{k}:{str(options.get(k, ''))}" for k in ["A", "B", "C", "D", "E"])
    else:
        opt_text = str(options).strip()
    raw = f"{q}\n{opt_text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def extract_answer_letter(text):
    if not text:
        return ""
    t = text.strip().upper()
    if len(t) == 1 and t in "ABCDE":
        return t
    m = ANSWER_RE.search(t)
    return m.group(1) if m else ""


def call_openai_compatible(candidate, system_prompt, user_prompt, timeout_sec, max_tokens, max_retries):
    api_key = resolve_api_key(candidate)
    if not api_key:
        raise RuntimeError(f"missing api key for {candidate.get('name')}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": candidate["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(candidate["base_url"], headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code >= 400:
                text = (resp.text or "").strip().replace("\n", " ")
                raise RuntimeError(f"http_{resp.status_code}: {text[:240]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last_err

    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--system_prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout_sec", type=int, default=120)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--request_interval_sec", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true", help="append to existing output and skip processed samples")
    args = parser.parse_args()

    with open(args.candidate, "r", encoding="utf-8") as f:
        candidate = json.load(f)

    with open(args.system_prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    rows = load_jsonl(args.dataset)
    random.seed(args.seed)
    if args.sample_size and 0 < args.sample_size < len(rows):
        rows = random.sample(rows, args.sample_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = set()
    existing_valid = 0
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    old = json.loads(line)
                except Exception:
                    continue
                done_keys.add(sample_key(old))
                if str(old.get("Answer", "")).strip().upper() in {"A", "B", "C", "D", "E"}:
                    existing_valid += 1
        print(f"[RESUME] existing_done={len(done_keys)} existing_valid={existing_valid}", flush=True)

    total = 0
    valid = 0
    failed = 0

    mode = "a" if (args.resume and out_path.exists()) else "w"
    with open(out_path, mode, encoding="utf-8") as wf:
        for i, item in enumerate(rows, start=1):
            if sample_key(item) in done_keys:
                continue
            prompt = build_question_text(item)
            total += 1
            try:
                raw = call_openai_compatible(
                    candidate=candidate,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    timeout_sec=args.timeout_sec,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
                pred = extract_answer_letter(raw)
                if pred in {"A", "B", "C", "D", "E"}:
                    item2 = dict(item)
                    item2["TeacherAnswer"] = pred
                    item2["TeacherRaw"] = raw
                    item2["OriginalAnswer"] = str(item.get("Answer") or item.get("answer") or "").strip().upper()
                    item2["Answer"] = pred
                    wf.write(json.dumps(item2, ensure_ascii=False) + "\n")
                    valid += 1
                    done_keys.add(sample_key(item))
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                err_item = {
                    "index": i,
                    "error": str(e),
                    "question": str(item.get("Question") or item.get("question") or "")[:120],
                }
                print("[ERR]", json.dumps(err_item, ensure_ascii=False), flush=True)

            if args.request_interval_sec > 0:
                time.sleep(args.request_interval_sec)

            if i % 20 == 0:
                print(
                    f"[PROGRESS] scanned={i}/{len(rows)} newly_valid={valid} failed={failed} total_valid={existing_valid + valid}",
                    flush=True,
                )

    print(
        f"[DONE] attempted={total} newly_valid={valid} failed={failed} total_valid={existing_valid + valid} output={out_path}"
    )
    if existing_valid + valid == 0:
        print("[FATAL] no valid teacher labels generated", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
