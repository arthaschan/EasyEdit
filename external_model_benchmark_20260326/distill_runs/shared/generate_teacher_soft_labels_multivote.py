#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

ANSWER_RE = re.compile(r"\b([A-E])\b")
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


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


def sample_key(item):
    q = str(item.get("Question") or item.get("question") or "").strip()
    options = item.get("Options") or item.get("options") or {}
    if isinstance(options, dict):
        opt_text = "\n".join(f"{k}:{str(options.get(k, ''))}" for k in OPTION_LETTERS)
    else:
        opt_text = str(options).strip()
    raw = f"{q}\n{opt_text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_question_text(item):
    q = str(item.get("Question") or item.get("question") or "").strip()
    options = item.get("Options") or item.get("options") or {}
    lines = [q]

    if isinstance(options, dict):
        for k in OPTION_LETTERS:
            if k in options:
                lines.append(f"{k}. {str(options[k]).strip()}")
    else:
        opt_text = str(options).strip()
        if opt_text:
            lines.append(opt_text)

    lines.append("请只输出一个大写字母（A/B/C/D/E）。")
    return "\n".join(lines)


def extract_answer_letter(text):
    if not text:
        return ""
    t = text.strip().upper()
    if len(t) == 1 and t in OPTION_LETTERS:
        return t
    m = ANSWER_RE.search(t)
    return m.group(1) if m else ""


def call_openai_compatible(candidate, system_prompt, user_prompt, timeout_sec, max_tokens, max_retries, temperature):
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
        "temperature": temperature,
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


def build_dist(votes):
    counts = {k: 0 for k in OPTION_LETTERS}
    for v in votes:
        if v in counts:
            counts[v] += 1
    total = sum(counts.values())
    if total <= 0:
        return {k: (1.0 if k == "A" else 0.0) for k in OPTION_LETTERS}
    return {k: counts[k] / float(total) for k in OPTION_LETTERS}


def pick_majority(dist, fallback="A"):
    best_k = fallback
    best_v = -1.0
    for k in OPTION_LETTERS:
        v = float(dist.get(k, 0.0))
        if v > best_v:
            best_v = v
            best_k = k
    return best_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing_labels", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--system_prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout_sec", type=int, default=120)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--request_interval_sec", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--extra_votes", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.candidate, "r", encoding="utf-8") as f:
        candidate = json.load(f)

    with open(args.system_prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    rows = load_jsonl(args.existing_labels)
    random.seed(args.seed)

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
                td = old.get("TeacherDist")
                if isinstance(td, dict) and sum(float(td.get(k, 0.0)) for k in OPTION_LETTERS) > 0:
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

            total += 1
            prompt = build_question_text(item)
            votes = []
            base_vote = str(item.get("TeacherAnswer") or item.get("Answer") or "").strip().upper()
            if base_vote in OPTION_LETTERS:
                votes.append(base_vote)

            raws = []
            try:
                for _ in range(max(0, args.extra_votes)):
                    raw = call_openai_compatible(
                        candidate=candidate,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        timeout_sec=args.timeout_sec,
                        max_tokens=args.max_tokens,
                        max_retries=args.max_retries,
                        temperature=args.temperature,
                    )
                    raws.append(raw)
                    v = extract_answer_letter(raw)
                    if v in OPTION_LETTERS:
                        votes.append(v)
                    if args.request_interval_sec > 0:
                        time.sleep(args.request_interval_sec)

                if not votes:
                    failed += 1
                    continue

                dist = build_dist(votes)
                maj = pick_majority(dist, fallback=votes[0])
                item2 = dict(item)
                item2["TeacherVotes"] = votes
                item2["TeacherRawVotes"] = raws
                item2["TeacherDist"] = dist
                item2["TeacherAnswer"] = maj
                item2["Answer"] = maj
                wf.write(json.dumps(item2, ensure_ascii=False) + "\n")

                valid += 1
                done_keys.add(sample_key(item))
            except Exception as e:
                failed += 1
                err_item = {
                    "index": i,
                    "error": str(e),
                    "question": str(item.get("Question") or item.get("question") or "")[:120],
                }
                print("[ERR]", json.dumps(err_item, ensure_ascii=False), flush=True)

            if i % 20 == 0:
                print(
                    f"[PROGRESS] scanned={i}/{len(rows)} newly_valid={valid} failed={failed} total_valid={existing_valid + valid}",
                    flush=True,
                )

    print(f"[DONE] attempted={total} newly_valid={valid} failed={failed} total_valid={existing_valid + valid} output={out_path}", flush=True)
    if existing_valid + valid == 0:
        print("[FATAL] no valid soft labels generated", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
