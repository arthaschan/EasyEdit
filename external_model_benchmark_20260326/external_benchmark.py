#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import requests
from volcenginesdkcore.signv4 import SignerV4

ANSWER_RE = re.compile(r"\b([A-E])\b")


def _retry_wait_seconds(resp, attempt):
    retry_after = ""
    if resp is not None:
        retry_after = str(resp.headers.get("Retry-After", "")).strip()
    if retry_after.isdigit():
        return max(float(retry_after), 1.0)
    # Conservative fallback for free-tier throttling windows.
    return 6.0 * (attempt + 1)


def resolve_api_key(candidate):
    # Preferred: explicit api_key field.
    direct_key = str(candidate.get("api_key", "")).strip()
    if direct_key:
        return direct_key, "direct"

    key_env = str(candidate.get("api_key_env", "")).strip()
    if not key_env:
        return "", "missing"

    # Backward compatibility: if a raw key was mistakenly placed in api_key_env,
    # treat it as the actual key instead of environment variable name.
    if key_env.startswith("sk-") or key_env.startswith("eyJ"):
        return key_env, "inline"
    # If it's not a valid env var token, treat it as an inline key.
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key_env):
        return key_env, "inline"

    env_val = os.getenv(key_env, "").strip()
    if env_val:
        return env_val, "env"

    return "", key_env


def resolve_field_or_env(candidate, direct_fields, env_fields, fallback_inline_when_env_missing=False):
    for k in direct_fields:
        v = str(candidate.get(k, "")).strip()
        if v:
            return v, f"direct:{k}"

    for k in env_fields:
        name_or_value = str(candidate.get(k, "")).strip()
        if not name_or_value:
            continue
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*", name_or_value):
            env_val = os.getenv(name_or_value, "").strip()
            if env_val:
                return env_val, f"env:{name_or_value}"
            if fallback_inline_when_env_missing:
                return name_or_value, f"inline:{k}"
            continue
        return name_or_value, f"inline:{k}"

    return "", "missing"


def resolve_doubao_aksk(candidate):
    ak, ak_src = resolve_field_or_env(
        candidate,
        direct_fields=["access_key_id", "ak"],
        env_fields=["access_key_id_env", "ak_env"],
        fallback_inline_when_env_missing=True,
    )
    sk, sk_src = resolve_field_or_env(
        candidate,
        direct_fields=["secret_access_key", "sk"],
        env_fields=["secret_access_key_env", "sk_env"],
        fallback_inline_when_env_missing=True,
    )
    sts, _ = resolve_field_or_env(
        candidate,
        direct_fields=["session_token", "security_token"],
        env_fields=["session_token_env", "security_token_env", "sts_token_env"],
    )
    return ak, sk, sts, ak_src, sk_src


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_question_text(item):
    q = str(item.get("question") or item.get("Question") or "").strip()
    options = item.get("options") or item.get("Options") or {}
    opt_keys = ["A", "B", "C", "D", "E"]
    lines = [q]

    if isinstance(options, dict):
        for k in opt_keys:
            if k in options:
                lines.append(f"{k}. {str(options[k]).strip()}")
    else:
        # Options sometimes come as a newline-joined string: "A xxx\nB yyy..."
        opt_text = str(options).strip()
        if opt_text:
            lines.append(opt_text)

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


def post_chat_completion(
    url,
    api_key,
    model,
    system_prompt,
    user_prompt,
    timeout_sec,
    max_tokens,
    max_retries=2,
):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(_retry_wait_seconds(resp, attempt))
                continue
            if resp.status_code >= 400:
                body_text = (resp.text or "").strip().replace("\n", " ")
                snippet = body_text[:240]
                raise requests.HTTPError(
                    f"{resp.status_code} Client Error for url: {url}; body={snippet}"
                )
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last_error

    raise last_error


def post_chat_completion_doubao_aksk(
    url,
    ak,
    sk,
    model,
    system_prompt,
    user_prompt,
    timeout_sec,
    max_tokens,
    session_token="",
    max_retries=2,
):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    u = urlsplit(url)
    path = u.path or "/"
    query = dict(parse_qsl(u.query, keep_blank_values=True))

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            headers = {
                "Host": u.netloc,
                "Content-Type": "application/json; charset=utf-8",
            }
            SignerV4.sign(
                path=path,
                method="POST",
                headers=headers,
                body=body,
                post_params={},
                query=query,
                ak=ak,
                sk=sk,
                region="cn-beijing",
                service="ark",
                session_token=session_token or None,
            )
            resp = requests.post(url, headers=headers, data=body.encode("utf-8"), timeout=timeout_sec)
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(_retry_wait_seconds(resp, attempt))
                continue
            if resp.status_code >= 400:
                body_text = (resp.text or "").strip().replace("\n", " ")
                snippet = body_text[:240]
                raise requests.HTTPError(
                    f"{resp.status_code} Client Error for url: {url}; body={snippet}"
                )
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last_error

    raise last_error


def evaluate_one_model(candidate, samples, system_prompt, timeout_sec, max_tokens, sleep_sec):
    provider = str(candidate.get("provider", "")).strip().lower()
    if bool(candidate.get("free_account", False)):
        return {
            "name": candidate["name"],
            "provider": candidate.get("provider", ""),
            "model": candidate["model"],
            "status": "skipped",
            "reason": f"{provider or 'provider'} free account: skipped by config",
        }

    api_key, key_source = resolve_api_key(candidate)
    use_doubao_aksk = provider == "doubao" and str(candidate.get("auth_mode", "")).lower() == "aksk"
    if use_doubao_aksk:
        ak, sk, sts_token, ak_src, sk_src = resolve_doubao_aksk(candidate)
        if not ak or not sk:
            return {
                "name": candidate["name"],
                "provider": candidate.get("provider", ""),
                "model": candidate["model"],
                "status": "skipped",
                "reason": f"missing doubao aksk: ak={ak_src}, sk={sk_src}",
            }
    elif not api_key:
        return {
            "name": candidate["name"],
            "provider": candidate.get("provider", ""),
            "model": candidate["model"],
            "status": "skipped",
            "reason": f"missing api key: {key_source}",
        }

    total = 0
    correct = 0
    parsed = 0
    errors = 0
    details = []
    candidate_max_retries = int(candidate.get("max_retries", 2))
    candidate_interval = float(candidate.get("request_interval_sec", 0.0))
    effective_sleep = max(float(sleep_sec), candidate_interval)

    for i, item in enumerate(samples, start=1):
        gt = str(item.get("answer") or item.get("Answer") or "").strip().upper()
        if gt not in {"A", "B", "C", "D", "E"}:
            continue

        prompt = build_question_text(item)
        total += 1

        try:
            if use_doubao_aksk:
                raw = post_chat_completion_doubao_aksk(
                    url=candidate["base_url"],
                    ak=ak,
                    sk=sk,
                    session_token=sts_token,
                    model=candidate["model"],
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    timeout_sec=timeout_sec,
                    max_tokens=max_tokens,
                    max_retries=candidate_max_retries,
                )
            else:
                raw = post_chat_completion(
                    url=candidate["base_url"],
                    api_key=api_key,
                    model=candidate["model"],
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    timeout_sec=timeout_sec,
                    max_tokens=max_tokens,
                    max_retries=candidate_max_retries,
                )
            pred = extract_answer_letter(raw)
            if pred:
                parsed += 1
            hit = pred == gt
            if hit:
                correct += 1

            details.append(
                {
                    "index": i,
                    "answer": gt,
                    "prediction": pred,
                    "hit": hit,
                    "raw": raw,
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

        if effective_sleep > 0:
            time.sleep(effective_sleep)

    acc = (100.0 * correct / total) if total else 0.0
    parsed_rate = (100.0 * parsed / total) if total else 0.0

    status = "ok"
    reason = ""
    if total > 0 and errors == total and parsed == 0:
        status = "failed"
        first_err = ""
        for d in details:
            if d.get("error"):
                first_err = d["error"]
                break
        reason = first_err or "all requests failed"

    return {
        "name": candidate["name"],
        "provider": candidate.get("provider", ""),
        "model": candidate["model"],
        "status": status,
        "reason": reason,
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 2),
        "parsed": parsed,
        "parsed_rate": round(parsed_rate, 2),
        "errors": errors,
        "details": details,
    }


def write_markdown(path, run_meta, results):
    lines = []
    lines.append("# External Teacher Leaderboard")
    lines.append("")
    lines.append(f"- timestamp: {run_meta['timestamp']}")
    lines.append(f"- dataset: `{run_meta['dataset']}`")
    lines.append(f"- sample_size: {run_meta['sample_size']}")
    lines.append(f"- seed: {run_meta['seed']}")
    lines.append("")
    lines.append("| Rank | Name | Provider | Model | Accuracy(%) | Parsed(%) | Total | Correct | Errors | Status |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---|")

    ok_results = [r for r in results if r.get("status") == "ok"]
    ok_results.sort(key=lambda x: x["accuracy"], reverse=True)

    rank = 1
    for r in ok_results:
        lines.append(
            f"| {rank} | {r['name']} | {r['provider']} | {r['model']} | {r['accuracy']:.2f} | {r['parsed_rate']:.2f} | {r['total']} | {r['correct']} | {r['errors']} | ok |"
        )
        rank += 1

    for r in results:
        if r.get("status") != "ok":
            lines.append(
                f"| - | {r['name']} | {r.get('provider','')} | {r.get('model','')} | - | - | - | - | - | {r.get('status','unknown')}: {r.get('reason','')} |"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--candidates", required=True, help="Path to candidates JSON")
    parser.add_argument("--system_prompt", required=True, help="Path to system prompt txt")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--sample_size", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout_sec", type=int, default=90)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--sleep_sec", type=float, default=0.0)
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

    enabled_candidates = [c for c in candidates if c.get("enabled", True)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "sample_size": len(data),
        "seed": args.seed,
        "timeout_sec": args.timeout_sec,
        "max_tokens": args.max_tokens,
        "sleep_sec": args.sleep_sec,
    }

    results = []
    for c in enabled_candidates:
        print(f"[RUN] {c['name']} ({c.get('provider','')}) model={c['model']}", flush=True)
        res = evaluate_one_model(
            candidate=c,
            samples=data,
            system_prompt=system_prompt,
            timeout_sec=args.timeout_sec,
            max_tokens=args.max_tokens,
            sleep_sec=args.sleep_sec,
        )
        results.append(res)
        if res.get("status") == "ok":
            print(
                f"[DONE] {c['name']} acc={res['accuracy']:.2f}% parsed={res['parsed_rate']:.2f}% total={res['total']}",
                flush=True,
            )
        else:
            print(f"[SKIP] {c['name']} reason={res.get('reason','')}", flush=True)

    json_path = out_dir / f"leaderboard_{timestamp}.json"
    md_path = out_dir / f"leaderboard_{timestamp}.md"
    latest_json = out_dir / "leaderboard_latest.json"
    latest_md = out_dir / "leaderboard_latest.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"run": run_meta, "results": results}, f, ensure_ascii=False, indent=2)

    write_markdown(md_path, run_meta, results)

    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump({"run": run_meta, "results": results}, f, ensure_ascii=False, indent=2)

    write_markdown(latest_md, run_meta, results)

    print(f"[OUT] {json_path}")
    print(f"[OUT] {md_path}")
    print(f"[OUT] {latest_json}")
    print(f"[OUT] {latest_md}")


if __name__ == "__main__":
    main()
