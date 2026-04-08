#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import requests
from volcenginesdkcore.signv4 import SignerV4


def resolve_field_or_env(candidate, direct_fields, env_fields, fallback_inline_when_env_missing=False):
    for k in direct_fields:
        v = str(candidate.get(k, "")).strip()
        if v:
            return v, f"direct:{k}"

    for k in env_fields:
        name_or_value = str(candidate.get(k, "")).strip()
        if not name_or_value:
            continue
        if name_or_value.isupper() and name_or_value.replace("_", "").isalnum():
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
    sts, sts_src = resolve_field_or_env(
        candidate,
        direct_fields=["session_token", "security_token"],
        env_fields=["session_token_env", "security_token_env", "sts_token_env"],
        fallback_inline_when_env_missing=True,
    )
    return ak, sk, sts, ak_src, sk_src, sts_src


def mask(s, head=4, tail=4):
    if not s:
        return ""
    if len(s) <= head + tail:
        return "*" * len(s)
    return s[:head] + "..." + s[-tail:]


def find_doubao_candidate(candidates):
    for c in candidates:
        if str(c.get("provider", "")).lower() == "doubao" and c.get("enabled", True):
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--timeout_sec", type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.candidates, "r", encoding="utf-8") as f:
        arr = json.load(f)

    cand = find_doubao_candidate(arr)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": ts,
        "status": "unknown",
        "candidate_found": bool(cand),
        "checks": {},
    }

    if not cand:
        report["status"] = "failed"
        report["error"] = "no enabled doubao candidate found"
    else:
        url = cand.get("base_url", "")
        model = cand.get("model", "")
        auth_mode = str(cand.get("auth_mode", "")).lower()
        ak, sk, sts, ak_src, sk_src, sts_src = resolve_doubao_aksk(cand)

        report["checks"] = {
            "auth_mode": auth_mode,
            "base_url": url,
            "model": model,
            "ak_source": ak_src,
            "sk_source": sk_src,
            "sts_source": sts_src,
            "ak_masked": mask(ak),
            "sk_masked": mask(sk),
            "sts_masked": mask(sts),
            "ak_len": len(ak),
            "sk_len": len(sk),
            "model_like_endpoint": str(model).startswith("ep-"),
        }

        if auth_mode != "aksk":
            report["status"] = "failed"
            report["error"] = "auth_mode is not aksk"
        elif not ak or not sk:
            report["status"] = "failed"
            report["error"] = "missing ak or sk"
        else:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with OK only."},
                ],
                "temperature": 0,
                "max_tokens": 8,
            }
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            u = urlsplit(url)
            path = u.path or "/"
            query = dict(parse_qsl(u.query, keep_blank_values=True))
            headers = {
                "Host": u.netloc,
                "Content-Type": "application/json; charset=utf-8",
            }

            try:
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
                    session_token=sts or None,
                )
                resp = requests.post(url, headers=headers, data=body.encode("utf-8"), timeout=args.timeout_sec)

                report["checks"]["signed_headers"] = sorted(list(headers.keys()))
                report["checks"]["authorization_prefix"] = headers.get("Authorization", "")[:24]
                report["http_status"] = resp.status_code
                report["response_text_head"] = resp.text[:400]

                if resp.status_code == 200:
                    report["status"] = "ok"
                elif resp.status_code == 401:
                    report["status"] = "failed"
                    report["error"] = "401 unauthorized: check ak/sk pair, endpoint ownership/region, and service authorization"
                else:
                    report["status"] = "failed"
                    report["error"] = f"unexpected status: {resp.status_code}"
            except Exception as e:
                report["status"] = "failed"
                report["error"] = str(e)

    out_json = out_dir / f"doubao_aksk_diag_{ts}.json"
    out_latest = out_dir / "doubao_aksk_diag_latest.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_latest, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OUT] {out_json}")
    print(f"[OUT] {out_latest}")
    print(f"[STATUS] {report.get('status')}")
    if report.get("error"):
        print(f"[ERROR] {report.get('error')}")


if __name__ == "__main__":
    main()
