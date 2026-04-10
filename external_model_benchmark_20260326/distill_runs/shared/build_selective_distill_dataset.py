#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
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
    return q + "\n" + opt_text


def normalize_dist(raw):
    vals = []
    for k in OPTION_LETTERS:
        try:
            vals.append(float(raw.get(k, 0.0)))
        except Exception:
            vals.append(0.0)
    s = sum(max(0.0, v) for v in vals)
    if s <= 0:
        return None
    vals = [max(0.0, v) / s for v in vals]
    return {k: vals[i] for i, k in enumerate(OPTION_LETTERS)}


def sharpen_or_smooth(dist, min_entropy, smooth_eps):
    vals = [float(dist[k]) for k in OPTION_LETTERS]
    ent = -sum(v * math.log(v + 1e-12) for v in vals)
    if ent >= min_entropy:
        return dist, ent

    # Label-smoothing style flattening to avoid one-hot collapse.
    eps = max(0.0, min(0.4, float(smooth_eps)))
    vals2 = [(1.0 - eps) * v + eps / len(vals) for v in vals]
    s2 = sum(vals2)
    vals2 = [v / s2 for v in vals2]
    ent2 = -sum(v * math.log(v + 1e-12) for v in vals2)
    return {k: vals2[i] for i, k in enumerate(OPTION_LETTERS)}, ent2


def argmax_letter(dist):
    return max(OPTION_LETTERS, key=lambda k: float(dist.get(k, 0.0)))


def margin_top2(dist):
    vals = sorted([float(dist.get(k, 0.0)) for k in OPTION_LETTERS], reverse=True)
    if len(vals) < 2:
        return 0.0
    return vals[0] - vals[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_data", required=True)
    p.add_argument("--teacher_soft", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--min_entropy", type=float, default=0.35)
    p.add_argument("--smooth_eps", type=float, default=0.20)
    p.add_argument("--min_margin", type=float, default=0.10)
    p.add_argument("--only_disagree", action="store_true")
    args = p.parse_args()

    gt_rows = load_jsonl(Path(args.gt_data))
    soft_rows = load_jsonl(Path(args.teacher_soft))

    soft_map = {}
    for r in soft_rows:
        key = sample_key(r)
        dist = r.get("TeacherDist", {})
        if not isinstance(dist, dict):
            continue
        nd = normalize_dist(dist)
        if nd is None:
            continue
        soft_map[key] = nd

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0
    selected = 0
    disagree = 0
    entropy_vals = []
    peak_vals = []

    with out_path.open("w", encoding="utf-8") as wf:
        for row in gt_rows:
            total += 1
            out = dict(row)
            key = sample_key(row)
            gt_ans = str(row.get("Answer", "")).strip().upper()

            dist = soft_map.get(key)
            if dist is None:
                out.pop("TeacherDist", None)
                out.pop("SelectiveSource", None)
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue

            matched += 1
            dist2, ent = sharpen_or_smooth(dist, args.min_entropy, args.smooth_eps)
            t_ans = argmax_letter(dist2)
            peak = max(float(dist2[k]) for k in OPTION_LETTERS)
            mar = margin_top2(dist2)
            is_disagree = (gt_ans in OPTION_LETTERS) and (t_ans != gt_ans)
            if is_disagree:
                disagree += 1

            pass_sel = (ent >= args.min_entropy) and (mar >= args.min_margin)
            if args.only_disagree:
                pass_sel = pass_sel and is_disagree

            if pass_sel:
                selected += 1
                out["TeacherDist"] = dist2
                out["SelectiveSource"] = "clean_teacher"
                out["TeacherAnswer"] = t_ans
                out["TeacherEntropy"] = round(ent, 6)
                out["TeacherPeak"] = round(peak, 6)
                entropy_vals.append(ent)
                peak_vals.append(peak)
            else:
                out.pop("TeacherDist", None)
                out.pop("SelectiveSource", None)
                out.pop("TeacherAnswer", None)

            wf.write(json.dumps(out, ensure_ascii=False) + "\n")

    report = {
        "total": total,
        "matched_soft": matched,
        "selected_for_distill": selected,
        "selected_ratio": round(selected / total, 4) if total else 0.0,
        "teacher_disagree_count": disagree,
        "teacher_disagree_ratio": round(disagree / matched, 4) if matched else 0.0,
        "selected_entropy_mean": round(sum(entropy_vals) / len(entropy_vals), 4) if entropy_vals else None,
        "selected_peak_mean": round(sum(peak_vals) / len(peak_vals), 4) if peak_vals else None,
        "min_entropy": args.min_entropy,
        "smooth_eps": args.smooth_eps,
        "min_margin": args.min_margin,
        "only_disagree": bool(args.only_disagree),
        "output": str(out_path),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
