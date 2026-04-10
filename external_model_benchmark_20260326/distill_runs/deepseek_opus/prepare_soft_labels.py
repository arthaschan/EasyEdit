#!/usr/bin/env python3
"""Convert DeepSeek hard teacher labels (TeacherAnswer) to soft TeacherDist format
compatible with build_selective_distill_dataset.py and choice-head distillation."""
import argparse
import json
import math
from pathlib import Path

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def hard_to_soft(teacher_ans: str, smooth_eps: float = 0.25):
    """Convert hard label to smoothed soft distribution."""
    teacher_ans = teacher_ans.strip().upper()
    if teacher_ans not in OPTION_LETTERS:
        return None
    dist = {}
    for k in OPTION_LETTERS:
        if k == teacher_ans:
            dist[k] = 1.0 - smooth_eps + smooth_eps / len(OPTION_LETTERS)
        else:
            dist[k] = smooth_eps / len(OPTION_LETTERS)
    return dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="DeepSeek teacher_train_selective.jsonl")
    p.add_argument("--output", required=True, help="Output with TeacherDist added")
    p.add_argument("--smooth_eps", type=float, default=0.25,
                   help="Label smoothing epsilon (default: 0.25)")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    converted = 0
    clean = 0
    mismatch = 0

    with inp.open("r", encoding="utf-8") as rf, out.open("w", encoding="utf-8") as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            t_ans = row.get("TeacherAnswer", "").strip().upper()
            gt_ans = row.get("Answer", row.get("OriginalAnswer", "")).strip().upper()

            dist = hard_to_soft(t_ans, args.smooth_eps)
            if dist is not None:
                row["TeacherDist"] = dist
                converted += 1
                if t_ans == gt_ans:
                    clean += 1
                else:
                    mismatch += 1

            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    ent = -sum(v * math.log(v + 1e-12) for v in hard_to_soft("A", args.smooth_eps).values())
    print(json.dumps({
        "total": total,
        "converted": converted,
        "clean": clean,
        "mismatch": mismatch,
        "smooth_eps": args.smooth_eps,
        "per_sample_entropy": round(ent, 4),
        "output": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
