#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_data", required=True)
    parser.add_argument("--output_train", required=True)
    parser.add_argument("--output_clean", required=True)
    parser.add_argument("--output_mismatch", required=True)
    args = parser.parse_args()

    teacher_rows = load_jsonl(Path(args.teacher_data))

    merged = []
    clean = []
    mismatch = []

    for row in teacher_rows:
        t = str(row.get("TeacherAnswer") or row.get("Answer") or "").strip().upper()
        o = str(row.get("OriginalAnswer") or "").strip().upper()

        if t in {"A", "B", "C", "D", "E"} and o in {"A", "B", "C", "D", "E"} and t == o:
            item = dict(row)
            item["Answer"] = t
            item["SelectiveSource"] = "clean_teacher"
            clean.append(item)
            merged.append(item)
        else:
            item = dict(row)
            # conflict samples fall back to ground truth answer
            if o in {"A", "B", "C", "D", "E"}:
                item["Answer"] = o
            item["SelectiveSource"] = "mismatch_gt"
            mismatch.append(item)
            merged.append(item)

    write_jsonl(Path(args.output_train), merged)
    write_jsonl(Path(args.output_clean), clean)
    write_jsonl(Path(args.output_mismatch), mismatch)

    total = len(merged)
    clean_n = len(clean)
    mismatch_n = len(mismatch)
    print(f"[OUT] train={args.output_train}")
    print(f"[OUT] clean={args.output_clean}")
    print(f"[OUT] mismatch={args.output_mismatch}")
    print(f"[STATS] total={total} clean={clean_n} mismatch={mismatch_n} clean_ratio={clean_n/max(1,total):.4f}")


if __name__ == "__main__":
    main()
