#!/usr/bin/env python3
"""Merge existing teacher labels (672 original CMExam) with new teacher labels (490 augmented)
to produce a full 1162-sample teacher-labeled dataset, aligned with the augmented data order."""
import argparse
import json
import hashlib
from pathlib import Path


def question_key(d):
    """Hash of Question+Options to match samples across files."""
    q = str(d.get("Question", "")).strip()
    opts = str(d.get("Options", "")).strip()
    return hashlib.sha1(f"{q}\n{opts}".encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--augmented_data", required=True, help="Full augmented training data (1162)")
    p.add_argument("--existing_teacher", required=True, help="Existing teacher labels (672)")
    p.add_argument("--new_teacher", required=True, help="New teacher labels for 490 new samples")
    p.add_argument("--output", required=True, help="Output: all 1162 with TeacherAnswer")
    args = p.parse_args()

    # Build lookup from existing + new teacher labels
    teacher_map = {}  # question_key -> row with TeacherAnswer

    for path in [args.existing_teacher, args.new_teacher]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                key = question_key(d)
                if d.get("TeacherAnswer"):
                    teacher_map[key] = d

    # Iterate over full augmented data, attach teacher labels
    matched = 0
    missing = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.augmented_data, encoding="utf-8") as rf, \
         output.open("w", encoding="utf-8") as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = question_key(d)
            if key in teacher_map:
                teacher = teacher_map[key]
                d["TeacherAnswer"] = teacher["TeacherAnswer"]
                d["TeacherRaw"] = teacher.get("TeacherRaw", "")
                d["OriginalAnswer"] = d.get("Answer", "")
                matched += 1
            else:
                missing += 1
            wf.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(json.dumps({
        "total_augmented": matched + missing,
        "matched_teacher": matched,
        "missing_teacher": missing,
        "output": str(output),
    }, ensure_ascii=False, indent=2))

    if missing > 0:
        print(f"[WARNING] {missing} samples have no teacher label - will be ignored during distillation")


if __name__ == "__main__":
    main()
