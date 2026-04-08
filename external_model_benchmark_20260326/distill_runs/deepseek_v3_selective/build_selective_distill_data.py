#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.teacher_data)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    clean = 0
    conflict = 0

    with in_path.open("r", encoding="utf-8") as rf, out_path.open("w", encoding="utf-8") as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total += 1

            teacher = str(d.get("TeacherAnswer", "")).strip().upper()
            original = str(d.get("OriginalAnswer", d.get("Answer", ""))).strip().upper()

            d["Answer"] = original
            if teacher in {"A", "B", "C", "D", "E"} and teacher == original:
                d["DistillWeight"] = 1.0
                clean += 1
            else:
                d["DistillWeight"] = 0.0
                conflict += 1

            wf.write(json.dumps(d, ensure_ascii=False) + "\n")

    agreement = 100.0 * clean / total if total else 0.0
    print(f"[DONE] total={total} clean={clean} conflict={conflict} agreement={agreement:.2f}% output={out_path}")


if __name__ == "__main__":
    main()
