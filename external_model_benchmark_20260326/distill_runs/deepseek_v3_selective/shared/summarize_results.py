#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path


def parse_acc(log_path: Path):
    if not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.findall(r"测试集准确率:\s*([0-9]+(?:\.[0-9]+)?)%", txt)
    if not m:
        return None
    return float(m[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--baseline", type=float, default=77.11)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    params = json.loads(Path(args.params).read_text(encoding="utf-8"))

    rows = []
    for p in params:
        name = p["name"]
        log = run_root / "logs" / f"train_{name}.log"
        acc = parse_acc(log)
        rows.append({
            "name": name,
            "alpha": p["alpha"],
            "temperature": p["temperature"],
            "learning_rate": p["learning_rate"],
            "accuracy": acc,
            "delta_vs_baseline": None if acc is None else round(acc - args.baseline, 2),
            "status": "ok" if acc is not None else "failed",
            "log": str(log),
        })

    ok = [r for r in rows if r["status"] == "ok"]
    ok.sort(key=lambda x: x["accuracy"], reverse=True)
    best = ok[0] if ok else None

    out_json = run_root / "selective_results_latest.json"
    out_md = run_root / "selective_results_latest.md"

    out_json.write_text(json.dumps({
        "run_time": datetime.now().isoformat(),
        "baseline": args.baseline,
        "results": rows,
        "best": best,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DeepSeek Selective Distillation Results",
        "",
        f"- baseline_student_acc: {args.baseline:.2f}%",
        "",
        "| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta vs Baseline | Status |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]

    rank = 1
    for r in ok:
        lines.append(
            f"| {rank} | {r['name']} | {r['alpha']:.2f} | {r['temperature']:.2f} | {r['learning_rate']:.5f} | {r['accuracy']:.2f} | {r['delta_vs_baseline']:+.2f} | ok |"
        )
        rank += 1

    for r in rows:
        if r["status"] != "ok":
            lines.append(
                f"| - | {r['name']} | {r['alpha']:.2f} | {r['temperature']:.2f} | {r['learning_rate']:.5f} | - | - | failed |"
            )

    lines += ["", "## Next Suggestions", ""]
    if best:
        lines.append(
            f"- Best combo: {best['name']} (alpha={best['alpha']}, temp={best['temperature']}, lr={best['learning_rate']}) => {best['accuracy']:.2f}%."
        )
        lines.append("- Round-2 local search: alpha +/-0.05 around best, lr in [1e-4, 2e-4], temp fixed at 1.5.")
        lines.append("- Keep selective rule: conflict samples use OriginalAnswer as anchor.")
    else:
        lines.append("- No successful run found; inspect logs and environment/API quota.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OUT] {out_json}")
    print(f"[OUT] {out_md}")


if __name__ == "__main__":
    main()
