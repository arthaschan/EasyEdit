#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_acc_from_log(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"测试集准确率:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if not matches:
        return None
    return float(matches[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_root", required=True)
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    root = Path(args.grid_root)
    run_id = args.run_id
    run_root = root / "runs" / run_id
    params = json.loads((root / "grid_params.json").read_text(encoding="utf-8"))

    results = []
    for p in params:
        name = p["name"]
        log_path = run_root / "logs" / f"train_{name}.log"
        acc = parse_acc_from_log(log_path)
        status = "ok" if acc is not None else "failed"
        results.append({
            "name": name,
            "alpha": p["alpha"],
            "temperature": p["temperature"],
            "accuracy": acc,
            "status": status,
            "log": str(log_path),
            "output_dir": str(run_root / "outputs" / name),
        })

    ok = [r for r in results if r["status"] == "ok"]
    ok.sort(key=lambda x: x["accuracy"], reverse=True)

    out_json = run_root / "grid_results_latest.json"
    out_md = run_root / "grid_results_latest.md"

    out_json.write_text(json.dumps({"run_id": run_id, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DeepSeek Distillation Grid Results",
        "",
        f"- run_id: {run_id}",
        "",
        "| Rank | Name | alpha | temperature | Accuracy(%) | Status |",
        "|---:|---|---:|---:|---:|---|",
    ]

    rank = 1
    for r in ok:
        lines.append(f"| {rank} | {r['name']} | {r['alpha']:.2f} | {r['temperature']:.2f} | {r['accuracy']:.2f} | ok |")
        rank += 1

    for r in results:
        if r["status"] != "ok":
            lines.append(f"| - | {r['name']} | {r['alpha']:.2f} | {r['temperature']:.2f} | - | failed |")

    lines += ["", "## Suggestions", ""]
    if ok:
        best = ok[0]
        lines.append(f"- Best combo now: {best['name']} (alpha={best['alpha']}, temperature={best['temperature']}) with {best['accuracy']:.2f}%.")
        lines.append("- Next try around best: alpha +/-0.1 and temperature +/-0.5.")
        lines.append("- Keep strict label filter and add 5-10% ground-truth anchors.")
    else:
        lines.append("- No successful run found. Check logs and API quota/network first.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OUT] {out_json}")
    print(f"[OUT] {out_md}")


if __name__ == "__main__":
    main()
