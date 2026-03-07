import json
from collections import Counter, defaultdict

def load_wrongs(path):
    wrongs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wrongs.append(json.loads(line))
    return wrongs


def summarize(wrongs):
    total = len(wrongs)
    by_gt = Counter()
    by_pred = Counter()
    samples = []
    for w in wrongs:
        gt = w.get("gt")
        pred = w.get("pred")
        by_gt[gt] += 1
        by_pred[pred] += 1
        samples.append(w)
    print(f"总错误样本: {total}")
    print("按正确答案分布:")
    for k,v in sorted(by_gt.items()):
        print(f"  {k}: {v}")
    print("按模型预测分布:")
    for k,v in sorted(by_pred.items()):
        print(f"  {k}: {v}")
    print("示例错误: (gt -> pred)")
    for w in samples[:10]:
        print(f"  {w.get('question')} | gt={w.get('gt')} pred={w.get('pred')} gen={w.get('gen')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析训练/验证集错误样本")
    parser.add_argument("--wrong_file", type=str, required=True, help="jsonl文件路径")
    args = parser.parse_args()
    wrongs = load_wrongs(args.wrong_file)
    summarize(wrongs)
