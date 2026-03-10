import json
from collections import Counter, defaultdict


TOPIC_KEYWORDS = {
    "解剖": [
        "下颌", "上颌", "舌", "颞", "翼外肌", "牙弓", "牙槽", "神经", "关节", "解剖"
    ],
    "修复": [
        "修复", "义齿", "烤瓷", "全冠", "嵌体", "桩核", "固位", "基牙", "洞形", "充填"
    ],
    "儿牙": [
        "小儿", "儿童", "乳牙", "恒牙", "萌出", "方颅", "佝偻病", "发育", "10个月", "4岁"
    ],
    "病理": [
        "炎", "感染", "溃疡", "肿瘤", "结核", "坏死", "休克", "DIC", "病理", "癌"
    ],
}

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


def classify_error(w):
    """将错误分为格式错误、知识错误或其他错误。"""
    pred = (w.get("pred") or "").strip().upper()
    gen = (w.get("gen") or "").strip()

    # 没有抽取到有效选项，通常是输出格式问题
    if pred == "":
        return "format_error"

    # 输出存在，但包含多余描述，通常是格式约束不稳定
    if len(gen) > 3 and any(ch in gen for ch in ["因为", "所以", "解析", "答案", "。", "，"]):
        return "format_error"

    # 有合法选项但选错，多数属于知识/推理错误
    if pred in ["A", "B", "C", "D", "E"]:
        return "knowledge_error"

    return "other_error"


def cluster_errors(wrongs):
    clustered = defaultdict(list)
    for w in wrongs:
        clustered[classify_error(w)].append(w)
    return clustered


def detect_topic(text: str) -> str:
    """按关键词将题目分桶到解剖/修复/儿牙/病理/其他。"""
    src = text or ""
    score = {k: 0 for k in TOPIC_KEYWORDS.keys()}
    for topic, kws in TOPIC_KEYWORDS.items():
        for kw in kws:
            if kw in src:
                score[topic] += 1
    best_topic = "其他"
    best_score = 0
    for topic, s in score.items():
        if s > best_score:
            best_score = s
            best_topic = topic
    return best_topic


def topic_weak_report(wrongs):
    """输出按题型关键词分桶的薄弱点报告。"""
    topic_counter = Counter()
    topic_examples = defaultdict(list)

    for w in wrongs:
        q = w.get("question", "")
        opts = w.get("options", "")
        topic = detect_topic(f"{q}\n{opts}")
        topic_counter[topic] += 1
        if len(topic_examples[topic]) < 5:
            topic_examples[topic].append(w)

    total = sum(topic_counter.values())
    print("\n题型薄弱点报告（按错题关键词分桶）:")
    for topic, cnt in topic_counter.most_common():
        ratio = (100.0 * cnt / total) if total else 0.0
        print(f"  {topic}: {cnt} ({ratio:.2f}%)")

    if total > 0:
        top_topic, top_cnt = topic_counter.most_common(1)[0]
        top_ratio = 100.0 * top_cnt / total
        print("\n最该补的数据类型建议:")
        print(f"  优先补 `{top_topic}` 相关题目（当前占错题 {top_ratio:.2f}%）。")
        if len(topic_counter) > 1:
            second_topic, second_cnt = topic_counter.most_common(2)[1]
            second_ratio = 100.0 * second_cnt / total
            print(f"  次优先补 `{second_topic}`（占错题 {second_ratio:.2f}%）。")

    print("\n各题型示例错题（最多5条）:")
    for topic, examples in topic_examples.items():
        print(f"\n[{topic}]")
        for w in examples:
            print(f"  Q: {w.get('question', '')} | gt={w.get('gt')} pred={w.get('pred')}")

    return {
        "topic_counts": dict(topic_counter),
        "top_topic": topic_counter.most_common(1)[0][0] if total else "其他",
    }


def print_cluster_summary(clustered):
    total = sum(len(v) for v in clustered.values())
    print("\n错误聚类结果:")
    for k in ["format_error", "knowledge_error", "other_error"]:
        cnt = len(clustered.get(k, []))
        ratio = (100.0 * cnt / total) if total else 0.0
        print(f"  {k}: {cnt} ({ratio:.2f}%)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析训练/验证集错误样本")
    parser.add_argument("--wrong_file", type=str, required=True, help="jsonl文件路径")
    parser.add_argument("--summary_out", type=str, default="", help="可选：导出聚类摘要json路径")
    args = parser.parse_args()
    wrongs = load_wrongs(args.wrong_file)
    summarize(wrongs)
    clustered = cluster_errors(wrongs)
    print_cluster_summary(clustered)
    topic_summary = topic_weak_report(wrongs)

    if args.summary_out:
        out = {
            "total": len(wrongs),
            "clusters": {k: len(v) for k, v in clustered.items()},
            "topics": topic_summary,
        }
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"已导出聚类摘要到: {args.summary_out}")
