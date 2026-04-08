#!/usr/bin/env bash
set -euo pipefail

RUN_ID="20260407_094814"
ROOT="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective"
RUN_JSON="$ROOT/runs/$RUN_ID/two_stage_results_latest.json"
RUN_MD="$ROOT/runs/$RUN_ID/two_stage_results_latest.md"
THESIS_MD="$ROOT/runs/$RUN_ID/thesis_ready_summary.md"
JOURNAL="$ROOT/tuning_journal.md"
PY="/home/student/arthas/EasyEdit3/.venv/bin/python"

while [[ ! -f "$RUN_JSON" ]]; do
  sleep 60
done

"$PY" - <<'PY'
import json
from pathlib import Path

run_id = '20260407_094814'
root = Path('/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective')
run_json = root / 'runs' / run_id / 'two_stage_results_latest.json'
run_md = root / 'runs' / run_id / 'two_stage_results_latest.md'
thesis_md = root / 'runs' / run_id / 'thesis_ready_summary.md'
journal = root / 'tuning_journal.md'

obj = json.loads(run_json.read_text(encoding='utf-8'))
baseline = float(obj.get('baseline', 77.11))
rows = [r for r in (obj.get('results') or []) if r.get('status') == 'ok' and r.get('accuracy') is not None]
rows.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
best = rows[0] if rows else None

lines = []
lines.append('# 论文可用结果小结（修复后Round-3）')
lines.append('')
lines.append(f'- run_id: {run_id}')
lines.append(f'- baseline_student_acc: {baseline:.2f}%')
lines.append(f'- result_file: {run_md}')
lines.append('')

if best:
    lines.append('## 关键结论')
    lines.append('')
    lines.append(f"- 最优组合: {best.get('name')} (alpha={best.get('alpha')}, temp={best.get('temperature')}, lr={best.get('learning_rate')})")
    lines.append(f"- 最优准确率: {best.get('accuracy'):.2f}%")
    lines.append(f"- 相对基线提升: {best.get('delta_vs_baseline'):+.2f} 个百分点")
    lines.append('')
    lines.append('## 实验解释')
    lines.append('')
    lines.append('- 两阶段策略先用真值稳定学生，再在高置信样本上蒸馏，降低了教师噪声传播风险。')
    lines.append('- 若仍未超基线，说明教师增益被噪声/容量上限抵消，需要进一步降低蒸馏权重或提高真值锚点比例。')
else:
    lines.append('## 关键结论')
    lines.append('')
    lines.append('- 本轮未产出有效结果，请检查 logs 中的失败样例并重跑。')

lines.append('')
lines.append('## 参数结果表（Top）')
lines.append('')
lines.append('| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta |')
lines.append('|---:|---|---:|---:|---:|---:|---:|')
for i, r in enumerate(rows[:12], start=1):
    lines.append(f"| {i} | {r.get('name')} | {r.get('alpha'):.2f} | {r.get('temperature'):.2f} | {r.get('learning_rate'):.5f} | {r.get('accuracy'):.2f} | {r.get('delta_vs_baseline'):+.2f} |")

thesis_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

# update journal
marker = '## Round-3 修复后自动回填结果'
section = []
section.append(marker)
section.append('')
section.append(f'- run_id: {run_id}')
section.append(f'- 汇总文件: runs/{run_id}/two_stage_results_latest.md')
section.append(f'- 论文小结: runs/{run_id}/thesis_ready_summary.md')
if best:
    section.append(f"- 最优组合: {best.get('name')} | acc={best.get('accuracy'):.2f}% | delta={best.get('delta_vs_baseline'):+.2f}")
else:
    section.append('- 本轮未产出有效结果。')
section.append('')
section.append('| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta |')
section.append('|---:|---|---:|---:|---:|---:|---:|')
for i, r in enumerate(rows[:12], start=1):
    section.append(f"| {i} | {r.get('name')} | {r.get('alpha'):.2f} | {r.get('temperature'):.2f} | {r.get('learning_rate'):.5f} | {r.get('accuracy'):.2f} | {r.get('delta_vs_baseline'):+.2f} |")
section_text = '\n'.join(section) + '\n'

text = journal.read_text(encoding='utf-8')
if marker in text:
    text = text.split(marker)[0].rstrip() + '\n\n' + section_text
else:
    text = text.rstrip() + '\n\n' + section_text
journal.write_text(text, encoding='utf-8')
print('[DONE] thesis summary and journal updated')
PY
