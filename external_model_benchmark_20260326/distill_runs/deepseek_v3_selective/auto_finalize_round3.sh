#!/usr/bin/env bash
set -euo pipefail

RUN_JSON="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective/runs/20260406_200549/two_stage_results_latest.json"
JOURNAL="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective/tuning_journal.md"
PY="/home/student/arthas/EasyEdit3/.venv/bin/python"

while [[ ! -f "$RUN_JSON" ]]; do
  sleep 60
done

"$PY" - <<'PY'
import json
from pathlib import Path
run_json=Path('/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective/runs/20260406_200549/two_stage_results_latest.json')
journal=Path('/home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective/tuning_journal.md')
obj=json.loads(run_json.read_text(encoding='utf-8'))
best=obj.get('best') or {}
results=[r for r in (obj.get('results') or []) if r.get('status')=='ok']
results.sort(key=lambda x:x.get('accuracy',0), reverse=True)
lines=[]
lines.append('## Round-3 自动回填结果')
lines.append('')
if best:
    lines.append(f"- 最优组合: {best.get('name')} (alpha={best.get('alpha')}, temp={best.get('temperature')}, lr={best.get('learning_rate')})")
    lines.append(f"- 最优准确率: {best.get('accuracy')}%")
    lines.append(f"- 相对基线(77.11): {best.get('delta_vs_baseline'):+.2f}")
else:
    lines.append('- 未产生有效结果，请检查日志。')
lines.append('')
lines.append('| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta |')
lines.append('|---:|---|---:|---:|---:|---:|---:|')
for i,r in enumerate(results[:12], start=1):
    lines.append(f"| {i} | {r.get('name')} | {r.get('alpha'):.2f} | {r.get('temperature'):.2f} | {r.get('learning_rate'):.5f} | {r.get('accuracy'):.2f} | {r.get('delta_vs_baseline'):+.2f} |")
block='\n'.join(lines)+'\n'
text=journal.read_text(encoding='utf-8')
marker='## Round-3 自动回填结果'
if marker in text:
    text=text.split(marker)[0].rstrip()+'\n\n'+block
else:
    text=text.rstrip()+'\n\n'+block
journal.write_text(text, encoding='utf-8')
print('[DONE] journal updated')
PY
