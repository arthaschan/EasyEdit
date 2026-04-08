# 论文可用结果小结（Round-4 Narrow）

- run_id: 20260407_150748
- baseline_student_acc: 77.11%
- result_file: /home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective/runs/20260407_150748/two_stage_results_latest.md

## 关键结论

- 最优组合: a015_lr05 (alpha=0.15, temp=1.5, lr=5e-05)
- 最优准确率: 75.90%
- 相对基线提升: -1.21 个百分点

## 实验解释

- 本轮为最优点邻域窄域搜索，用于验证增益稳定性与参数敏感区间。
- 若最优继续提升，说明上一轮最优附近仍有可挖掘空间。

## 参数结果表（Top）

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta |
|---:|---|---:|---:|---:|---:|---:|
| 1 | a015_lr05 | 0.15 | 1.50 | 0.00005 | 75.90 | -1.21 |
| 2 | a015_lr10 | 0.15 | 1.50 | 0.00010 | 74.70 | -2.41 |
| 3 | a013_lr10 | 0.13 | 1.50 | 0.00010 | 73.49 | -3.62 |
| 4 | a017_lr10 | 0.17 | 1.50 | 0.00010 | 73.49 | -3.62 |
