# DeepSeek->Qwen2.5-7B 选择性蒸馏调优日志

## 背景

- 学生基线准确率（不蒸馏）：77.11%
- 老师（DeepSeek）直接测试准确率：87.95%
- 首轮蒸馏最好结果：75.90%（低于学生基线）

## 问题判断

- 伪标签与原始答案存在冲突，直接全量替换会把噪声注入学生模型。
- 已统计：teacher_eq_original = 577/672，约 85.86%。

## 本轮优化思路（已执行）

- 采用“选择性蒸馏”：
  - 一致样本（TeacherAnswer==OriginalAnswer）：保留教师标签
  - 冲突样本（TeacherAnswer!=OriginalAnswer）：回退使用 OriginalAnswer 作为锚点
- 目标：减少噪声蒸馏，先恢复并超过 77.11% 基线

## 参数搜索设置（Round-2 局部搜索）

- 固定：temperature=1.5，rank=16，lora_alpha=32，batch_size=2，grad_acc=4，lr=1.5e-4，epochs=3
- 搜索：alpha in [0.15, 0.20, 0.25, 0.30]
- 参数文件：`grid_params.json`

## 运行指令

```bash
cd /home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs/deepseek_v3_selective
./run_selective_grid.sh
```

## 运行产物

- 运行目录：`runs/<RUN_ID>/`
- 主日志：`runs/<RUN_ID>/logs/selective_master.log`
- 每组训练日志：`runs/<RUN_ID>/logs/train_*.log`
- 汇总结果：
  - `runs/<RUN_ID>/selective_results_latest.json`
  - `runs/<RUN_ID>/selective_results_latest.md`

## 执行状态

- 状态：已完成
- run_id：20260405_152319
- 主日志：`runs/20260405_152319/logs/selective_master.log`
- 结果文件（完成后）：
  - `runs/20260405_152319/selective_results_latest.json`
  - `runs/20260405_152319/selective_results_latest.md`
- 结果：已回填（见下表）

## 结果回填

| Rank | 组合 | alpha | temperature | lr | Accuracy(%) | Delta vs 77.11 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | a015_t15 | 0.15 | 1.50 | 1.5e-4 | 77.11 | +0.00 |
| 2 | a025_t15 | 0.25 | 1.50 | 1.5e-4 | 75.90 | -1.21 |
| 3 | a020_t15 | 0.20 | 1.50 | 1.5e-4 | 74.70 | -2.41 |
| 4 | a030_t15 | 0.30 | 1.50 | 1.5e-4 | 74.70 | -2.41 |

结论：本轮选择性蒸馏最佳仅追平基线（77.11%），未实现超越。

## 下一步建议（执行版）

- 采用两阶段训练替代“单阶段蒸馏”：
  - Stage-1：仅用原始真值监督训练 1 epoch（不加蒸馏 KL）
  - Stage-2：在一致样本上启用蒸馏，冲突样本继续用原始真值
- 第二轮参数搜索：
  - alpha: [0.05, 0.10, 0.15, 0.20]
  - learning_rate: [1.0e-4, 1.5e-4, 2.0e-4]
  - temperature 固定 1.5
- 增加真值锚点比例：每个 batch 强制混入 20%-30% 原始真值样本，抑制教师噪声扩散。

## Round-3 两阶段局部搜索（已启动）

目标：在不降低基线 77.11% 的前提下，尝试获得正增益。

两阶段流程：

- Stage-1（稳态预热）：仅原始真值训练 1 epoch，alpha=0，default_distill_mask=0
- Stage-2（选择性蒸馏）：使用 selective 数据继续训练，
  - clean_teacher 样本启用 KL
  - mismatch_gt 样本禁用 KL（仅 CE）

搜索空间（12组）：

- alpha: [0.05, 0.10, 0.15, 0.20]
- learning_rate: [1.0e-4, 1.5e-4, 2.0e-4]
- temperature: 1.5（固定）

执行脚本：

- `run_two_stage_round3.sh`

结果产物（完成后）：

- `runs/<RUN_ID>/two_stage_results_latest.json`
- `runs/<RUN_ID>/two_stage_results_latest.md`

本次执行实例：

- run_id: `20260406_200549`
- 主日志: `runs/20260406_200549/logs/two_stage_master.log`
- 结果文件（完成后）:
  - `runs/20260406_200549/two_stage_results_latest.json`
  - `runs/20260406_200549/two_stage_results_latest.md`

## Round-3 自动回填结果

- 未产生有效结果，请检查日志。

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta |
|---:|---|---:|---:|---:|---:|---:|

## Round-3 失败原因定位与修复

- 现象：12组均显示 failed，stage1 成功，stage2 全部退出码 1。
- 根因：`train_dental_lora7.py` 在函数内存在 `from peft import PeftModel` 局部导入，触发 Python 作用域规则，导致恢复断点时抛出：
  - `UnboundLocalError: local variable 'PeftModel' referenced before assignment`
- 影响：本轮 20260406_200549 的结果无效（属于代码异常，不代表参数组合无效）。
- 修复：移除函数内重复导入，保留文件顶部统一导入，已通过 `py_compile` 校验。

## Round-3 重新执行（修复后）

- 新 run_id：`20260407_094814`
- 主日志：`runs/20260407_094814/logs/two_stage_master.log`
- 结果文件（完成后）：
  - `runs/20260407_094814/two_stage_results_latest.json`
  - `runs/20260407_094814/two_stage_results_latest.md`

## Round-3 修复后自动回填结果

- run_id: 20260407_094814
- 汇总文件: runs/20260407_094814/two_stage_results_latest.md
- 论文小结: runs/20260407_094814/thesis_ready_summary.md
- 最优组合: a015_lr10 | acc=79.52% | delta=+2.41

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta |
|---:|---|---:|---:|---:|---:|---:|
| 1 | a015_lr10 | 0.15 | 1.50 | 0.00010 | 79.52 | +2.41 |
| 2 | a020_lr15 | 0.20 | 1.50 | 0.00015 | 78.31 | +1.20 |
| 3 | a010_lr15 | 0.10 | 1.50 | 0.00015 | 77.11 | +0.00 |
| 4 | a020_lr20 | 0.20 | 1.50 | 0.00020 | 77.11 | +0.00 |
| 5 | a005_lr10 | 0.05 | 1.50 | 0.00010 | 73.49 | -3.62 |
| 6 | a005_lr15 | 0.05 | 1.50 | 0.00015 | 73.49 | -3.62 |
| 7 | a010_lr10 | 0.10 | 1.50 | 0.00010 | 73.49 | -3.62 |
| 8 | a005_lr20 | 0.05 | 1.50 | 0.00020 | 72.29 | -4.82 |
| 9 | a015_lr20 | 0.15 | 1.50 | 0.00020 | 72.29 | -4.82 |
| 10 | a010_lr20 | 0.10 | 1.50 | 0.00020 | 71.08 | -6.03 |
| 11 | a015_lr15 | 0.15 | 1.50 | 0.00015 | 71.08 | -6.03 |
| 12 | a020_lr10 | 0.20 | 1.50 | 0.00010 | 71.08 | -6.03 |

## Round-4 Narrow 自动回填结果

- run_id: 20260407_150748
- 汇总文件: runs/20260407_150748/two_stage_results_latest.md
- 论文小结: runs/20260407_150748/thesis_ready_summary.md
- 最优组合: a015_lr05 | acc=75.90% | delta=-1.21

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta |
|---:|---|---:|---:|---:|---:|---:|
| 1 | a015_lr05 | 0.15 | 1.50 | 0.00005 | 75.90 | -1.21 |
| 2 | a015_lr10 | 0.15 | 1.50 | 0.00010 | 74.70 | -2.41 |
| 3 | a013_lr10 | 0.13 | 1.50 | 0.00010 | 73.49 | -3.62 |
| 4 | a017_lr10 | 0.17 | 1.50 | 0.00010 | 73.49 | -3.62 |
