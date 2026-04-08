# DeepSeek Two-Stage Selective Distillation Results

- baseline_student_acc: 77.11%

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta vs Baseline | Status |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | rep3_a015_lr10 | 0.15 | 1.50 | 0.00010 | 75.90 | -1.21 | ok |
| 2 | rep1_a015_lr10 | 0.15 | 1.50 | 0.00010 | 71.08 | -6.03 | ok |
| 3 | rep2_a015_lr10 | 0.15 | 1.50 | 0.00010 | 71.08 | -6.03 | ok |

## Next Suggestions

- Best combo: rep3_a015_lr10 (alpha=0.15, temp=1.5, lr=0.0001) => 75.90%.
- If best still <= baseline, reduce alpha further (0.00-0.10) and raise GT anchor ratio.
- If best > baseline, run narrow search around best with lr +/-5e-5.
