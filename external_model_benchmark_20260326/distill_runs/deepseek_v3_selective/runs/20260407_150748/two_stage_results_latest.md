# DeepSeek Two-Stage Selective Distillation Results

- baseline_student_acc: 77.11%

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta vs Baseline | Status |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | a015_lr05 | 0.15 | 1.50 | 0.00005 | 75.90 | -1.21 | ok |
| 2 | a015_lr10 | 0.15 | 1.50 | 0.00010 | 74.70 | -2.41 | ok |
| 3 | a013_lr10 | 0.13 | 1.50 | 0.00010 | 73.49 | -3.62 | ok |
| 4 | a017_lr10 | 0.17 | 1.50 | 0.00010 | 73.49 | -3.62 | ok |

## Next Suggestions

- Best combo: a015_lr05 (alpha=0.15, temp=1.5, lr=5e-05) => 75.90%.
- If best still <= baseline, reduce alpha further (0.00-0.10) and raise GT anchor ratio.
- If best > baseline, run narrow search around best with lr +/-5e-5.
