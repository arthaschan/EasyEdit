# DeepSeek Two-Stage Selective Distillation Results

- baseline_student_acc: 77.11%

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta vs Baseline | Status |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | a008_lr10 | 0.08 | 1.50 | 0.00010 | 79.52 | +2.41 | ok |
| 2 | a004_lr10 | 0.04 | 1.50 | 0.00010 | 75.90 | -1.21 | ok |
| 3 | a006_lr10 | 0.06 | 1.50 | 0.00010 | 75.90 | -1.21 | ok |
| 4 | a002_lr10 | 0.02 | 1.50 | 0.00010 | 74.70 | -2.41 | ok |

## Next Suggestions

- Best combo: a008_lr10 (alpha=0.08, temp=1.5, lr=0.0001) => 79.52%.
- If best still <= baseline, reduce alpha further (0.00-0.10) and raise GT anchor ratio.
- If best > baseline, run narrow search around best with lr +/-5e-5.
