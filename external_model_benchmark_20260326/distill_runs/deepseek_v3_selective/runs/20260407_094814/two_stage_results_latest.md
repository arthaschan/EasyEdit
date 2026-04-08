# DeepSeek Two-Stage Selective Distillation Results

- baseline_student_acc: 77.11%

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta vs Baseline | Status |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | a015_lr10 | 0.15 | 1.50 | 0.00010 | 79.52 | +2.41 | ok |
| 2 | a020_lr15 | 0.20 | 1.50 | 0.00015 | 78.31 | +1.20 | ok |
| 3 | a010_lr15 | 0.10 | 1.50 | 0.00015 | 77.11 | +0.00 | ok |
| 4 | a020_lr20 | 0.20 | 1.50 | 0.00020 | 77.11 | +0.00 | ok |
| 5 | a005_lr10 | 0.05 | 1.50 | 0.00010 | 73.49 | -3.62 | ok |
| 6 | a005_lr15 | 0.05 | 1.50 | 0.00015 | 73.49 | -3.62 | ok |
| 7 | a010_lr10 | 0.10 | 1.50 | 0.00010 | 73.49 | -3.62 | ok |
| 8 | a005_lr20 | 0.05 | 1.50 | 0.00020 | 72.29 | -4.82 | ok |
| 9 | a015_lr20 | 0.15 | 1.50 | 0.00020 | 72.29 | -4.82 | ok |
| 10 | a010_lr20 | 0.10 | 1.50 | 0.00020 | 71.08 | -6.03 | ok |
| 11 | a015_lr15 | 0.15 | 1.50 | 0.00015 | 71.08 | -6.03 | ok |
| 12 | a020_lr10 | 0.20 | 1.50 | 0.00010 | 71.08 | -6.03 | ok |

## Next Suggestions

- Best combo: a015_lr10 (alpha=0.15, temp=1.5, lr=0.0001) => 79.52%.
- If best still <= baseline, reduce alpha further (0.00-0.10) and raise GT anchor ratio.
- If best > baseline, run narrow search around best with lr +/-5e-5.
