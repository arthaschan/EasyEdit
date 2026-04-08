# DeepSeek Two-Stage Selective Distillation Results

- baseline_student_acc: 77.11%

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta vs Baseline | Status |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | a008_lr15 | 0.08 | 1.50 | 0.00015 | 73.49 | -3.62 | ok |
| 2 | a012_lr15 | 0.12 | 1.50 | 0.00015 | 73.49 | -3.62 | ok |
| 3 | a016_lr15 | 0.16 | 1.50 | 0.00015 | 73.49 | -3.62 | ok |
| 4 | a004_lr10 | 0.04 | 1.50 | 0.00010 | 72.29 | -4.82 | ok |
| 5 | a004_lr15 | 0.04 | 1.50 | 0.00015 | 72.29 | -4.82 | ok |
| 6 | a012_lr10 | 0.12 | 1.50 | 0.00010 | 72.29 | -4.82 | ok |
| 7 | a008_lr10 | 0.08 | 1.50 | 0.00010 | 71.08 | -6.03 | ok |
| 8 | a016_lr10 | 0.16 | 1.50 | 0.00010 | 69.88 | -7.23 | ok |

## Next Suggestions

- Best combo: a008_lr15 (alpha=0.08, temp=1.5, lr=0.00015) => 73.49%.
- If best still <= baseline, reduce alpha further (0.00-0.10) and raise GT anchor ratio.
- If best > baseline, run narrow search around best with lr +/-5e-5.
