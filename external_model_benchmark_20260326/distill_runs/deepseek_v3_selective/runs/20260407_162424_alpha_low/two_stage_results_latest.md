# DeepSeek Two-Stage Selective Distillation Results

- baseline_student_acc: 77.11%

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta vs Baseline | Status |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | a012_lr10 | 0.12 | 1.50 | 0.00010 | 77.11 | +0.00 | ok |
| 2 | a010_lr10 | 0.10 | 1.50 | 0.00010 | 75.90 | -1.21 | ok |
| 3 | a014_lr10 | 0.14 | 1.50 | 0.00010 | 71.08 | -6.03 | ok |
| 4 | a011_lr10 | 0.11 | 1.50 | 0.00010 | 68.67 | -8.44 | ok |

## Next Suggestions

- Best combo: a012_lr10 (alpha=0.12, temp=1.5, lr=0.0001) => 77.11%.
- If best still <= baseline, reduce alpha further (0.00-0.10) and raise GT anchor ratio.
- If best > baseline, run narrow search around best with lr +/-5e-5.
