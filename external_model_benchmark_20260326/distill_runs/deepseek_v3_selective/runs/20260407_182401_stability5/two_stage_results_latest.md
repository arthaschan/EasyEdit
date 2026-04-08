# DeepSeek Two-Stage Selective Distillation Results

- baseline_student_acc: 77.11%

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta vs Baseline | Status |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | rep4_a015_lr10 | 0.15 | 1.50 | 0.00010 | 74.70 | -2.41 | ok |
| 2 | rep5_a015_lr10 | 0.15 | 1.50 | 0.00010 | 74.70 | -2.41 | ok |
| 3 | rep1_a015_lr10 | 0.15 | 1.50 | 0.00010 | 73.49 | -3.62 | ok |
| 4 | rep2_a015_lr10 | 0.15 | 1.50 | 0.00010 | 72.29 | -4.82 | ok |
| 5 | rep3_a015_lr10 | 0.15 | 1.50 | 0.00010 | 72.29 | -4.82 | ok |

## Next Suggestions

- Best combo: rep4_a015_lr10 (alpha=0.15, temp=1.5, lr=0.0001) => 74.70%.
- If best still <= baseline, reduce alpha further (0.00-0.10) and raise GT anchor ratio.
- If best > baseline, run narrow search around best with lr +/-5e-5.
