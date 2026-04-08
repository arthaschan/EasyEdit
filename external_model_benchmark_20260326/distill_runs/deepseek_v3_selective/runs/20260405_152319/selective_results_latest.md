# DeepSeek Selective Distillation Results

- baseline_student_acc: 77.11%

| Rank | Name | alpha | temp | lr | Accuracy(%) | Delta vs Baseline | Status |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | a015_t15 | 0.15 | 1.50 | 0.00015 | 77.11 | +0.00 | ok |
| 2 | a025_t15 | 0.25 | 1.50 | 0.00015 | 75.90 | -1.21 | ok |
| 3 | a020_t15 | 0.20 | 1.50 | 0.00015 | 74.70 | -2.41 | ok |
| 4 | a030_t15 | 0.30 | 1.50 | 0.00015 | 74.70 | -2.41 | ok |

## Next Suggestions

- Best combo: a015_t15 (alpha=0.15, temp=1.5, lr=0.00015) => 77.11%.
- Round-2 local search: alpha +/-0.05 around best, lr in [1e-4, 2e-4], temp fixed at 1.5.
- Keep selective rule: conflict samples use OriginalAnswer as anchor.
