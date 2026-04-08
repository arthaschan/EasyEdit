# Doubao Improve Round3 Results

- baseline_student_acc: 77.11%
- previous_best_acc: 80.72%

| Rank | Name | lr | epochs | rank | lora_alpha | seed | Accuracy(%) | Delta vs Baseline | Delta vs PrevBest | Status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | sft_lr15_ep3_s42 | 0.00015 | 3 | 16 | 32 | 42 | 80.72 | +3.61 | +0.00 | ok |
| 2 | sft_lr13_ep3_s7 | 0.00013 | 3 | 16 | 32 | 7 | 78.31 | +1.20 | -2.41 | ok |
| 3 | sft_lr12_ep4_s33 | 0.00012 | 4 | 16 | 32 | 33 | 77.11 | +0.00 | -3.61 | ok |
| 4 | sft_lr15_ep3_s11 | 0.00015 | 3 | 16 | 32 | 11 | 74.70 | -2.41 | -6.02 | ok |
| 5 | sft_lr15_ep3_r32_s42 | 0.00015 | 3 | 32 | 64 | 42 | 74.70 | -2.41 | -6.02 | ok |
| 6 | sft_lr17_ep3_s21 | 0.00017 | 3 | 16 | 32 | 21 | 73.49 | -3.62 | -7.23 | ok |
| 7 | sft_lr15_ep2_s42 | 0.00015 | 2 | 16 | 32 | 42 | 73.49 | -3.62 | -7.23 | ok |
| 8 | sft_lr10_ep3_r32_s55 | 0.00010 | 3 | 32 | 64 | 55 | 71.08 | -6.03 | -9.64 | ok |
