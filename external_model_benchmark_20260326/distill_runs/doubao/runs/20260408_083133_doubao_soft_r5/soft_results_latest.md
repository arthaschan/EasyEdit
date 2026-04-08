# Doubao Soft Distill Round5 Results

- baseline_student_acc: 77.11%
- previous_best_acc: 80.72%

| Rank | Name | lr | epochs | rank | lora_alpha | alpha | seed | Accuracy(%) | Delta vs Baseline | Delta vs PrevBest | Status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | soft5_a20_lr15e4_r32_s42 | 0.00015 | 3 | 32 | 64 | 0.20 | 42 | 77.11 | +0.00 | -3.61 | ok |
| 2 | soft5_a20_lr12e4_r24_s42 | 0.00012 | 3 | 24 | 48 | 0.20 | 42 | 77.11 | +0.00 | -3.61 | ok |
| 3 | soft5_a35_lr15e4_r32_s42 | 0.00015 | 3 | 32 | 64 | 0.35 | 42 | 74.70 | -2.41 | -6.02 | ok |
| 4 | soft5_a35_lr12e4_r24_s52 | 0.00012 | 3 | 24 | 48 | 0.35 | 52 | 71.08 | -6.03 | -9.64 | ok |
