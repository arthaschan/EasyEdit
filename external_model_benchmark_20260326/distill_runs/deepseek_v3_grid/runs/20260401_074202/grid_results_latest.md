# DeepSeek Distillation Grid Results

- run_id: 20260401_074202

| Rank | Name | alpha | temperature | Accuracy(%) | Status |
|---:|---|---:|---:|---:|---|
| 1 | a03_t15 | 0.30 | 1.50 | 75.90 | ok |
| 2 | a07_t25 | 0.70 | 2.50 | 74.70 | ok |
| 3 | a07_t30 | 0.70 | 3.00 | 74.70 | ok |
| 4 | a05_t20 | 0.50 | 2.00 | 73.49 | ok |
| 5 | a05_t25 | 0.50 | 2.50 | 73.49 | ok |
| 6 | a03_t20 | 0.30 | 2.00 | 72.29 | ok |

## Suggestions

- Best combo now: a03_t15 (alpha=0.3, temperature=1.5) with 75.90%.
- Next try around best: alpha +/-0.1 and temperature +/-0.5.
- Keep strict label filter and add 5-10% ground-truth anchors.
