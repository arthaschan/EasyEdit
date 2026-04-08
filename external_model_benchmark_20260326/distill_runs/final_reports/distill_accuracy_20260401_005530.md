# Distillation Accuracy Report

- device: cuda:0
- base_model: /home/student/arthas/EasyEdit3/Qwen2.5-7B-Instruct
- test_data: /home/student/arthas/EasyEdit3/data/cmexam_dental_choice_test.jsonl

| Teacher | Accuracy(%) | Correct | Total | Status |
|---|---:|---:|---:|---|
| DeepSeek-V3 | 73.49 | 61 | 83 | ok |
| moonshot-v1-32k | 73.49 | 61 | 83 | ok |
| doubao | 73.49 | 61 | 83 | ok |

## Improvement Suggestions

- Best current teacher: DeepSeek-V3 (73.49%).
- For lower-accuracy teachers, increase label quality filtering: keep only samples where teacher output is a strict single A-E letter.
- Tune distillation weight: try alpha in [0.3, 0.7] and temperature in [1.5, 3.0].
- Use curriculum: first train on high-confidence teacher labels, then full teacher set.
- Add 5-10% ground-truth supervised anchors to reduce teacher-specific bias drift.
- If one teacher underperforms, use weighted teacher mixture rather than pure single-teacher distillation.
