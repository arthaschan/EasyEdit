# Teacher-to-Student Distillation Runs

This directory contains three independent distillation pipelines:
- DeepSeek-V3 -> Qwen2.5-7B-Instruct
- moonshot-v1-32k -> Qwen2.5-7B-Instruct
- doubao -> Qwen2.5-7B-Instruct

## Structure

- `shared/generate_teacher_labels.py`: calls teacher API, produces teacher-labeled JSONL
- `<teacher>/teacher_candidate.json`: one teacher API config
- `<teacher>/run_distill_train.sh`: generate teacher labels and then train student model
- `run_all_teacher_distill.sh`: run all three pipelines sequentially

## Run

```bash
cd /home/student/arthas/EasyEdit3/external_model_benchmark_20260326/distill_runs
bash run_all_teacher_distill.sh
```

## Notes

- Each teacher run writes logs under `<teacher>/logs/`
- Teacher-labeled dataset is saved under `<teacher>/artifacts/teacher_train.jsonl`
- Student model output is saved under `<teacher>/outputs/`
- Resume support:
	- Teacher labeling uses `--resume` and appends into the same `teacher_train.jsonl`, skipping already processed samples.
	- Student training saves epoch checkpoints under `<teacher>/outputs/.../checkpoints/epoch_*` and rerun uses `--resume` to continue.
