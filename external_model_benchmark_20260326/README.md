# External Model Benchmark (Unified Folder)

This folder contains all files for benchmarking non-Qwen external teacher models on dental multiple-choice QA.

## Included Models

Configured in `candidates.json`:
- DeepSeek-V3 (`deepseek-chat`)
- ChatGPT-4.1 (`gpt-4.1`)
- Kimi (`moonshot-v1-8k`)
- Doubao (Volcengine Ark endpoint; replace endpoint id)

## Files

- `external_benchmark.py`: unified evaluator for OpenAI-compatible chat APIs.
- `run_benchmark.sh`: one-command runner that writes logs and outputs.
- `candidates.json`: candidate model list and API endpoint config.
- `prompt_system.txt`: strict system prompt (must output A-E only).
- `.env.example`: API key variable template.
- `logs/`: run logs.
- `outputs/`: leaderboard JSON/Markdown outputs.

## Environment Variables

Set keys before running:

```bash
export DEEPSEEK_API_KEY=...
export OPENAI_API_KEY=...
export MOONSHOT_API_KEY=...
export DOUBAO_ARK_API_KEY=...
export DOUBAO_ACCESS_KEY_ID=...
export DOUBAO_SECRET_ACCESS_KEY=...
```

Notes:
- For Doubao, replace `model` in `candidates.json` with your Ark endpoint ID.
- Doubao supports AK/SK signing: set `auth_mode` to `aksk`, then set `access_key_id_env` and `secret_access_key_env`.
- OpenAI free account can be skipped by setting `free_account: true` in `candidates.json`.
- Missing keys are handled safely: the model is marked as `skipped` with reason in outputs.

## Run

```bash
cd external_model_benchmark_20260326
chmod +x run_benchmark.sh
./run_benchmark.sh
```

Optional runtime overrides:

```bash
SAMPLE_SIZE=500 SEED=123 TIMEOUT_SEC=120 MAX_TOKENS=16 ./run_benchmark.sh
```

Optional dataset override:

```bash
DATASET_PATH=/abs/path/to/your_dataset.jsonl ./run_benchmark.sh
```

## Outputs

- `outputs/leaderboard_latest.json`
- `outputs/leaderboard_latest.md`
- `outputs/leaderboard_<timestamp>.json`
- `outputs/leaderboard_<timestamp>.md`
- `logs/run_<timestamp>.log`

The markdown leaderboard is sorted by accuracy (descending).
