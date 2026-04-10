#!/usr/bin/env bash
# Master orchestrator: runs Doubao R9 then DeepSeek Opus R1 sequentially
set -euo pipefail

BENCH_DIR="/home/student/arthas/EasyEdit3/external_model_benchmark_20260326"
DOUBAO_DIR="$BENCH_DIR/distill_runs/doubao"
OPUS_DIR="$BENCH_DIR/distill_runs/deepseek_opus"

echo "============================================="
echo "[MASTER] Starting dual distillation training"
echo "[MASTER] $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

echo ""
echo ">>> PHASE 1: Doubao R9 (choice-head → SFT, 5 seeds) <<<"
echo "============================================="
chmod +x "$DOUBAO_DIR/run_two_stage_head_then_sft_round9.sh"
bash "$DOUBAO_DIR/run_two_stage_head_then_sft_round9.sh"
echo ""
echo "[MASTER] Doubao R9 completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo ">>> PHASE 2: DeepSeek Opus R1 (choice-head → SFT, 5 seeds) <<<"
echo "============================================="
chmod +x "$OPUS_DIR/run_head_then_sft_round1.sh"
bash "$OPUS_DIR/run_head_then_sft_round1.sh"
echo ""
echo "[MASTER] DeepSeek Opus R1 completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "============================================="
echo "[MASTER] ALL DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# Show final summary
echo ""
echo "=== FINAL SUMMARY ==="
DOUBAO_RESULT=$(find "$DOUBAO_DIR/runs" -name "headsft_results_latest.json" -newer "$0" -print | sort | tail -1)
OPUS_RESULT=$(find "$OPUS_DIR/runs" -name "opus_results_latest.json" -print | sort | tail -1)

if [[ -n "$DOUBAO_RESULT" ]]; then
  echo "[Doubao R9]"
  cat "$DOUBAO_RESULT"
fi
echo ""
if [[ -n "$OPUS_RESULT" ]]; then
  echo "[DeepSeek Opus R1]"
  cat "$OPUS_RESULT"
fi
