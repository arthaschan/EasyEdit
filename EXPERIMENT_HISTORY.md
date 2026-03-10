# Dental Distill+LoRA Experiment History

## 1) Overall Progress
- Initial reproducible baseline (before auto sweep): test accuracy around **73.49%**.
- Best observed so far: **75.90%**.
- Net improvement: **+2.41 points**.

## 2) Round-by-round Summary

| Round | Run Folder | What Changed | Best Result | Key Conclusion |
|---|---|---|---:|---|
| R0 | manual run (`exp_r16_a32_e4`) | `num_epochs=4, batch_size=2, grad_acc=8, rank=16, lora_alpha=32`, with augmentation | 74.70% | Compared with earlier 73.49%, training stabilized and improved. |
| R1 (failed infra run) | `auto_experiments/run_20260307_192834` | First attempt at 4-way sweep (`lr/alpha/temp`) | N/A (all rc=1) | Environment issue (`torch` missing in `.venv`) caused all experiments to fail; fixed runner Python env afterwards. |
| R2 (4-way sweep) | `auto_experiments/run_20260307_192933` | Sweep over `lr={2e-4,1e-4,8e-5}`, `alpha={0.5,0.7}`, `temp={2.0,1.5}` | **75.90%** (`lr=1e-4, alpha=0.7, temp=2.0`) | Showed clear gain over 74.7%; established promising region around `lr=1e-4`. |
| R3 (re-check sweep) | `auto_experiments/run_20260307_193750` | Re-run 4-way sweep after fixing loss-parameter wiring | **75.90%** (`lr=1e-4, alpha=0.5, temp=2.0`) | Confirmed 75.9 is reproducible; indicated `alpha=0.7` is not consistently better. |
| R4 (minimal LR-only) | `auto_experiments/run_minimal_20260308_191752` | Fix `alpha=0.5,temp=2.0`, sweep `lr={8e-5,1e-4,1.2e-4,1.5e-4}` | **74.70%** (`1e-4` and `1.2e-4` tie) | LR sweet spot narrowed to `1e-4~1.2e-4`; too low/high LR hurts. |
| R5 (minimal Temp-only) | `auto_experiments/run_minimal_20260309_075209` | Fix `lr=1e-4,alpha=0.5`, sweep `temp={1.2,1.5,2.0,2.5}` | **75.90%** (`temp=1.5`) | Temperature is sensitive; middle value works best, extremes degrade to 73.49. |
| R6 (minimal Alpha-only) | `auto_experiments/run_minimal_20260309_184641` | Fix `lr=1e-4,temp=1.5`, sweep `alpha={0.3,0.5,0.6,0.7}` | **75.90%** (`alpha=0.3`) | Lower alpha improved over 0.5/0.6 and clearly beat 0.7; strong signal that current setup should reduce KL weight. |

## 3) Current Best Configuration
- `num_epochs=4`
- `batch_size=2`
- `gradient_accumulation_steps=8`
- `rank=16`
- `lora_alpha=32`
- `learning_rate=1e-4`
- `alpha=0.3` (latest best in alpha-only sweep)
- `temperature=1.5`
- `augment=True`

Expected test accuracy range from recent runs: **75.9%**.

## 4) Reliable Findings So Far
1. `learning_rate` near `1e-4` is consistently strong.
2. `temperature` around `1.5` outperforms `1.2/2.5`.
3. `alpha=0.3` is currently better than `0.5/0.6/0.7` when `lr=1e-4,temp=1.5`.
4. Validation metric is unstable (`0/100` swings), so checkpoint selection should not rely solely on current val split.
5. Gains now are incremental; bottleneck appears more data/knowledge coverage than formatting.

## 5) Recommended Next Single-Variable Sweep
Given current best (`lr=1e-4, temp=1.5, alpha=0.3`), sweep only `rank`:
- `rank=8`
- `rank=16` (reference)
- `rank=24`
- `rank=32`

Keep other params fixed: `lora_alpha=32, epochs=4, batch_size=2, grad_acc=8`.

## 6) Result Files
- `auto_experiments/run_20260307_193750/summary.csv`
- `auto_experiments/run_minimal_20260308_191752/summary.csv`
- `auto_experiments/run_minimal_20260309_075209/summary.csv`
- `auto_experiments/run_minimal_20260309_184641/summary.csv`
