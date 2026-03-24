# AI Context Handover (Latest)

Last updated: 2026-03-24

## 1. Current Project Focus
- This workspace is now used as a dental QA training/evaluation project.
- Most EasyEdit/Steer framework code has been archived out of root to keep the project lightweight and task-focused.

## 2. Major AI Work Completed (Chronological)
1. Evaluation consistency fix:
- Resolved score mismatch between training logs and retest scripts.
- Unified evaluation protocol (prompt format, options handling, deterministic generation).
- Reproduced the target run score at 80.72%.

2. Teacher baseline verification:
- Updated testing to directly evaluate 32B teacher (`Qwen2.5-32B-Instruct`).
- Verified teacher direct test accuracy at 83.13% under the aligned protocol.

3. Distillation experiments and analysis:
- Completed multiple black-box distillation comparisons (A/C/E style configs).
- Added no-overlap train/val splits and multi-seed follow-up.
- Extended analysis with hard-boost pilot; no stable gain vs baseline in current setup.

4. Error analysis for 80.72% run:
- Produced train/test category-wise accuracy tables.
- Exported detailed test failure list.
- Built error attribution by knowledge tag and confusion-option pattern.

5. Repository slimming:
- First slimming: moved EasyEdit/Steer related code to backup folder.
- Second slimming: moved historical logs, temp artifacts, and many old experiment outputs to an external archive path.

## 3. Key Results to Remember
- 32B teacher direct test: 83.13%
- Target student run (32B -> 7B, seed2 best): 80.72%
- Train/Test for that run: 98.66% / 80.72%
- Main weak test areas observed:
  - Threshold/range definition questions
  - Neighboring anatomy/position concepts
  - Typical imaging sign confusion
  - Negation-style wording robustness

## 4. Active vs Archived Artifacts

### 4.1 Kept in active workspace (`/home/student/arthas/EasyEdit3`)
- Core scripts:
  - `train_dental_lora32.py`
  - `autoTestQwen32.py`
  - `deploy_dental_robot.py`
  - `deploy_dental_robot7.py`
  - `train_targeted_sft.py`
  - `train_cot.py`
- Core model/data directories (kept):
  - `Qwen2.5-*-Instruct/`
  - `data/`
  - `dental_qwen2.5_1.5b_lora/`
  - `dental_qwen2.5_7b_choice_lora/`
  - `dental_qwen2.5_7b_choice_lora_standard/`
  - `dental_qwen2.5_7b_choice_lora_distill_from32_seed2/`

### 4.2 Archived out of active workspace
- Archive root:
  - `/home/student/arthas/EasyEdit3_archives/secondary_slim_20260324_213749`
- Archived contents include:
  - historical logs (`*.log`, `*.nohup.log`)
  - temporary/perf folders (`tmp_*`, `logs`, `history0305`, etc.)
  - many old distill experiment output directories
  - previous cleanup backup bundle

## 5. Fast Resume Checklist (When You Return)
1. Open `EXPERIMENT_HISTORY.md` and this file first.
2. Confirm whether required logs/checkpoints are in workspace or in archive path.
3. Use `conda run -n easyedit` for runtime checks and training scripts.
4. Before starting a new round, define:
- fixed eval protocol
- exact seed list
- output directory naming rule
5. After each run, append only high-value deltas (not raw terminal dump).

## 6. Cross-Machine Continuity (Important)
To avoid restarting from scratch on another computer:
1. Commit and push this repository (includes this handover file and history files).
2. Also migrate archive data if needed:
- source: `/home/student/arthas/EasyEdit3_archives/secondary_slim_20260324_213749`
- copy with rsync/scp to the new machine.
3. On the new machine, keep the same relative relationship if possible:
- active repo: `.../EasyEdit3`
- archive root: `.../EasyEdit3_archives/...`
4. If archive path changes, update path references in this file and `EXPERIMENT_HISTORY.md`.

## 7. Suggested Logging Convention (Going Forward)
- Keep one summary file updated: `AI_CONTEXT_LATEST.md` (this file).
- Keep experiment result history in `EXPERIMENT_HISTORY.md`.
- For each major run batch, add:
  - objective
  - exact command
  - metric summary
  - decision taken
  - next action

## 8. Environment Notes
- Preferred runtime env for scripts: conda env `easyedit`.
- The local `.venv` may not include heavy ML deps (e.g., `torch`), so use it carefully for non-training tasks.
