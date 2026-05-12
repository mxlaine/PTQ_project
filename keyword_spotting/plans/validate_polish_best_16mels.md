# Plan: Validate & polish the 16_mels_delta_delta best result

## Context

Current best on `16_mels_delta_delta`: **97.28% test acc** (val ~97.45%) at hidden=64, lr=2e-3, wd=0, dropout=0.1, freq_mask=5, time_mask=6, warmup=10, cosine, `--use-new-gru`, no EMA, 325 epochs. Reached in [slurm/17719851](../slurm/17719851/) tasks 35 and 39 — but only **seed=0** has been run.

Two gaps before declaring this the final number:

1. **Advisor mandate not yet satisfied** — the advisor requires 3 seeds × 4 hidden sizes with mean ± std. The 64-task lr/wd sweep was single-seed. We literally don't know if 97.28% is the typical result for this config or a favorable seed.
2. **EMA never tested at the new optimum** — the existing EMA sweep ([slurm/run_best5_ema_sweep.sbatch](../slurm/run_best5_ema_sweep.sbatch)) used the *previous* best (lr=1e-3, wd=1e-4). Label smoothing has *never* been swept (default 0.05 always).

User chose **low-effort polish**: do these two things, then accept the result. Realistic upside is +0.1-0.3%; realistic downside is discovering the true mean is ~97.10-97.20% and 97.28% was a lucky draw — either way we land on a defensible number.

## Approach

### Step 1 — Canonical 3-seed × 4-hidden validation (mandatory)

Run the advisor-mandated grid at the new best config:
- Hidden sizes: {16, 32, 48, 64}
- Seeds: {0, 1, 2}
- Fixed: lr=2e-3, wd=0, dropout=0.1, freq=5, time=6, warmup=10, cosine, `--use-new-gru`, label_smoothing=0.05 (default), 325 epochs
- **12 runs total**, ~7-9h each on V100 → fits comfortably in 16h time limit per array task

Create `slurm/run_best_3seed_validation.sbatch` based on the structure of [slurm/run_lr_wd_sweep.sbatch](../slurm/run_lr_wd_sweep.sbatch). Map array index → (hidden_size, seed) with `--array=0-11`.

**Outcome**: gives `97.XX% ± 0.YY%` for hidden=64 and the full hidden-size curve the advisor wants for plots.

### Step 2 — Polish sweep at hidden=64 (single seed exploratory)

2×3 grid at hidden=64, seed=0, all other hyperparameters fixed at best config:
- EMA: {off, on (decay=0.999, start=epoch 162)}
- Label smoothing: {0.0, 0.05, 0.10}
- **6 runs** total, ~7-9h each → `--array=0-5`

Create `slurm/run_polish_ema_ls.sbatch` (template: [slurm/run_best5_ema_sweep.sbatch](../slurm/run_best5_ema_sweep.sbatch), but with the **new** baseline `LR=2e-3 WEIGHT_DECAY=0`).

### Step 3 — Decision gate

After Steps 1 & 2 complete, use [scripts/summarize_results.py](../scripts/summarize_results.py) to aggregate, then:
- If **any polish cell beats the seed-0 baseline by ≥ 0.15%**, run that single cell at seeds {1, 2} (2 more runs) to confirm with 3-seed mean. If confirmed, that becomes the new best.
- Otherwise: accept the 3-seed mean from Step 1 as the final reported number and stop tuning. Move on.

### What we are deliberately NOT doing (and why)

- **Mixup / CutMix / SWA / TTA / SNR sweep / LR re-tuning**: outside "low-effort polish" budget; expected gains < EMA/LS at higher cost.
- **More feature configs / architectures / attention**: ruled out by [CLAUDE.md](../CLAUDE.md) and advisor constraints.
- **Wider augmentation grid**: freq=5/time=6 was already the optimum from a prior sweep; revisiting gives diminishing returns.

## Critical files

- New: `slurm/run_best_3seed_validation.sbatch` (12-task array, validates current best)
- New: `slurm/run_polish_ema_ls.sbatch` (6-task array, EMA × label_smoothing at hidden=64)
- Reuse as template: [slurm/run_lr_wd_sweep.sbatch](../slurm/run_lr_wd_sweep.sbatch) (cleanest log-organization & arg-mapping pattern)
- Reuse as template: [slurm/run_best5_ema_sweep.sbatch](../slurm/run_best5_ema_sweep.sbatch) (EMA flag wiring; **change baseline to lr=2e-3, wd=0**)
- No source code changes needed — all knobs already exposed in [src/main.py](../src/main.py) (`--ema`, `--label-smoothing`, `--seed`).

## Verification

1. After submitting Step 1: check job output dir `slurm/<jobid>/` exists, `.out` files start logging, no immediate Python errors in any task's `.err`.
2. After Step 1 finishes: run `python scripts/summarize_results.py <jobid>` (or follow the script's interface) and confirm 12 test-acc lines appear, grouped by hidden_size, with mean ± std.
3. After Step 2 finishes: same aggregation, then compare each cell to the 97.28% baseline.
4. End-to-end sanity (optional, if either sbatch is edited): pre-flight one task locally on CPU using `--use-new-gru`, reduced epochs (e.g. `EPOCHS=2`) — confirm it runs end-to-end and writes outputs to the expected paths.

## Stopping criterion (explicit)

If after Steps 1-3 the best 3-seed mean test accuracy is ≤ 97.28% + 0.15%, **stop**. Report the 3-seed mean ± std for hidden=64 (plus the full hidden-size curve) as the final result for `16_mels_delta_delta`, and pivot to the next project deliverable.
