# Codebase Cleanup Plan

## Context

The keyword-spotting codebase has accumulated configurable options that don't earn their keep. AMP wasn't yielding measurable gains, `num_workers` and `num_layers` are knobs nobody tunes, `speed_perturb` didn't help, and the `balanced_sampler` is implemented as a **WeightedRandomSampler with replacement** (i.e. an over-sampler) — but the BC-ResNet recipe the project benchmarks against uses **under-sampling**. So the option as it stands isn't actually testing the strategy intended.

This cleanup removes the dead options, replaces `balanced_sampler` with a proper BC-ResNet–style under-sampler, and tightens up names/structure in the training loop. The fixed 2× GRU architecture constraint (CLAUDE.md) is reinforced by removing the `--num-layers` CLI flag.

The intended outcome: a leaner, less surprising codebase where the remaining knobs reflect actual experiments run.

---

## Decisions

| Item | Decision |
|---|---|
| `balanced_sampler` | **Replace with BC-ResNet–style under-sampler.** Keep the `--balanced-sampler` CLI flag, but change the implementation. |
| `num_workers` | **Hardcode to 4** inside `build_dataloaders`. Remove CLI flag + SLURM plumbing. |
| Timing instrumentation in `train_one_epoch` | **Keep but rename clearly** (`t_data` → `time_data_loading`, etc.). |
| `num_layers` | **Remove CLI flag**, but keep `num_layers` as a constructor argument on `KeywordGRU` / `NewGRU` (defaulting to 2). Don't rip it out of the model code. |
| `AMP`, `speed_perturb` | **Remove entirely** (no replacement). |

---

## Scope of changes

### 1. Remove AMP

**Files:** [src/main.py](../src/main.py)

- Remove `--amp` CLI arg ([src/main.py:168](../src/main.py#L168)).
- Remove `use_amp` and `scaler` parameters from `train_one_epoch` ([src/main.py:28](../src/main.py#L28)) and `evaluate` ([src/main.py:102](../src/main.py#L102)) signatures.
- Remove the `torch.amp.autocast(...)` contexts ([src/main.py:55-57](../src/main.py#L55-L57), [src/main.py:117-119](../src/main.py#L117-L119)).
- Remove the `if use_amp:` / `else:` branch around backward+step ([src/main.py:64-73](../src/main.py#L64-L73)) — keep the standard `loss.backward(); clip_grad_norm_(...); optimizer.step()` path.
- Remove `scaler = torch.amp.GradScaler(...)` ([src/main.py:206](../src/main.py#L206)).
- Remove the AMP print line and the `use_amp=` kwargs at call sites ([src/main.py:251, 290, 293, 321, 326-327](../src/main.py#L251)).

### 2. Remove `num_workers` CLI flag (hardcode to 4)

**Files:** [src/main.py](../src/main.py), [src/utils.py](../src/utils.py), [src/test_new_gru.py](../src/test_new_gru.py), all SLURM scripts.

- Remove `--num-workers` CLI arg ([src/main.py:143](../src/main.py#L143)).
- Remove `num_workers=args.num_workers` from `build_dataloaders` call ([src/main.py:181](../src/main.py#L181)).
- Remove the print line ([src/main.py:252](../src/main.py#L252)).
- In [src/utils.py:237](../src/utils.py#L237): drop `num_workers` parameter; hardcode `num_workers=4` and `persistent_workers=True` in all three DataLoaders ([src/utils.py:269, 271, 279, 281, 288, 290](../src/utils.py#L269)).
- In [src/test_new_gru.py:70](../src/test_new_gru.py#L70): drop `num_workers=2` kwarg.
- SLURM scripts: remove `NUM_WORKERS` variable definitions and `--num-workers` flags from all `slurm/*.sbatch` files (5 tracked + 2 untracked, see inventory below).

### 3. Replace `balanced_sampler` with BC-ResNet-style under-sampler

**Files:** [src/utils.py](../src/utils.py), [src/main.py](../src/main.py)

**Implementation approach** (BC-ResNet style):

- Replace the existing `make_sample_weights()` helper ([src/utils.py:178-192](../src/utils.py#L178-L192)) with a new `BalancedUnderSampler(torch.utils.data.Sampler)` class that:
  1. At construction, groups `train_set` indices by label.
  2. On each `__iter__` (i.e. each epoch), determines the target per-class count = size of the **smallest non-zero class** in the dataset.
  3. For each class, randomly samples `target_count` indices **without replacement** (target = min, so no class needs replacement).
  4. Concatenates and shuffles the resulting indices, yields them.
  5. `__len__` returns `target_count * num_classes`.
- In `build_dataloaders` ([src/utils.py:256-264](../src/utils.py#L256-L264)), replace the `WeightedRandomSampler` branch with `BalancedUnderSampler(train_set)`. Keep the existing `if balanced_sampler: sampler=...; shuffle=False else: sampler=None; shuffle=True` structure.
- **Class weights logic** ([src/main.py:228-230](../src/main.py#L228-L230)): keep the existing inverse-coupling (when `balanced_sampler` is on, skip the `unknown=1.20`/`silence=1.10` boost) — under-sampling already balances classes, so the weight tweaks would double-count. Add a brief comment explaining the rationale.
- The `--balanced-sampler` CLI flag stays unchanged ([src/main.py:167](../src/main.py#L167)).

### 4. Remove `speed_perturb` entirely

**Files:** [src/main.py](../src/main.py), [src/utils.py](../src/utils.py), `slurm/run_augmentation_sweep.sbatch`

- Remove `--speed-perturb` CLI arg ([src/main.py:158](../src/main.py#L158)) and call-site kwarg ([src/main.py:183](../src/main.py#L183)).
- Remove print line ([src/main.py:241](../src/main.py#L241)) and plot-metadata branch ([src/main.py:339-340](../src/main.py#L339-L340)).
- In [src/utils.py](../src/utils.py): delete `_SPEED_RATES`, `_get_speed_resampler()`, `_random_speed_perturb()` ([src/utils.py:105-124](../src/utils.py#L105-L124)). Drop `speed_perturb` parameter from `collate_fn` ([src/utils.py:127](../src/utils.py#L127)) and `make_train_collate` ([src/utils.py:172](../src/utils.py#L172)). Remove the conditional invocation ([src/utils.py:139-140](../src/utils.py#L139-L140)). Remove from `build_dataloaders` signature and pass-through ([src/utils.py:239, 274](../src/utils.py#L239)).
- Remove `--speed-perturb` plumbing from [slurm/run_augmentation_sweep.sbatch](../slurm/run_augmentation_sweep.sbatch).

### 5. Remove `--num-layers` CLI flag (keep model constructor arg)

**Files:** [src/main.py](../src/main.py), all SLURM scripts.

- Remove `--num-layers` CLI arg ([src/main.py:155](../src/main.py#L155)).
- At [src/main.py:193](../src/main.py#L193): replace `num_layers=args.num_layers` with no kwarg (rely on the default of 2 in `KeywordGRU.__init__`).
- Remove print line ([src/main.py:238](../src/main.py#L238)) and the `f"layers={args.num_layers}"` plot label ([src/main.py:346](../src/main.py#L346)).
- SLURM scripts: remove `NUM_LAYERS` variable + `--num-layers` flag from all 5 tracked `.sbatch` files (and any untracked ones that reference it).
- **Do NOT touch** [src/model.py:14, 52](../src/model.py#L14) or [src/new_gru.py:55, 64, 69, 90, 97, 110, 136](../src/new_gru.py#L55) — leave `num_layers` as a constructor parameter defaulting to 2.

### 6. Rename timing variables in `train_one_epoch`

**File:** [src/main.py](../src/main.py) (lines 28-99)

Mechanical rename only — no logic changes:

| Old | New |
|---|---|
| `t_data` | `time_data_loading` |
| `t_fwd` | `time_forward` |
| `t_bwd` | `time_backward` |
| `t_loop_start` | `epoch_start_time` |
| `t_iter_start` | `batch_start_time` |
| `t_fwd_start` | `forward_start_time` |
| `t_bwd_start` | `backward_start_time` |

Update the matching variables in the return tuple and at call sites in `main()` ([src/main.py:282-327](../src/main.py#L282-L327)).

### 7. Light touch-ups (in scope, low risk)

- **`collate_fn`** ([src/utils.py:127-169](../src/utils.py#L127-L169)): after removing `speed_perturb`, the function shrinks naturally. Extract the silence-synthesis block ([src/utils.py:159-164](../src/utils.py#L159-L164)) into a small helper `_synthesize_silence_batch(num_silence, target_length)` for clarity. The 10% silence ratio remains hardcoded (existing behavior).
- **`build_dataloaders`** ([src/utils.py:234-295](../src/utils.py#L234-L295)): after removing `num_workers` and `speed_perturb` params, the signature shrinks. The 3× DataLoader-creation duplication is acceptable for now — leave it (premature abstraction).
- **`parse_args`**: after the removals, the parser is small enough that argument groups aren't worth adding.

### 8. Out of scope (explicitly NOT doing)

- Refactoring `train_one_epoch` to a profiler context manager.
- Refactoring `main()` into smaller functions.
- Renaming GRU gate variables (`gi`, `gh`, `i_r`, `i_z`, `i_n`, …) in [src/new_gru.py](../src/new_gru.py) — these are standard PyTorch GRU notation.
- Touching [src/model.py](../src/model.py) or [src/new_gru.py](../src/new_gru.py) beyond what's required for the `--num-layers` CLI removal (i.e. nothing).

---

## Files modified

**Source:**
- [src/main.py](../src/main.py) — CLI args, training loop, eval, main
- [src/utils.py](../src/utils.py) — DataLoader builder, collate_fn, sampler
- [src/test_new_gru.py](../src/test_new_gru.py) — drop `num_workers=2` arg

**SLURM (remove `NUM_WORKERS`, `NUM_LAYERS`, `--speed-perturb` as applicable):**
- [slurm/run_augmentation_sweep.sbatch](../slurm/run_augmentation_sweep.sbatch)
- [slurm/run_best5_balanced.sbatch](../slurm/run_best5_balanced.sbatch)
- [slurm/run_best5_new_gru.sbatch](../slurm/run_best5_new_gru.sbatch)
- [slurm/run_best5_warmrestart.sbatch](../slurm/run_best5_warmrestart.sbatch)
- [slurm/run_keyword_gru_v2.sbatch](../slurm/run_keyword_gru_v2.sbatch)
- Untracked: `slurm/run_best5_ema_sweep.sbatch`, `slurm/run_dropout_sweep.sbatch` — apply same removals if they reference the deleted flags.

---

## Verification

1. **Static check:** `python -c "from src.main import main"` and `python -c "import src.utils, src.model, src.new_gru"` import without errors.
2. **Unit test:** run [src/test_new_gru.py](../src/test_new_gru.py) — verifies the GRU/NewGRU paths still match after the call-site cleanup.
3. **Tiny smoke training run:** `python src/main.py --epochs 1 --batch-size 64` — confirms the basic path works (no AMP, no speed_perturb, no `--num-layers` arg) and that timing prints render with the new variable names.
4. **Under-sampler smoke test:** `python src/main.py --epochs 1 --balanced-sampler` — confirms the new `BalancedUnderSampler` produces a class-balanced epoch (print per-class counts in the first batch as a sanity check, or temporarily log `len(sampler)` and class histogram).
5. **SLURM dry-run:** `bash -n slurm/run_keyword_gru_v2.sbatch` etc. on the modified SLURM scripts to catch shell syntax issues from the variable removals.
