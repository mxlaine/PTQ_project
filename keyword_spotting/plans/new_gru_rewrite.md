# Speed up NewGRU + clean up best-5 sbatch scripts

## Context

The current NewGRU run (`17663005`) is taking **~5–6 min/epoch** on a V100 (4 epochs in ~23 min, vs the nn.GRU baseline `17620206_0` which finished 400 epochs in ~2 hours → ~30 s/epoch). At 5 min/epoch × 400 epochs that's >33 hours per task — well over the 8 h SLURM limit, so the job will time out.

Root cause: [src/new_gru.py:91](src/new_gru.py#L91) processes the sequence **one timestep at a time in Python**, calling `F.linear` twice per step. nn.GRU uses cuDNN's fused parallel kernel; NewGRU does not, and `torch.compile` cannot rescue it (hence the warning in the `.err` file: *"Torchinductor does not support code generation for complex operators. Performance may be worse than eager."*).    

There are also two small issues in the sbatch scripts to clean up:
- `run_best5_new_gru.sbatch` does a redundant `module load scicomp-python-env/2025.1` *after* activating the venv → causes the version-mismatch reload warning.
- `run_best5_balanced.sbatch` has a stray `export CC=$(which gcc)` with a misleading comment ("after module load") but there is no module load. It's harmless but confusing.

The goal is to (a) make NewGRU fast enough to complete within 8 h and (b) leave the runs reproducing the all-time best config exactly the same in every other respect.

## Recommended approach

### 1. Hoist the input projection out of the timestep loop ([src/new_gru.py:69-110](src/new_gru.py#L69-L110))

Currently each call to `NewGRUCell.forward` runs `F.linear(x_t, weight_ih, bias_ih)` per timestep — a tiny `(B, in)·(in, 3H)` matmul executed `T` times serially. Replace with one batched matmul over the whole sequence, then loop only over the recurrent part (which genuinely depends on `h_{t-1}`).

Concretely, rewrite `NewGRU.forward` to:

```python
for layer, cell in enumerate(self.cells):
    h = h0[layer]
    # ONE matmul for the whole sequence, instead of T per-timestep matmuls
    gi_all = F.linear(layer_input, cell.weight_ih, cell.bias_ih)   # (B, T, 3H)
    outputs = [] if collect_outputs else None
    for t in range(seq_len):
        gi = gi_all[:, t]
        gh = F.linear(h, cell.weight_hh, cell.bias_hh)             # still per-step
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h = (1.0 - z) * n + z * h
        if outputs is not None:
            outputs.append(h)
    ...
```

This is **mathematically identical** to the current code — same params, same activation, same arithmetic — and preserves the `weight_ih_l{layer}` / `weight_hh_l{layer}` / `bias_*` parameter names so `from_torch_gru()` still works. We delete `NewGRUCell.forward`'s use inside the loop (the cell module can stay for parameter ownership) or fold the cell logic directly into `NewGRU.forward` and keep `NewGRUCell` as a parameter container.

Expected speedup: **3–8×** (turns `T≈100` tiny matmuls into 1 large one per layer). With `T=100`, B=64, in=72, this is the dominant cost per epoch.

### 2. Wrap the recurrent loop with `@torch.jit.script` (optional, additional ~1.5–2×)

After step 1, the remaining bottleneck is Python-level loop overhead over `T` steps. A scripted inner loop closes most of that gap. Implement as a free function `_recurrent_loop(gi_all, h, weight_hh, bias_hh) -> Tensor` decorated with `@torch.jit.script`. Keep it optional behind a flag if it complicates debugging.

### 3. Disable `torch.compile` for the NewGRU subtree

Currently [src/main.py:143](src/main.py#L143) wraps the whole model in `torch.compile`. That's fine for nn.GRU, but for NewGRU it (a) emits the "complex operators" warning, (b) likely produces slower code than eager, and (c) traces a graph with `T` repeated copies of the cell which is expensive to compile.

Two options — pick one:
- **Cheap**: add `self.gru = torch.compiler.disable(self.gru)` in `KeywordGRU.__init__` when `use_new_gru=True`, mirroring the existing pattern at [src/model.py:48](src/model.py#L48).
- **Cleaner**: in `main.py`, only call `torch.compile(model)` when `not args.use_new_gru`.

I'd go with the first — keeps `main.py` untouched and matches the SpecAugment exclusion already in `model.py`.

### 4. Fix `run_best5_new_gru.sbatch`

Remove [slurm/run_best5_new_gru.sbatch:64](slurm/run_best5_new_gru.sbatch#L64) (`module load scicomp-python-env/2025.1`). It's downgrading the env that was already loaded by the venv — that's the source of the *"reloaded with a version change"* warning. The dropout-sweep script that produced the best result does **not** load this module, so removing it brings this script in line with the working baseline.

After removal, the script will be identical to `run_best5_balanced.sbatch` except for `--use-new-gru` vs `--balanced-sampler` — which is the intended difference.

### 5. (Optional) Tidy `run_best5_balanced.sbatch`

The `export CC=$(which gcc)` block at [slurm/run_best5_balanced.sbatch:64-65](slurm/run_best5_balanced.sbatch#L64-L65) is dead — there's no `module load` and no compilation step that needs a C compiler. The all-time-best `run_dropout_sweep.sbatch` does not have it. Removing keeps the env minimal and parity with the baseline. Harmless to leave, but leaving it in is the only diff from the script that produced 17620206_0.

## What I'm explicitly **not** changing

- **No AMP / autocast.** It would give another 1.3–1.7× on V100 but changes numerics — risks downstream PTQ work and breaks parity with the all-time best run. Leave for a separate decision.
- **No batch size / num_workers changes.** The dropout sweep config that produced 97.60% used batch=64, num_workers=4; preserving those.
- **No hyperparameter changes** in the best-5 arrays. The `LR_WARMUPS=(10 10 10 10 10)` deviates from the original sweep (rank 2-5 used warmup=0), but that's a deliberate user choice and outside the scope of "speed up."

## Files to modify

- [src/new_gru.py](src/new_gru.py) — hoist input projection (step 1), optionally jit-script inner loop (step 2)
- [src/model.py](src/model.py) — wrap `self.gru` in `torch.compiler.disable` when `use_new_gru=True` (step 3)
- [slurm/run_best5_new_gru.sbatch](slurm/run_best5_new_gru.sbatch) — drop the redundant `module load` (step 4)
- [slurm/run_best5_balanced.sbatch](slurm/run_best5_balanced.sbatch) — drop dead `export CC` block (step 5, optional)

## Verification

1. **Numerical parity test** before/after the NewGRU rewrite:
   ```python
   import torch, torch.nn as nn
   from new_gru import NewGRU, from_torch_gru
   torch.manual_seed(0)
   gru = nn.GRU(input_size=72, hidden_size=64, num_layers=2, batch_first=True)
   ng = from_torch_gru(gru)
   x = torch.randn(4, 100, 72)
   y_ref, h_ref = gru(x)
   y_new, h_new = ng(x)
   assert torch.allclose(y_ref, y_new, atol=1e-5)
   assert torch.allclose(h_ref, h_new, atol=1e-5)
   ```
2. **Speed sanity check** on a single GPU node interactively (or `--time=00:30:00` slurm job): 1 epoch should now take **<1 min** instead of 5–6 min. If it doesn't, profile with `torch.profiler` before submitting the array.
3. **Re-submit `run_best5_new_gru.sbatch`** once parity + speed are confirmed. Cancel `17663005` first (it will time out anyway).
4. After step 4 cleanup, the `.err` file should no longer contain the *"reloaded with a version change"* line — confirms the redundant module load is gone.

## On the warnings in `keyword-gru-best5-newgru-17663005_0.err`

- **"reloaded with a version change: scicomp-python-env/2025.2 => 2025.1"** — caused by step 4's redundant `module load`. Fix it by removing the line; not a correctness issue, just noise.
- **"Torchinductor does not support code generation for complex operators. Performance may be worse than eager."** — caused by `torch.compile` failing on NewGRU's per-timestep loop. Fixed by step 3 (disable compile for NewGRU). Won't appear once disabled.
