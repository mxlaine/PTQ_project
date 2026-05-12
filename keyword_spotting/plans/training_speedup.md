# Speedup plan for keyword-spotting training

## Context

Each training run does ~85k samples × ~1342 batches × 400 epochs of redundant work because mel-spectrogram, dB conversion, and delta features run **on GPU inside `model.forward`** ([model.py:74-83](src/model.py#L74-L83)). Workers sit idle while the GPU serially does feature extraction *then* the GRU. A single dropout/SpecAugment sweep therefore takes far longer than necessary, slowing iteration on the broader experiment campaign (`run_dropout_sweep.sbatch` = 25 tasks, `run_best5_*.sbatch` = 5 each).

ChatGPT proposed four levers. After reading the code, only two are clean wins; the others are either invalid or out-of-scope:

| Lever | Verdict | Reason |
|---|---|---|
| 1. Precompute mel features **to disk** | ❌ Invalid as stated | [utils.py:101-116](src/utils.py#L101-L116) applies random time-shift / gain / noise to the **raw waveform** every epoch, so mels are not constant across epochs. |
| 1'. Move mel/dB/delta into **CPU dataloader workers** | ✅ Adopt | Preserves all augmentation, parallelizes features across workers, removes them from the GPU critical path. |
| 2. `torch.compile` on NewGRU | ⏭ Skip | User has explicitly excluded it for now. |
| 3. AMP (`autocast` + `GradScaler`) | ✅ Adopt (opt-in) | Free fp16 tensor-core speedup on V100 for `F.linear` in NewGRU and `nn.GRU`; no current AMP plumbing. |
| 4. Larger batch size | ⏭ Out-of-scope here | The dropout sweep is mid-campaign — changing batch size shifts SGD trajectory and effective LR schedule, hurting comparability with prior runs. Easy to revisit afterwards via the existing `--batch-size-train` arg. |

Goal: make a future dropout/best-5 sweep meaningfully faster (rough expectation 2–3× wall-clock) **without changing training semantics** for runs that don't opt into AMP.

## Approach

### Change 1 — Move mel/dB/delta to CPU dataloader workers

Compute features inside `collate_fn` after waveform-level augmentation (and after the synthetic silence samples are appended), so each of the 4 workers prepares a feature tensor in parallel while the GPU is busy on the previous batch. The model's forward path then starts at the SpecAugment / normalization step.

**Files:**

- [src/utils.py](src/utils.py)
  - Add a small `FeatureExtractor` (plain object, not `nn.Module` — workers don't need autograd) holding `MelSpectrogram`, `AmplitudeToDB`, and an optional `ComputeDeltas`, configured from `n_mels` / `use_delta` / `use_delta_delta`. Reuse [`build_mel_spectrogram`](src/model.py#L9-L18).
  - Extend `collate_fn` to take an optional `feature_extractor`; when provided, after the existing `torch.stack(waveforms)` it runs `mel → db → optional deltas/delta-deltas → cat on dim=2` and returns `(features, labels)` instead of `(waveforms, labels)`.
  - Plumb the extractor through `make_train_collate`, `train_collate`, `eval_collate`, and `build_dataloaders` (new kwargs: `n_mels`, `use_delta`, `use_delta_delta`, default `None` = legacy waveform-mode, preserving back-compat for any other caller).

- [src/model.py](src/model.py)
  - Add `precomputed_features: bool = False` to `KeywordGRU.__init__`.
  - **Keep** `self.mel / self.db / self.compute_deltas` constructed unconditionally so existing checkpoints (whose `state_dict` contains the mel filterbank buffer) still load cleanly.
  - In `forward`, if `self.precomputed_features` is True, treat `x` as already-stacked features `(B, F, T)` (or `(B, 1, F, T)` — handle both via `if x.dim() == 4: x = x.squeeze(1)`) and jump straight to the SpecAugment / normalization block at [model.py:88-94](src/model.py#L88-L94). Otherwise behave exactly as today.

- [src/main.py](src/main.py)
  - Pass the feature config into `build_dataloaders` so workers compute the matching features.
  - Construct the model with `precomputed_features=True`.
  - No CLI changes needed — this is a strict speedup with identical numerical semantics.

**Augmentation ordering (unchanged):** time-shift / gain / noise still happen per-sample on the raw waveform, then silence samples are injected, then the whole batch is stacked, then mel features are computed. SpecAugment continues to run on the spectrogram inside `model.forward` (train-only).

**Checkpoint compatibility:** preserved — model still owns the mel buffers, just bypasses them in forward.

### Change 2 — Optional AMP

Add behind a flag so existing runs are bit-identical and only new runs opt in.

**Files:**

- [src/main.py](src/main.py)
  - New CLI flag `--amp` (default off).
  - In `main()`: `scaler = torch.amp.GradScaler('cuda', enabled=args.amp)`; pass `args.amp` and `scaler` into `train_one_epoch` and `evaluate`.
  - `train_one_epoch`: wrap `output = model(data); loss = criterion(...)` in `with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):`. Replace `loss.backward(); clip; optimizer.step()` with `scaler.scale(loss).backward(); scaler.unscale_(optimizer); clip; scaler.step(optimizer); scaler.update()`.
  - `evaluate`: wrap forward in `autocast` (no scaler needed since no backward).
  - Print AMP status near the existing config printout.

**Risk notes:** SpecAugment masking and the `std`-normalization in `model.forward` should be safe under autocast (autocast picks per-op precision). NewGRU's recurrent loop runs `F.linear` / `sigmoid` / `tanh`, which autocast handles. If a sweep with `--amp` shows divergence vs. fp32, the flag stays off by default so existing experiments are unaffected.

## Critical files to touch

- [src/utils.py](src/utils.py) — `collate_fn`, `make_train_collate`, `train_collate`, `eval_collate`, `build_dataloaders`; add `FeatureExtractor`.
- [src/model.py](src/model.py) — add `precomputed_features` flag and the bypass in `forward`.
- [src/main.py](src/main.py) — wire feature config into `build_dataloaders`, set `precomputed_features=True`, add `--amp` plumbing in `train_one_epoch` / `evaluate`.

No changes needed to [src/new_gru.py](src/new_gru.py), the slurm scripts, or any analysis scripts. Existing slurm sbatch files keep working unchanged; pass `--amp` to opt in.

## Verification

1. **Unit-level smoke test** (locally, CPU is fine):
   - Run a 1-epoch training with the default config and confirm it completes and reports a sane train acc.
   - Run with `--use-new-gru` to confirm the new path also works for NewGRU.
   - Run with `--amp` on GPU only and confirm it completes.

2. **Numerical-equivalence check** (no AMP):
   - Seed Python/NumPy/Torch, run 1 epoch on `main` and 1 epoch on this branch with `--num-workers 0` (so collate is deterministic in-process).
   - Compare final train loss / acc — should match to within FP rounding (~1e-4 in loss). Any larger gap means feature ordering shifted.

3. **Wall-clock benchmark**:
   - Time 5 epochs on `main` vs. branch, both `--num-workers 4`, batch 64, no AMP. Expect ≥1.5× speedup from Change 1 alone.
   - Then time 5 epochs with `--amp` added. Expect another ~1.3–2× on top.
   - Use the existing slurm script with a reduced `--epochs 5` and inspect the `.out` log timestamps.

4. **End-to-end sanity**: kick off one task of `run_dropout_sweep.sbatch` for 20 epochs and confirm validation accuracy curve looks comparable to a recent reference run (`/scratch/work/lainem31/PTQ_project/keyword_spotting/slurm/17620206/`).
