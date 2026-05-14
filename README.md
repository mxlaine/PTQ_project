# Keyword Spotting on FPGA-Constrained Hardware

This work is part of an IC Design project. The model has to be implemented in
hardware, so a number of constraints have to be accounted for and the rest of the
effort is spent maximising accuracy within them:


- Architecture: 2 stacked unidirectional GRU layers + linear head
- Max hidden size: 64 (≈ 47k parameters total)
- Output classes: 12 (10 keywords + unknown + silence)
- Dataset: Google Speech Commands v2

## Model

`KeywordGRU` ([keyword_spotting/src/model.py](keyword_spotting/src/model.py)):

- 16- or 48-band log-mel spectrogram, optional Δ and ΔΔ features,
  per-utterance mean/std normalisation.
- 2 unidirectional GRU layers + linear classifier head.
- SpecAugment (frequency + time masking) applied to
  the mels and deltas/delta-deltas before normalisation.

### Custom GRU (`NewGRU`)

[keyword_spotting/src/new_gru.py](keyword_spotting/src/new_gru.py) reimplements
a GRU to have better visibility of weights for post-training quantization:

- Compiles the recurrent loop step with `@torch.jit.script` so it runs in C++ on GPU.
- 1 to 1 match of nn.GRU in accuracy.
- Much slower than cuDNN nn.GRU

## Training pipeline

`src/main.py` is main entry point of training, testing and validation. Various flags allow different 
parameters to be tested:

- Data augmentation with random time shift, background-noise mixing at
  SNRs in 0–15 dB (drawn from the Speech Commands `_background_noise_` clips),
  silence synthesis (10% of each batch), SpecAugment.
- Optional BC-ResNet style under-sampler that
  draws an equal number of examples from each class (unknown class is much larger than other classes in
  original dataset due to the task being a subset of the full 35 class task).
- Optimisation with AdamW, gradient clipping, label smoothing, cosine LR decay or
  cosine warm restarts with optional linear warmup and per-cycle ceiling decay.
- Seed controls model for reproducibility.

Run locally:

```bash
source .venv/bin/activate
python keyword_spotting/src/main.py --help
```

Example training run `--use-new-gru --spec-augment
--balanced-sampler --label-smoothing 0.05 --lr-scheduler cosine --epochs 325`.

All flags available with ´python3 src/main.py --help´

## Experiment infrastructure

Various SLURM array jobs under [keyword_spotting/slurm/](keyword_spotting/slurm/),
Examples in the repo cover:

- `run_keyword_gru.sbatch` — feature config x SpecAugment mask sizes
- `run_keyword_gru_size_sweep.sbatch` — hidden size (16/32/48/64)
- `run_lr_wd_sweep.sbatch` — learning rate x weight decay
- `run_dropout_sweep.sbatch` — dropout
- `run_specaugment_sweep.sbatch` — finer SpecAugment grid
- `run_augmentation_sweep.sbatch` — toggle augmentation components

Each task writes its training log into a directory under `slurm/<jobid>/`,
and the best checkpoint into `models/<jobid>/`. Plots of train/val/test curves
are created under `plots/<jobid>/`.

Every sweep includes the four required hidden
sizes (16, 32, 48, 64).

## Analysis tooling

[keyword_spotting/scripts/](keyword_spotting/scripts/) contains small CLIs for
post-hoc analysis:

- `plot_from_log.py` regenerate training curves from a single log file.
- `plot_sweep.py` / `plot_group.py` overlay curves across a sweep or across
  re-seeded "best-of" runs.
- `summarize_results.py` parse logs across one or more job directories into a
  table of best/final accuracy, runtime, and configuration. 

## Tech stack

PyTorch, torchaudio, TorchScript, NumPy, Matplotlib, SLURM.
