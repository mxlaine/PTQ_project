# Keyword Spotting on FPGA-Constrained Hardware

This work is part of an IC Design project. The model has to be implemented in
hardware, so a number of constraints have to be accounted for and the rest of the
effort is spent maximising accuracy within them:

| Architecture | 2 stacked unidirectional GRU layers + linear head |
| Max hidden size | 64 (≈ 47k parameters total) |
| Output classes | 12 (10 keywords + `unknown` + `silence`) |
| Dataset | Google Speech Commands v2 |

## Model

`KeywordGRU` ([keyword_spotting/src/model.py](keyword_spotting/src/model.py)) is
deliberately small and shaped to match the planned FPGA datapath:

- **Frontend:** 16- or 48-band log-mel spectrogram, optional Δ and ΔΔ features,
  per-utterance mean/std normalisation.
- **Sequence model:** 2 stacked unidirectional GRU layers (hidden size swept over
  16/32/48/64).
- **Head:** Linear classifier consuming the final hidden state of the last GRU
  layer. Attention/average pooling was explicitly ruled out — it costs both
  parameters and FPGA routing complexity.
- **Optional during training:** SpecAugment (frequency + time masking) applied to
  the mel/Δ stack before normalisation.

### Custom GRU (`NewGRU`)

[keyword_spotting/src/new_gru.py](keyword_spotting/src/new_gru.py) reimplements
the GRU recurrence as an explicit Python/TorchScript loop:

- Hoists the input-to-hidden projection out of the time loop into a single
  batched `F.linear`, leaving only the small hidden-to-hidden recurrence inside
  the per-timestep loop.
- Compiles the inner step with `@torch.jit.script` so it runs in C++ on GPU.
- Mirrors the gate ordering and bias decomposition of `torch.nn.GRU`; a
  conversion helper (`from_torch_gru`) copies trained weights across, and
  `test_new_gru.py` enforces numerical parity with the cuDNN reference.

The point isn't speed — it's having a reference implementation we control,
so RTL behaviour can be cross-checked against it bit-for-bit later in the
project.

## Training pipeline

`src/main.py` is the single training entrypoint and supports the full set of
knobs explored during the project:

- **Augmentation:** random time shift, gain jitter, background-noise mixing at
  SNRs in 0–15 dB (drawn from the Speech Commands `_background_noise_` clips),
  silence synthesis (10% of each batch), SpecAugment.
- **Class balancing:** optional BC-ResNet style per-epoch under-sampler that
  draws an equal number of examples from each class.
- **Optimisation:** AdamW, gradient clipping, label smoothing, cosine LR decay or
  cosine warm restarts with optional linear warmup and per-cycle ceiling decay.
- **EMA:** optional exponential moving average of model weights, with a
  configurable start epoch.
- **Reproducibility:** seed controls model init, sampler shuffle, and dataloader
  worker streams.

Run locally:

```bash
source .venv/bin/activate
python keyword_spotting/src/main.py --help
```

A representative training run uses `--use-new-gru --spec-augment
--balanced-sampler --label-smoothing 0.05 --lr-scheduler cosine --epochs 325`.

## Experiment infrastructure

Every experiment is a SLURM array job under [keyword_spotting/slurm/](keyword_spotting/slurm/),
parameterised so a single submission sweeps a Cartesian product of settings.
Examples in the repo cover:

- `run_keyword_gru.sbatch` — feature config × SpecAugment mask sizes
- `run_keyword_gru_size_sweep.sbatch` — hidden size (16/32/48/64)
- `run_lr_wd_sweep.sbatch` — learning rate × weight decay
- `run_dropout_sweep.sbatch` — dropout
- `run_specaugment_sweep.sbatch` — finer SpecAugment grid
- `run_augmentation_sweep.sbatch` — toggle augmentation components
- `run_best5_*.sbatch` — best-of configurations reseeded for variance analysis
  (EMA, balanced sampler, warm restarts, NewGRU)

Each task writes its training log into a per-job directory under `slurm/<jobid>/`,
and the best checkpoint into `models/<jobid>/`. Plots of train/val/test curves
are emitted under `plots/<jobid>/` so a finished sweep can be inspected by
opening one directory.

Per project convention, every sweep must include the four required hidden
sizes (16, 32, 48, 64) so that any reported result can be read against the
parameter-budget frontier.

## Analysis tooling

[keyword_spotting/scripts/](keyword_spotting/scripts/) contains small CLIs for
post-hoc analysis:

- `plot_from_log.py` — regenerate training curves from a single log file.
- `plot_sweep.py` / `plot_group.py` — overlay curves across a sweep or across
  re-seeded "best-of" runs.
- `summarize_results.py` — parse logs across one or more job directories into a
  table of best/final accuracy, runtime, and configuration.

## Tech stack

PyTorch, torchaudio, TorchScript, NumPy, Matplotlib, SLURM. Target deployment
is a custom FPGA — the Python side of this repo is the design and validation
sandbox for that target.

## Side directory: `cnn_testing/`

A small MNIST CNN notebook kept alongside the main project for unrelated
experimentation; it does not share code with `keyword_spotting/`.
