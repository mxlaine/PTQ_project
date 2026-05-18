# Keyword Spotting on FPGA-Constrained Hardware

This work is part of an IC Design project. The model has to be implemented in
hardware, so a number of constraints have to be accounted for and the rest of the
effort is spent maximising accuracy within them:

| Architecture | 2 stacked unidirectional GRU layers + linear head |
| Max hidden size | 64 (≈ 47k parameters total) |
| Output classes | 12 (10 keywords + `unknown` + `silence`) |
| Dataset | Google Speech Commands v2 |

## Setup

```bash
git clone <repo-url>
cd PTQ_project
python -m venv .venv
source .venv/bin/activate
pip install -r keyword_spotting/requirements.txt
```

The Google Speech Commands v2 dataset is downloaded automatically by
`torchaudio` into `keyword_spotting/data/` on the first run — no manual
download step is needed.

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
- **Reproducibility:** seed controls model init, sampler shuffle, and dataloader
  worker streams.

Run locally (with the virtualenv from **Setup** active):

```bash
python keyword_spotting/src/main.py --help
```

A representative training run uses `--use-new-gru --spec-augment
--balanced-sampler --label-smoothing 0.05 --lr-scheduler cosine --epochs 325`.

## PTQ (Post-Training Quantisation)

`src/calibrate_ptq.py` is the PTQ entrypoint. It loads an FP32 checkpoint,
runs percentile-based INT8 calibration, evaluates accuracy, writes
`scales.json`, `summary.json`, and `size_report.json` to `--out-dir`, and
prints a full "Static model footprint / Memory bandwidth / Layer-by-layer"
breakdown to stdout. The quantisation modules live in
[keyword_spotting/src/ptq/](keyword_spotting/src/ptq/).

## Experiment infrastructure

Every experiment is a SLURM array job under [keyword_spotting/slurm/](keyword_spotting/slurm/),
parameterised so a single submission sweeps a Cartesian product of settings.
Active scripts:

- `run_keyword_gru_size_sweep.sbatch` — hidden size sweep (16/32/48/64)
- `run_ptq_sweep.sbatch` — PTQ calibration sweep across hidden sizes and seeds
- `run_ptq_single.sbatch` — single PTQ calibration run

The sbatch scripts resolve the project and virtualenv from the `PROJECT_ROOT`
and `VENV_ACTIVATE` environment variables (they fall back to the author's
cluster paths). Set them for your own environment when submitting:

```bash
PROJECT_ROOT=/path/to/keyword_spotting \
VENV_ACTIVATE=/path/to/.venv/bin/activate \
sbatch keyword_spotting/slurm/run_ptq_sweep.sbatch
```

Each task writes its training log into a per-job directory under `slurm/<jobid>/`,
and the best checkpoint into `models/<jobid>/`. Plots of train/val/test curves
are emitted under `plots/<jobid>/` so a finished sweep can be inspected by
opening one directory. Reference PTQ outputs from completed sweeps are committed
under [keyword_spotting/results/](keyword_spotting/results/).

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

## Tests

[keyword_spotting/tests/](keyword_spotting/tests/) holds numerical-parity and
quantisation checks (custom GRU vs `torch.nn.GRU`, PTQ round-trip). Run them with:

```bash
cd keyword_spotting && python -m pytest tests/
```

## Tech stack

PyTorch, torchaudio, TorchScript, NumPy, Matplotlib, SLURM. Target deployment
is a custom FPGA — the Python side of this repo is the design and validation
sandbox for that target.
