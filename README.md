# Keyword Spotting on FPGA-Constrained Hardware

This work is part of an IC Design project. The model has to be implemented in
hardware, so a number of constraints have to be accounted for:


- Architecture: 2 stacked unidirectional GRU layers + linear head
- Max hidden size: 64 (≈ 47k parameters total)
- Output classes: 12 (10 keywords + unknown + silence)
- Dataset: Google Speech Commands v2

## Setup

```bash
git clone git@github.com:mxlaine/PTQ_project.git
cd PTQ_project
python -m venv .venv
source .venv/bin/activate
pip install -r keyword_spotting/requirements.txt
```

The Google Speech Commands v2 dataset is downloaded automatically by
`torchaudio` into `keyword_spotting/data/` on the first run — no manual
download step is needed.

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

Run locally (with the virtualenv from **Setup** active):

```bash
python keyword_spotting/src/main.py --help
```

Example training run `--use-new-gru --spec-augment
--balanced-sampler --label-smoothing 0.05 --lr-scheduler cosine --epochs 325`.

## PTQ (Post-Training Quantization)

`src/calibrate_ptq.py` loads an FP32 checkpoint,
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

## Tests

[keyword_spotting/tests/](keyword_spotting/tests/) holds numerical-parity and
quantisation checks (custom GRU vs `torch.nn.GRU`, PTQ round-trip). Run them with:

```bash
cd keyword_spotting && python -m pytest tests/
```

## Tech stack

PyTorch, torchaudio, TorchScript, NumPy, Matplotlib, SLURM.
