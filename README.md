# PTQ Project

PyTorch code for the IC Design Project on keyword spotting task on Google Speech Commands v2 dataset +
CNN experimentation.

## Overview

The main project lives in `keyword_spotting/` and trains a GRU-based keyword
spotting model on Google Speech Commands v2. The model is intended for FPGA
deployment, so the architecture is limited to a fixed 2x unidirectional GRU
plus a linear head. Model has approx. 47k params with max hidden layer size of 64.

`cnn_testing/` contains a separate MNIST notebook used for CNN experiments.

## Repository Layout

- `keyword_spotting/src/` - training, model, and data handling code
- `keyword_spotting/slurm/` - SLURM batch scripts for sweeps and runs
- `keyword_spotting/models/` - saved checkpoints from training jobs
- `keyword_spotting/plots/` - training curves generated from job logs
- `keyword_spotting/data/` - local Speech Commands dataset files
- `cnn_testing/` - MNIST notebook and data

## Keyword Spotting Workflow

Activate the virtual environment and run the training entrypoint directly:

```bash
source .venv/bin/activate
python keyword_spotting/src/main.py --help
```

For SLURM runs, use the batch scripts in `keyword_spotting/slurm/`. The main
sweep script is `keyword_spotting/slurm/run_keyword_gru.sbatch`.

The training code supports SpecAugment, speed perturbation, LR warmup, and
balanced sampling. Checkpoints and plots are written under `keyword_spotting/`
and are ignored by git.

