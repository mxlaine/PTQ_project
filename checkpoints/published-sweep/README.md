# Checkpoints for the README results

These are the 12 original FP32 state dictionaries used by PTQ sweep `17803125`:
16 log-mel bands with Δ/ΔΔ, hidden sizes 16/32/48/64, and seeds 0/1/2. They were
recovered from the Triton `keyword_spotting.zip` archive, without modifying the
checkpoint bytes. Together they occupy 1,441,752 bytes (about 1.38 MiB).

[manifest.json](manifest.json) maps each checkpoint to its archived summary,
original cluster path, SHA-256 checksum, [training log](training-logs/), and [PTQ evaluation log](evaluation-logs/).
Paths used for local files are relative to the repository root. The original
summaries remain unchanged so their cluster provenance is preserved.

The training logs are from job `17792221`. They record 325 epochs, AdamW with
learning rate 0.002 and zero weight decay, cosine scheduling with 10 warmup
epochs, dropout 0.1, and SpecAugment with frequency mask 5 and time mask 0.
Class-balanced sampling was disabled. Training used a Tesla V100 on Triton.
The logs contain the validation-based checkpoint decisions and final test results.

From the repository root:

```bash
python scripts/reproduce_results.py --all --verify-only
python scripts/reproduce_results.py --hidden-size 64 --seed 0
```

The second command requires the project dependencies and Speech Commands v2
(downloaded on first use). Add `--all` to evaluate all 12 checkpoints. It runs
fresh calibration with batch size 256 (plus 25 synthetic silence clips per full
batch), 16 validation batches, and percentile 99.99;
new reports go to `results/reproduced/`. It does not overwrite
or regenerate the archived published results.

The checkpoints store FP32 weights and frontend buffers, not packed INT8 models
or optimizer state. Other feature configurations and exploratory checkpoints in
the original archive are outside this bundle. Numerical results may vary across
hardware and library versions; the README numbers are the archived Triton run,
not a claim that every new environment has reproduced them.
