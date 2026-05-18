#!/usr/bin/env python3
"""Build ptq_checkpoint_index.json from a completed training sweep job.

Usage:
    # Full 36-task sweep (run_keyword_gru_size_sweep.sbatch):
    python scripts/build_ptq_index.py --job-id 17XXXXXX

    # Best-config 12-task sweep (run_best_config_seeds.sbatch):
    python scripts/build_ptq_index.py --job-id 17XXXXXX --sweep best-seeds

    # Feature-compare 24-task sweep (run_best_config_feature_compare.sbatch):
    python scripts/build_ptq_index.py --job-id 17XXXXXX --sweep feature-compare
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HIDDEN_SIZES = [16, 32, 48, 64]
FEATURE_CONFIGS = ["16_mels", "16_mels_delta_delta", "48_mels"]
SEEDS = [0, 1, 2]


def _make_full_sweep_tasks():
    # matches run_keyword_gru_size_sweep.sbatch: seed innermost, then hidden, then feature
    tasks = []
    for feat in FEATURE_CONFIGS:
        for hid in HIDDEN_SIZES:
            for seed in SEEDS:
                tasks.append((feat, hid, seed))
    return tasks


def _make_best_seeds_tasks():
    # matches run_best_config_seeds.sbatch: seed_idx = task % 3, hid_idx = task // 3
    tasks = []
    for hid in HIDDEN_SIZES:
        for seed in SEEDS:
            tasks.append(("16_mels_delta_delta", hid, seed))
    return tasks


def _make_feature_compare_tasks():
    # matches run_best_config_feature_compare.sbatch: seed innermost, then hidden,
    # then feature (16_mels, 48_mels).
    tasks = []
    for feat in ["16_mels", "48_mels"]:
        for hid in HIDDEN_SIZES:
            for seed in SEEDS:
                tasks.append((feat, hid, seed))
    return tasks


SWEEPS = {
    "full": {
        "tasks": _make_full_sweep_tasks(),
        "log_prefix": "keyword-gru-size",
    },
    "best-seeds": {
        "tasks": _make_best_seeds_tasks(),
        "log_prefix": "kws-best-seeds",
    },
    "feature-compare": {
        "tasks": _make_feature_compare_tasks(),
        "log_prefix": "kws-feat-compare",
    },
}


def parse_test_acc(log_path: Path) -> float | None:
    pattern = re.compile(r"^Test: loss=[\d.]+, acc=\d+/\d+ \(([\d.]+)%\)")
    try:
        for line in log_path.read_text().splitlines():
            m = pattern.match(line)
            if m:
                return float(m.group(1))
    except FileNotFoundError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Build PTQ checkpoint index from a training sweep job")
    parser.add_argument("--job-id", required=True, help="Slurm array job ID of the completed training sweep")
    parser.add_argument(
        "--sweep",
        choices=list(SWEEPS.keys()),
        default="full",
        help="Which sweep script produced this job (default: full)",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "results" / "ptq_checkpoint_index.json"),
        help="Output JSON path (default: results/ptq_checkpoint_index.json)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge into existing index instead of overwriting (new entries win on conflict)",
    )
    args = parser.parse_args()

    job_id = args.job_id
    sweep = SWEEPS[args.sweep]
    models_dir = PROJECT_ROOT / "models" / job_id
    logs_dir = PROJECT_ROOT / "slurm" / job_id
    log_prefix = sweep["log_prefix"]

    entries = []
    missing = []

    for task_id, (feature_config, hidden_size, seed) in enumerate(sweep["tasks"]):
        ckpt = models_dir / f"best_keyword_gru_{feature_config}_{job_id}_{task_id}.pt"
        log = logs_dir / f"{log_prefix}-{job_id}_{task_id}.out"

        if not ckpt.exists():
            missing.append(f"  task {task_id:2d} ({feature_config} h={hidden_size} s={seed}): checkpoint missing at {ckpt}")
            continue

        test_acc = parse_test_acc(log)
        if test_acc is None:
            missing.append(f"  task {task_id:2d} ({feature_config} h={hidden_size} s={seed}): test acc not found in {log}")
            continue

        entries.append(
            {
                "feature_config": feature_config,
                "hidden_size": hidden_size,
                "seed": seed,
                "checkpoint": str(ckpt),
                "fp32_test_acc": test_acc,
            }
        )

    if missing:
        print(f"WARNING: {len(missing)} task(s) could not be indexed:", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)

    out_path = Path(args.out)

    if args.merge and out_path.exists():
        existing = json.loads(out_path.read_text())
        key = lambda e: (e["feature_config"], e["hidden_size"], e["seed"])
        merged = {key(e): e for e in existing}
        merged.update({key(e): e for e in entries})
        entries = list(merged.values())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2))
    print(f"Wrote {len(entries)} entries to {out_path}")
    if missing:
        print(f"  ({len(missing)} task(s) skipped — see warnings above)")


if __name__ == "__main__":
    main()
