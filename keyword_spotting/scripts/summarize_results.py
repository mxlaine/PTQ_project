#!/usr/bin/env python3
"""Scan slurm/<job>/<prefix>-<job>_<task>.out files and print a sorted results table."""
import re
import sys
from pathlib import Path


def _first(pattern, text, default="-"):
    m = re.search(pattern, text)
    return m.group(1) if m else default


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")

    test_acc = _first(r"Test: .*\((\d+\.?\d*)%\)", text)
    feature = _first(r"Feature config:\s*(\S+)\s*->", text)
    if feature == "-":
        feature = _first(r"Running feature config:\s*(\S+)", text)
    n_mels = _first(r"'n_mels'\s*:\s*(\d+)", text)
    use_delta = _first(r"'use_delta'\s*:\s*(True|False)", text)
    use_delta_delta = _first(r"'use_delta_delta'\s*:\s*(True|False)", text)

    sa_m = re.search(r"SpecAugment:\s*(True|False)\s+freq_mask=(\d+)\s+time_mask=(\d+)", text)
    if sa_m and sa_m.group(1) == "True":
        freq_mask = sa_m.group(2)
        time_mask = sa_m.group(3)
    else:
        freq_mask = "-"
        time_mask = "-"

    hidden = _first(r"Hidden size:\s*(\d+)", text)
    speed = _first(r"Speed perturbation:\s*(True|False)", text)
    warmup = _first(r"LR warmup epochs:\s*(\d+)", text)
    balanced = _first(r"Balanced sampler:\s*(True|False)", text)

    mm = re.search(r"(?:keyword-gru[^-]*-)?(\d{6,})_(\d+)\.out$", path.name)
    job_id = mm.group(1) if mm else "-"
    task_id = mm.group(2) if mm else "-"

    return {
        "job": job_id,
        "task": task_id,
        "test_acc": test_acc,
        "feature": feature,
        "n_mels": n_mels,
        "delta": use_delta,
        "dd": use_delta_delta,
        "freq": freq_mask,
        "time": time_mask,
        "hidden": hidden,
        "speed": speed,
        "warmup": warmup,
        "balanced": balanced,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    slurm_dir = root / "slurm"

    if not slurm_dir.is_dir():
        print(f"No slurm directory found at {slurm_dir}", file=sys.stderr)
        sys.exit(1)

    logs = sorted(slurm_dir.glob("*/keyword-gru-*.out"))
    if not logs:
        print("No keyword-gru-*.out files found.", file=sys.stderr)
        sys.exit(1)

    rows = [parse_log(p) for p in logs]

    # Sort by test_acc descending; logs with no test result go to the bottom.
    def sort_key(r):
        try:
            return -float(r["test_acc"])
        except (ValueError, TypeError):
            return 1.0

    rows.sort(key=sort_key)

    headers = ["job", "task", "test%", "feature", "n_mels", "delta", "dd", "freq", "time", "hidden", "speed", "warmup", "balanced"]
    col_keys = ["job", "task", "test_acc", "feature", "n_mels", "delta", "dd", "freq", "time", "hidden", "speed", "warmup", "balanced"]

    col_widths = [max(len(h), max((len(str(r[k])) for r in rows), default=0)) for h, k in zip(headers, col_keys)]

    sep = "  ".join("-" * w for w in col_widths)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))

    print(header_line)
    print(sep)
    for row in rows:
        line = "  ".join(str(row[k]).ljust(w) for k, w in zip(col_keys, col_widths))
        print(line)


if __name__ == "__main__":
    main()
