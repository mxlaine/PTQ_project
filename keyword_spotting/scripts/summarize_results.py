#!/usr/bin/env python3
"""Scan slurm/<job>/<prefix>-<job>_<task>.out files and print a sorted results table.

By default prints one row per log file, sorted by test_acc descending.
With --aggregate, groups rows that share (feature, hidden, dropout, scheduler, lr,
warmup, freq, time, balanced, new_gru, ema) and reports mean ± std of test_acc
across seeds.
"""
import argparse
import re
import statistics
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
    num_layers = _first(r"Num layers:\s*(\d+)", text)
    epochs = _first(r"Epochs:\s*(\d+)", text)
    lr = _first(r"LR:\s*([0-9e.+-]+)", text)
    wd = _first(r"weight_decay=([0-9e.+-]+)", text)
    speed = _first(r"Speed perturbation:\s*(True|False)", text)
    warmup = _first(r"LR warmup epochs:\s*(\d+)", text)
    scheduler = _first(r"LR scheduler:\s*(\S+)", text)
    balanced = _first(r"Balanced sampler:\s*(True|False)", text)
    dropout = _first(r"Dropout:\s*([0-9]+\.?[0-9]*)", text)
    params = _first(r"Total model parameters:\s*([\d,]+)", text)
    macs = _first(r"Total MACs:\s*([\d,]+)", text)
    seed = _first(r"Seed:\s*(\d+)", text)
    new_gru = _first(r"New GRU:\s*(True|False)", text)
    ema = _first(r"EMA:\s*(True|False)", text)
    ema_test = _first(r"EMA Test: .*\((\d+\.?\d*)%\)", text)
    best_val = "-"
    for m in re.finditer(r"saved best:\s*(\d+\.?\d*)%", text):
        best_val = m.group(1)

    mm = re.search(r"(?:keyword-gru[^-]*-)?(\d{6,})_(\d+)\.out$", path.name)
    job_id = mm.group(1) if mm else "-"
    task_id = mm.group(2) if mm else "-"

    return {
        "job": job_id,
        "task": task_id,
        "best_val": best_val,
        "test_acc": test_acc,
        "feature": feature,
        "n_mels": n_mels,
        "dropout": dropout,
        "delta": use_delta,
        "dd": use_delta_delta,
        "freq": freq_mask,
        "time": time_mask,
        "hidden": hidden,
        "layers": num_layers,
        "epochs": epochs,
        "lr": lr,
        "wd": wd,
        "scheduler": scheduler,
        "speed": speed,
        "warmup": warmup,
        "balanced": balanced,
        "new_gru": new_gru,
        "ema": ema,
        "ema_test": ema_test,
        "params": params,
        "macs": macs,
        "seed": seed,
    }


GROUP_KEYS = ("feature", "hidden", "dropout", "scheduler", "lr", "wd", "warmup",
              "freq", "time", "balanced", "new_gru", "ema")


def aggregate_rows(rows):
    """Group by GROUP_KEYS and produce one row per group with mean ± std of test_acc."""
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in GROUP_KEYS)
        groups.setdefault(key, []).append(r)

    out = []
    for key, items in groups.items():
        vals = []
        for it in items:
            try:
                vals.append(float(it["test_acc"]))
            except (ValueError, TypeError):
                pass
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        sample = items[0]
        agg = {k: sample[k] for k in GROUP_KEYS}
        agg["n_seeds"] = str(len(vals))
        agg["test_mean"] = f"{mean:.2f}"
        agg["test_std"] = f"{std:.2f}"
        agg["params"] = sample["params"]
        agg["macs"] = sample["macs"]
        out.append(agg)

    out.sort(key=lambda r: -float(r["test_mean"]))
    return out


def print_table(rows, headers, col_keys):
    if not rows:
        print("(no rows)")
        return
    col_widths = [max(len(h), max((len(str(r.get(k, "-"))) for r in rows), default=0)) for h, k in zip(headers, col_keys)]
    sep = "  ".join("-" * w for w in col_widths)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print(sep)
    for row in rows:
        line = "  ".join(str(row.get(k, "-")).ljust(w) for k, w in zip(col_keys, col_widths))
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", action="store_true",
                        help="Group rows by (feature, hidden, ...) and report mean ± std of test_acc across seeds")
    parser.add_argument("--job", default=None,
                        help="Restrict to logs from this slurm job id (e.g. 17620206)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    slurm_dir = root / "slurm"

    if not slurm_dir.is_dir():
        print(f"No slurm directory found at {slurm_dir}", file=sys.stderr)
        sys.exit(1)

    if args.job:
        logs = sorted((slurm_dir / args.job).glob("keyword-gru-*.out"))
    else:
        logs = sorted(slurm_dir.glob("*/keyword-gru-*.out"))
    if not logs:
        print("No keyword-gru-*.out files found.", file=sys.stderr)
        sys.exit(1)

    rows = [parse_log(p) for p in logs]

    if args.aggregate:
        agg = aggregate_rows(rows)
        headers = ["feature", "hidden", "dropout", "scheduler", "lr", "wd", "warmup",
                   "freq", "time", "balanced", "new_gru", "ema",
                   "n_seeds", "test_mean", "test_std", "params", "macs"]
        col_keys = headers
        print_table(agg, headers, col_keys)
        return

    def sort_key(r):
        try:
            return -float(r["test_acc"])
        except (ValueError, TypeError):
            return 1.0

    rows.sort(key=sort_key)

    headers  = ["job", "task", "seed", "best_val", "test_acc", "ema_test", "feature", "n_mels", "dropout", "delta", "dd", "freq", "time", "hidden", "layers", "epochs", "lr", "wd", "scheduler", "speed", "warmup", "balanced", "new_gru", "ema", "params", "macs"]
    col_keys = headers
    print_table(rows, headers, col_keys)


if __name__ == "__main__":
    main()
