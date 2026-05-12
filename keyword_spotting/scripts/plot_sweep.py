#!/usr/bin/env python3
"""Produce four sweep-summary plots for a slurm job:

  1. accuracy_vs_hidden_size.png — one line per feature config, mean ± std band.
  2. training_curves_best.png    — accuracy vs epoch for the best seed per (feature, hidden) cell.
  3. bar_final_accuracy.png      — grouped bars by (feature, hidden), seed-error bars.
  4. macs_vs_accuracy_pareto.png — scatter of MACs (log) vs mean test_acc.

Usage: python scripts/plot_sweep.py <job_id>

Reads logs from slurm/<job_id>/keyword-gru-*.out and writes PNGs to plots/<job_id>/.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_results import parse_log  # reuse log parser


def _to_float(s, default=None):
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return default


def _to_int(s, default=None):
    v = _to_float(s)
    return int(v) if v is not None else default


def collect(job_dir: Path):
    """Parse all logs, return list of dicts with numeric fields where possible."""
    logs = sorted(job_dir.glob("keyword-gru-*.out"))
    rows = []
    for p in logs:
        r = parse_log(p)
        r["_path"] = p
        r["_test_acc"] = _to_float(r["test_acc"])
        r["_hidden"] = _to_int(r["hidden"])
        r["_seed"] = _to_int(r["seed"], default=0)
        r["_macs"] = _to_int(r["macs"])
        r["_params"] = _to_int(r["params"])
        rows.append(r)
    return [r for r in rows if r["_test_acc"] is not None and r["_hidden"] is not None]


# Fields that, when they vary, distinguish different experimental conditions.
_HPARAM_FIELDS = ("feature", "lr", "wd", "scheduler", "dropout",
                  "balanced", "new_gru", "ema")


def _varying_fields(rows):
    """Return the subset of _HPARAM_FIELDS that take >1 distinct value across rows."""
    return {f for f in _HPARAM_FIELDS if len({r.get(f, "-") for r in rows}) > 1}


def _hparam_key(r):
    return tuple(r.get(f, "-") for f in _HPARAM_FIELDS)


def _hparam_label(key, vary):
    """Short label showing only the fields that actually vary."""
    parts = []
    for f, v in zip(_HPARAM_FIELDS, key):
        if f not in vary or v in ("-", "False"):
            continue
        if f in ("feature", "scheduler"):
            parts.append(v)
        elif f == "wd" and v in ("0", "0.0", "0e+00", "0e0"):
            continue
        else:
            parts.append(f"{f}={v}")
    return " | ".join(parts) if parts else str(key)


def group_by_hparams(rows):
    """Returns {hparam_key: [rows...]} grouping everything except hidden_size."""
    groups = defaultdict(list)
    for r in rows:
        groups[_hparam_key(r)].append(r)
    return groups


def group_by_feature_hidden(rows):
    """Returns {(feature, hidden): [rows...]}."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["feature"], r["_hidden"])].append(r)
    return groups


def mean_std(values):
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5


def plot_accuracy_vs_hidden(rows, outpath: Path):
    vary = _varying_fields(rows)
    by_key = defaultdict(list)
    for r in rows:
        by_key[_hparam_key(r)].append(r)

    plt.figure(figsize=(9, 5))
    for ki, key in enumerate(sorted(by_key)):
        items_by_hidden = defaultdict(list)
        for r in by_key[key]:
            items_by_hidden[r["_hidden"]].append(r["_test_acc"])
        pts = sorted((h, *mean_std(accs)) for h, accs in items_by_hidden.items())
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        label = _hparam_label(key, vary)
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label, color=f"C{ki}")
        ys_lo = [y - e for y, e in zip(ys, es)]
        ys_hi = [y + e for y, e in zip(ys, es)]
        plt.fill_between(xs, ys_lo, ys_hi, alpha=0.12, color=f"C{ki}")
    plt.xlabel("hidden_size")
    plt.ylabel("test accuracy (%)")
    plt.title("Test accuracy vs hidden_size (mean ± std across seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_training_curves_best(rows, outpath: Path):
    vary = _varying_fields(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(_hparam_key(r), r["_hidden"])].append(r)

    plt.figure(figsize=(10, 6))
    for ci, (key_hidden, items) in enumerate(sorted(groups.items())):
        hpkey, hidden = key_hidden
        best = max(items, key=lambda r: r["_test_acc"])
        text = best["_path"].read_text(errors="replace")
        val_accs = [float(m.group(1)) for m in re.finditer(r"Val: .*\((\d+\.?\d*)%\)", text)]
        if not val_accs:
            continue
        ep = range(1, len(val_accs) + 1)
        every = max(1, len(val_accs) // 20)
        base_label = _hparam_label(hpkey, vary)
        label = f"h={hidden} | {base_label} (test={best['_test_acc']:.2f}%)"
        plt.plot(ep, val_accs, linewidth=1.0, alpha=0.8,
                 marker="o", markevery=every, markersize=4,
                 label=label, color=f"C{ci % 10}")
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy (%)")
    plt.title("Validation curves — best seed per config")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=6, loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_bar_final(rows, outpath: Path):
    vary = _varying_fields(rows)
    by_key = defaultdict(list)
    for r in rows:
        by_key[_hparam_key(r)].append(r)
    hiddens = sorted({r["_hidden"] for r in rows})

    plt.figure(figsize=(10, 5))
    for ki, key in enumerate(sorted(by_key)):
        items_by_hidden = defaultdict(list)
        for r in by_key[key]:
            items_by_hidden[r["_hidden"]].append(r["_test_acc"])
        means, stds = [], []
        for h in hiddens:
            m, s = mean_std(items_by_hidden.get(h, []))
            means.append(m if m is not None else float("nan"))
            stds.append(s if s is not None else 0.0)
        label = _hparam_label(key, vary)
        plt.errorbar(hiddens, means, yerr=stds, marker="o", capsize=3,
                     linewidth=1.5, markersize=6, label=label, color=f"C{ki}")
    plt.xticks(hiddens)
    plt.xlabel("hidden_size")
    plt.ylabel("test accuracy (%)")
    plt.title("Final test accuracy vs hidden_size — mean ± std across seeds")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_macs_pareto(rows, outpath: Path):
    vary = _varying_fields(rows)
    by_key = defaultdict(list)
    for r in rows:
        by_key[_hparam_key(r)].append(r)

    plt.figure(figsize=(9, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for ki, key in enumerate(sorted(by_key)):
        xs, ys, sizes = [], [], []
        items_by_hidden = defaultdict(list)
        for r in by_key[key]:
            items_by_hidden[r["_hidden"]].append(r)
        for h, items in items_by_hidden.items():
            macs_vals = [r["_macs"] for r in items if r["_macs"] is not None]
            params_vals = [r["_params"] for r in items if r["_params"] is not None]
            accs = [r["_test_acc"] for r in items]
            if not macs_vals or not accs:
                continue
            mean_acc, _ = mean_std(accs)
            xs.append(macs_vals[0])
            ys.append(mean_acc)
            sizes.append(20 + (params_vals[0] / 1000.0 if params_vals else 0))
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs_s = [xs[i] for i in order]
            ys_s = [ys[i] for i in order]
            sizes_s = [sizes[i] for i in order]
            label = _hparam_label(key, vary)
            plt.plot(xs_s, ys_s, linewidth=1.2, alpha=0.6, color=f"C{ki}")
            plt.scatter(xs_s, ys_s, s=sizes_s, marker=markers[ki % len(markers)],
                        label=label, alpha=0.85, edgecolors="black", linewidths=0.5,
                        zorder=3, color=f"C{ki}")
    plt.xscale("log")
    plt.xlabel("MACs per inference (log scale)")
    plt.ylabel("mean test accuracy (%)")
    plt.title("MACs vs accuracy — Pareto view (marker size ∝ params)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="slurm array job id (directory under slurm/)")
    args = parser.parse_args()

    job_dir = ROOT / "slurm" / args.job_id
    if not job_dir.is_dir():
        print(f"No job directory at {job_dir}", file=sys.stderr)
        sys.exit(1)

    rows = collect(job_dir)
    if not rows:
        print("No parseable logs found.", file=sys.stderr)
        sys.exit(1)

    out_dir = ROOT / "plots" / args.job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_vs_hidden(rows, out_dir / "accuracy_vs_hidden_size.png")
    plot_training_curves_best(rows, out_dir / "training_curves_best.png")
    plot_bar_final(rows, out_dir / "bar_final_accuracy.png")
    plot_macs_pareto(rows, out_dir / "macs_vs_accuracy_pareto.png")
    print(f"Wrote 4 plots to {out_dir}")


if __name__ == "__main__":
    main()
