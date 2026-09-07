#!/usr/bin/env python3
"""Plot 4 summary charts for every group of 4 tasks in a job directory.

Usage: python scripts/plot_group.py <job_id>

Example:
  python scripts/plot_group.py 17716652
  -> discovers all *.out files, groups by task//4 (stride 4),
     and writes plots/<job_id>/group_<start>_*.png for each group:
       _accuracy.png   — test acc vs hidden_size
       _curves.png     — validation curves per hidden size
       _final.png      — final test acc with error bars
       _pareto.png     — MACs vs test acc
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
from summarize_results import parse_log

# Fields considered when building legend labels.
_HPARAM_FIELDS = ("feature", "lr", "wd", "scheduler", "dropout",
                  "balanced", "new_gru", "ema")


def _varying_fields(rows):
    return {f for f in _HPARAM_FIELDS if len({r.get(f, "-") for r in rows}) > 1}


def _hparam_label(r, vary):
    """Label for a single row showing only fields that vary (plus h=hidden always)."""
    parts = [f"h={r['_hidden']}"]
    for f in _HPARAM_FIELDS:
        if f not in vary:
            continue
        v = r.get(f, "-")
        if v in ("-", "False"):
            continue
        parts.append(v if f in ("feature", "scheduler") else f"{f}={v}")
    return " | ".join(parts)


def _shared_meta(rows):
    """Title line showing hyperparam values that are the same for all rows."""
    r0 = rows[0]
    parts = []
    for f in _HPARAM_FIELDS:
        vals = {r.get(f, "-") for r in rows}
        if len(vals) != 1:
            continue
        v = next(iter(vals))
        if v in ("-", "False"):
            continue
        parts.append(v if f in ("feature", "scheduler") else f"{f}={v}")
    return "  |  ".join(parts)


def find_log(job_dir: Path, job_id: str, task: int):
    matches = list(job_dir.glob(f"*-{job_id}_{task}.out"))
    return matches[0] if matches else None


def _to_float(s, default=None):
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return default


def _to_int(s, default=None):
    v = _to_float(s)
    return int(v) if v is not None else default


def _fmt_macs(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def plot_accuracy(rows, tasks, job_id, meta, out_dir: Path):
    xs = [r["_hidden"] for r in rows]
    ys = [r["_test_acc"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o", color="steelblue", linewidth=1.5, markersize=7)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_ylim(max(0, min(ys) - 1.0), max(ys) + 0.8)
    ax.set_xticks(xs)
    ax.set_xlabel("hidden_size")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"Tasks {tasks[0]}–{tasks[-1]}  ({job_id})\n{meta}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"group_{tasks[0]}_accuracy.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Wrote {p}")


def plot_curves(rows, tasks, job_id, meta, out_dir: Path):
    vary = _varying_fields(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    for ci, r in enumerate(rows):
        text = r["_path"].read_text(errors="replace")
        val_accs = [float(m.group(1)) for m in re.finditer(r"Val: .*\((\d+\.?\d*)%\)", text)]
        if not val_accs:
            continue
        ep = range(1, len(val_accs) + 1)
        every = max(1, len(val_accs) // 20)
        label = f"{_hparam_label(r, vary)} (test={r['_test_acc']:.2f}%)"
        ax.plot(ep, val_accs, linewidth=1.2, alpha=0.85,
                marker="o", markevery=every, markersize=4,
                label=label, color=f"C{ci}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title(f"Validation curves — tasks {tasks[0]}–{tasks[-1]}  ({job_id})\n{meta}", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = out_dir / f"group_{tasks[0]}_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Wrote {p}")


def plot_final(rows, tasks, job_id, meta, out_dir: Path):
    vary = _varying_fields(rows)
    # Group by hparam key (everything except hidden) to aggregate seeds if present.
    by_key = defaultdict(list)
    for r in rows:
        key = tuple(r.get(f, "-") for f in _HPARAM_FIELDS)
        by_key[key].append(r)

    hiddens = sorted({r["_hidden"] for r in rows})

    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, (key, group_rows) in enumerate(sorted(by_key.items())):
        by_h = defaultdict(list)
        for r in group_rows:
            by_h[r["_hidden"]].append(r["_test_acc"])
        means, stds, xs = [], [], []
        for h in hiddens:
            accs = by_h.get(h, [])
            if not accs:
                continue
            n = len(accs)
            m = sum(accs) / n
            s = (sum((a - m) ** 2 for a in accs) / n) ** 0.5 if n > 1 else 0.0
            xs.append(h)
            means.append(m)
            stds.append(s)
        # Build label from first row of this group
        sample = next(r for r in group_rows)
        label = _hparam_label(sample, vary) if vary else None
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3,
                    linewidth=1.5, markersize=6,
                    label=label, color=f"C{ci}")
        for x, y in zip(xs, means):
            ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xticks(hiddens)
    ax.set_xlabel("hidden_size")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"Final accuracy — tasks {tasks[0]}–{tasks[-1]}  ({job_id})\n{meta}", fontsize=10)
    ax.grid(True, alpha=0.3)
    if vary:
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / f"group_{tasks[0]}_final.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Wrote {p}")


def plot_pareto(rows, tasks, job_id, meta, out_dir: Path):
    vary = _varying_fields(rows)
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    by_key = defaultdict(list)
    for r in rows:
        key = tuple(r.get(f, "-") for f in _HPARAM_FIELDS)
        by_key[key].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, (key, group_rows) in enumerate(sorted(by_key.items())):
        pts = []
        for r in group_rows:
            macs = _to_int(r.get("macs"))
            params = _to_int(r.get("params"))
            if macs is None:
                continue
            size = 20 + (params / 1000.0 if params else 0)
            pts.append((macs, r["_test_acc"], size, r))
        if not pts:
            continue
        pts.sort(key=lambda t: t[0])
        xs = [t[0] for t in pts]
        ys = [t[1] for t in pts]
        sizes = [t[2] for t in pts]
        sample = pts[0][3]
        label = _hparam_label(sample, vary) if vary else None
        ax.plot(xs, ys, linewidth=1.2, alpha=0.6, color=f"C{ci}")
        ax.scatter(xs, ys, s=sizes, marker=markers[ci % len(markers)],
                   label=label, alpha=0.85, edgecolors="black", linewidths=0.5,
                   zorder=3, color=f"C{ci}")
        for x, y, _, r in pts:
            ax.annotate(f"h={r['_hidden']}\n{_fmt_macs(x)} MACs", (x, y),
                        textcoords="offset points", xytext=(4, 4), fontsize=8,
                        linespacing=1.4)

    ax.set_xscale("log")
    ax.set_xlabel("MACs per inference (log scale)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"MACs vs accuracy — tasks {tasks[0]}–{tasks[-1]}  ({job_id})\n{meta}", fontsize=10)
    ax.grid(True, alpha=0.3)
    if vary:
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / f"group_{tasks[0]}_pareto.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Wrote {p}")


def _discover_tasks(job_dir: Path, job_id: str) -> list[int]:
    pattern = re.compile(rf"-{re.escape(job_id)}_(\d+)\.out$")
    nums = []
    for p in job_dir.iterdir():
        m = pattern.search(p.name)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def _parse_rows(job_dir, job_id, tasks):
    rows = []
    for t in tasks:
        path = find_log(job_dir, job_id, t)
        if path is None:
            print(f"  task {t}: no log found, skipping", file=sys.stderr)
            continue
        r = parse_log(path)
        if r["test_acc"] == "-":
            print(f"  task {t}: no test_acc yet, skipping", file=sys.stderr)
            continue
        try:
            r["_task"] = t
            r["_path"] = path
            r["_test_acc"] = float(str(r["test_acc"]).replace(",", ""))
            r["_hidden"] = int(float(str(r["hidden"]).replace(",", "")))
        except (ValueError, TypeError):
            print(f"  task {t}: could not parse fields, skipping", file=sys.stderr)
            continue
        rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job_id", metavar="JOB_ID", help="e.g. 17716652")
    args = parser.parse_args()

    job_id = args.job_id
    job_dir = ROOT / "slurm" / job_id
    if not job_dir.is_dir():
        print(f"No job directory: {job_dir}", file=sys.stderr)
        sys.exit(1)

    all_tasks = _discover_tasks(job_dir, job_id)
    if not all_tasks:
        print("No .out files found.", file=sys.stderr)
        sys.exit(1)

    # Group by stride of 4: tasks 0-3, 4-7, 8-11, …
    groups: dict[int, list[int]] = defaultdict(list)
    for t in all_tasks:
        groups[t // 4 * 4].append(t)

    out_dir = ROOT / "plots" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    plotted = 0
    for start, group_tasks in sorted(groups.items()):
        print(f"\n--- group starting at task {start}: {group_tasks} ---")
        rows = _parse_rows(job_dir, job_id, group_tasks)
        if not rows:
            print(f"  no completed tasks in group, skipping", file=sys.stderr)
            continue
        rows.sort(key=lambda r: r["_hidden"])
        meta = _shared_meta(rows)
        group_dir = out_dir / f"group_{start}"
        group_dir.mkdir(exist_ok=True)
        plot_accuracy(rows, group_tasks, job_id, meta, group_dir)
        plot_curves(rows, group_tasks, job_id, meta, group_dir)
        plot_final(rows, group_tasks, job_id, meta, group_dir)
        plot_pareto(rows, group_tasks, job_id, meta, group_dir)
        plotted += 1

    if not plotted:
        print("No completed groups found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
