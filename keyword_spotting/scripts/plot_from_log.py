#!/usr/bin/env python3
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print('Usage: plot_from_log.py <logfile>')
    sys.exit(1)

logpath = Path(sys.argv[1])
text = logpath.read_text()

train = [float(m.group(1)) for m in re.finditer(r"Train Acc: (\d+\.?\d*)%", text)]
val = [float(m.group(1)) for m in re.finditer(r"Val: .*\((\d+\.?\d*)%\)", text)]

m = re.search(r"Test: .*\((\d+\.?\d*)%\)", text)
if m:
    test_acc = float(m.group(1))
else:
    test_acc = None

# derive job/task from filename if possible
bn = logpath.name
job = 'local'
task = '0'
mm = re.search(r"keyword-gru-(\d+)_(\d+)\.out", bn)
if mm:
    job, task = mm.group(1), mm.group(2)

outdir = Path(__file__).resolve().parents[1] / 'plots' / job
outdir.mkdir(parents=True, exist_ok=True)
outfile = outdir / f"train_val_from_log_{job}_{task}.png"
log_outfile = outdir / f"train_val_from_log_{job}_{task}_log.png"

# Parse sweep metadata for top-of-plot annotation
def _search(pattern, text, group=1):
    m = re.search(pattern, text)
    return m.group(group) if m else None

feature = _search(r"Running feature config:\s*(\S+)", text) or _search(r"Feature config:\s*(\S+)", text)
n_mels = _search(r"'n_mels'\s*:\s*(\d+)", text)
use_delta = _search(r"'use_delta'\s*:\s*(True|False)", text)
use_delta_delta = _search(r"'use_delta_delta'\s*:\s*(True|False)", text)
speed_perturb = _search(r"Speed perturbation:\s*(True|False)", text)
balanced_sampler = _search(r"Balanced sampler:\s*(True|False)", text)

m_sa = re.search(r"SpecAugment[: ]+.*freq_mask=?\s*(\d+).*time_mask=?\s*(\d+)", text)
freq_mask = m_sa.group(1) if m_sa else None
time_mask = m_sa.group(2) if m_sa else None

hidden = _search(r"Hidden size:\s*(\d+)", text)
num_layers = _search(r"Num layers:\s*(\d+)", text)
dropout = _search(r"Dropout:\s*([\d.]+)", text)
lr = _search(r"^LR:\s*([\d.e+-]+)", text) or _search(r"\bLR:\s*([\d.e+-]+)", text)
warmup = _search(r"LR warmup epochs:\s*(\d+)", text)
new_gru = _search(r"New GRU:\s*(True|False)", text)

m_sched = re.search(r"LR scheduler:\s*cosine-warm-restarts T0=(\d+) T_mult=(\d+)", text)
if m_sched:
    scheduler_str = f"cosine-warm-restarts(T0={m_sched.group(1)},Tm={m_sched.group(2)})"
elif re.search(r"LR scheduler:\s*cosine", text):
    scheduler_str = "cosine"
else:
    scheduler_str = None

line1 = []
if feature:
    line1.append(f"feature={feature}")
if n_mels:
    line1.append(f"n_mels={n_mels}")
if use_delta is not None:
    line1.append(f"use_delta={use_delta}")
if use_delta_delta is not None:
    line1.append(f"use_delta_delta={use_delta_delta}")
if m_sa:
    line1.append(f"SpecAugment(freq={freq_mask},time={time_mask})")
if speed_perturb == "True":
    line1.append("speed_perturb=True")
if balanced_sampler == "True":
    line1.append("balanced_sampler=True")
if new_gru == "True":
    line1.append("new_gru=True")

line2 = []
if hidden:
    line2.append(f"hidden_size={hidden}")
if num_layers:
    line2.append(f"layers={num_layers}")
if dropout:
    line2.append(f"dropout={dropout}")
if lr:
    line2.append(f"lr={lr}")
if warmup:
    line2.append(f"warmup_epochs={warmup}")
if scheduler_str:
    line2.append(f"scheduler={scheduler_str}")

meta_txt = " | ".join(line1)
if line2:
    meta_txt += "\n" + " | ".join(line2)

ms = 3
lw = 1.0


def draw_plot(train_y, val_y, test_y, test_label, ylabel, title, outpath, logscale):
    fig, ax = plt.subplots(figsize=(10, 5))
    if train_y:
        ax.plot(range(1, len(train_y)+1), train_y, marker='^', markersize=ms,
                linewidth=lw, label='Training', alpha=0.8)
    if val_y:
        ax.plot(range(1, len(val_y)+1), val_y, marker='o', markersize=ms,
                linewidth=lw, label='Validation', alpha=0.8)
    if test_y is not None:
        # Test marker: a short dash at the right edge + label outside the axes,
        # so it never obstructs the training/validation curves.
        ax.axhline(y=test_y, xmin=0.97, xmax=1.0, color='r', linewidth=2.0)
        ax.annotate(
            test_label,
            xy=(1.0, test_y), xycoords=('axes fraction', 'data'),
            xytext=(5, 0), textcoords='offset points',
            va='center', ha='left', fontsize=8, color='r',
            annotation_clip=False,
        )
    if logscale:
        ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if meta_txt:
        fig.suptitle(meta_txt, fontsize=9, y=0.99)
    ax.grid(True, which='both', alpha=0.3)
    if train_y or val_y:
        ax.legend(loc='best')
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved plot to {outpath}")


# Linear accuracy plot
draw_plot(
    train, val, test_acc,
    f"Test {test_acc:.2f}%" if test_acc is not None else "",
    'Accuracy (%)',
    f'Training and Validation Accuracy ({job}_{task})',
    outfile, logscale=False,
)

# Log-scale error-rate plot: error = 100 - accuracy on a log y-axis, which
# spreads out near-ceiling differences for comparison with SotA KWS methods
# (e.g. BC-ResNet). EPS keeps perfect-accuracy points loggable.
EPS = 0.05
train_err = [max(100.0 - a, EPS) for a in train]
val_err = [max(100.0 - a, EPS) for a in val]
test_err = max(100.0 - test_acc, EPS) if test_acc is not None else None
draw_plot(
    train_err, val_err, test_err,
    f"Test {test_err:.2f}%" if test_err is not None else "",
    'Error rate (%)',
    f'Training and Validation Error — log scale ({job}_{task})',
    log_outfile, logscale=True,
)
