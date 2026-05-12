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

plt.figure(figsize=(10,5))
ms = 3
lw = 1.0
if train:
    plt.plot(range(1, len(train)+1), train, marker='^', markersize=ms, linewidth=lw, label='Training Accuracy', alpha=0.8)
if val:
    plt.plot(range(1, len(val)+1), val, marker='o', markersize=ms, linewidth=lw, label='Validation Accuracy', alpha=0.8)
if test_acc is not None:
    plt.axhline(y=test_acc, color='r', linestyle='--', linewidth=lw, label='Final Test Accuracy')
    # annotate near the end of validation curve
    x_annot = len(val) if val else (len(train) if train else 0)
    if x_annot == 0:
        x_annot = 1
    plt.annotate(f"Test: {test_acc:.2f}%", (x_annot, test_acc), textcoords='offset points', xytext=(6,6), fontsize=8)

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title(f'Training and Validation Accuracy ({job}_{task})')
if meta_txt:
    plt.suptitle(meta_txt, fontsize=9, y=0.99)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(outfile)
print(f"Saved plot to {outfile}")
