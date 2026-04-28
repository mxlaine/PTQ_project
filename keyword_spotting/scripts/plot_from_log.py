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
feature = None
n_mels = None
use_delta = None
use_delta_delta = None
specaugment = None
freq_mask = None
time_mask = None

m_feat = re.search(r"Running feature config:\s*(\S+)", text)
if not m_feat:
    m_feat = re.search(r"Feature config:\s*(\S+)", text)
if m_feat:
    feature = m_feat.group(1)

m_n = re.search(r"'n_mels'\s*:\s*(\d+)", text)
if m_n:
    n_mels = m_n.group(1)

m_ud = re.search(r"'use_delta'\s*:\s*(True|False)", text)
if m_ud:
    use_delta = m_ud.group(1)

m_udd = re.search(r"'use_delta_delta'\s*:\s*(True|False)", text)
if m_udd:
    use_delta_delta = m_udd.group(1)

m_sa = re.search(r"SpecAugment[: ]+.*freq_mask=?\s*(\d+).*time_mask=?\s*(\d+)", text)
if m_sa:
    specaugment = True
    freq_mask = m_sa.group(1)
    time_mask = m_sa.group(2)

meta_parts = []
if feature:
    meta_parts.append(f"feature={feature}")
if n_mels:
    meta_parts.append(f"n_mels={n_mels}")
if use_delta is not None:
    meta_parts.append(f"use_delta={use_delta}")
if use_delta_delta is not None:
    meta_parts.append(f"use_delta_delta={use_delta_delta}")
if specaugment:
    meta_parts.append(f"SpecAugment(freq={freq_mask},time={time_mask})")

meta_txt = " | ".join(meta_parts)

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
