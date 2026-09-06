"""Rebuild README results and figures from the committed three-seed PTQ sweep."""
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "keyword_spotting/results/ptq_sweep/17803125"
OUT = ROOT / "docs/images"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for hidden in (16, 32, 48, 64):
        summaries, sizes = [], []
        for seed in (0, 1, 2):
            folder = SWEEP / f"16_mels_delta_delta_h{hidden}_s{seed}"
            summaries.append(json.loads((folder / "summary.json").read_text()))
            sizes.append(json.loads((folder / "size_report.json").read_text()))
        assert all(s["hidden_size"] == hidden for s in summaries)
        assert [s["seed"] for s in summaries] == [0, 1, 2]
        fp = [s["fp32_test_acc"] for s in summaries]
        quant = [s["int8_test_acc"] for s in summaries]
        rows.append(dict(hidden=hidden, params=summaries[0]["params"],
                         fp=mean(fp), fp_std=stdev(fp), quant=mean(quant),
                         quant_std=stdev(quant), drop=mean(a-b for a,b in zip(fp, quant)),
                         fp_kib=sizes[0]["fp32_weight_kib"],
                         int_kib=sizes[0]["int8_weight_kib"]))
    table = ["| Hidden size | Parameters | FP32 test accuracy | INT8 simulation test accuracy | Weight storage FP32 → INT8 |",
             "|---:|---:|---:|---:|---:|"]
    for r in rows:
        table.append(f"| {r['hidden']} | {r['params']:,} | {r['fp']:.2f} ± {r['fp_std']:.2f}% | {r['quant']:.2f} ± {r['quant_std']:.2f}% | {r['fp_kib']:.2f} → {r['int_kib']:.2f} KiB |")
    text = "\n".join(table)
    readme = ROOT / "README.md"
    content = readme.read_text()
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    before, rest = content.split(start)
    _, after = rest.split(end)
    readme.write_text(before + start + "\n" + text + "\n" + end + after)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    for label, key, color in [("FP32", "fp", "#2563eb"), ("INT8 simulation", "quant", "#c2410c")]:
        ax.errorbar([r["hidden"] for r in rows], [r[key] for r in rows],
                    yerr=[r[key+"_std"] for r in rows], marker="o", capsize=4, label=label, color=color)
    ax.set(xlabel="GRU hidden size", ylabel="Test accuracy (%)", xticks=[16,32,48,64],
           title="Speech Commands v2 · mean ± sample SD, three seeds")
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(OUT / "accuracy.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    x = list(range(len(rows)))
    ax.bar([i-.18 for i in x], [r["fp_kib"] for r in rows], .36, label="FP32 weights", color="#2563eb")
    ax.bar([i+.18 for i in x], [r["int_kib"] for r in rows], .36, label="INT8 weights (estimate)", color="#c2410c")
    ax.set(xticks=x, xticklabels=[r["hidden"] for r in rows], xlabel="GRU hidden size",
           ylabel="Weight storage (KiB)", title="4× smaller weight representation · excludes biases and buffers")
    ax.legend(); fig.savefig(OUT / "weight-storage.png", dpi=170); plt.close(fig)
    print(text)
    print(f"GRU-64 mean accuracy drop: {rows[-1]['drop']:.4f} percentage points")


if __name__ == "__main__":
    main()
