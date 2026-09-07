"""Verify bundled checkpoints and rerun calibration for the README sweep."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checkpoints/README-sweep/manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, choices=(16, 32, 48, 64), default=64)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--all", action="store_true", help="Run all 12 published configurations")
    parser.add_argument("--verify-only", action="store_true", help="Check hashes without PyTorch or data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/reproduced",
                        help="New evaluation output; archived results are kept separately")
    args = parser.parse_args()
    entries = json.loads(MANIFEST.read_text())
    selected = [e for e in entries if args.all or
                (e["hidden_size"], e["seed"]) == (args.hidden_size, args.seed)]
    # Verify every selected file before starting an expensive evaluation.
    for entry in selected:
        checkpoint = ROOT / entry["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
        print(f"Verified h{entry['hidden_size']} seed {entry['seed']}: {checkpoint.name}", flush=True)
    if args.verify_only:
        return
    out_dir = args.out_dir.resolve()
    archive = (ROOT / "results/ptq_sweep").resolve()
    if out_dir == archive or archive in out_dir.parents or out_dir in archive.parents:
        parser.error("--out-dir must be separate from the archived PTQ sweep")
    for entry in selected:
        run = f"{entry['feature_config']}_h{entry['hidden_size']}_s{entry['seed']}"
        subprocess.run([
            sys.executable, str(ROOT / "src/calibrate_ptq.py"),
            "--checkpoint", str(ROOT / entry["checkpoint"]),
            "--feature-config", entry["feature_config"],
            "--hidden-size", str(entry["hidden_size"]), "--seed", str(entry["seed"]),
            "--calib-batches", str(entry["calib_batches"]),
            "--percentile", str(entry["percentile"]),
            "--batch-size-eval", str(entry["batch_size_eval"]),
            "--spec-augment", "--freq-mask-param", "5", "--time-mask-param", "0",
            "--dropout", "0.1", "--out-dir", str(out_dir / run),
        ], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
