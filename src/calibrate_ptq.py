import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from model import KeywordGRU
from ptq.quant_model import QuantizedKeywordGRU, from_fp32
from ptq.quant_utils import collect_scales, freeze_quantizers, set_quant_mode
from utils import NUM_CLASSES, build_dataloaders, compute_macs


N_FRAMES = 101  # MelSpectrogram(center=True) on 16000 samples @ hop=160 -> floor(16000/160)+1


FEATURE_CONFIGS = {
    "16_mels": {"n_mels": 16, "use_delta": False, "use_delta_delta": False},
    "16_mels_delta_delta": {"n_mels": 16, "use_delta": True, "use_delta_delta": True},
    "48_mels": {"n_mels": 48, "use_delta": False, "use_delta_delta": False},
}


def evaluate(model: nn.Module, loader, device, split_name: str, max_batches: int = -1) -> float:
    model.eval()
    correct = 0
    total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            if max_batches >= 0 and i >= max_batches:
                break
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            preds = model(data).argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            print(f"  {split_name} batch {i+1}: running acc={100.0*correct/total:.2f}% ({correct}/{total})", flush=True)
    acc = 100.0 * correct / total
    dt = time.perf_counter() - t0
    print(f"{split_name}: acc={correct}/{total} ({acc:.2f}%)  time={dt:.1f}s", flush=True)
    return acc


def calibrate(model: nn.Module, loader, device, num_batches: int) -> None:
    model.eval()
    set_quant_mode(model, "observe")
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= num_batches:
                break
            data = data.to(device, non_blocking=True)
            model(data)
    freeze_quantizers(model)
    set_quant_mode(model, "quantize")


def parse_args():
    p = argparse.ArgumentParser(description="PTQ INT8 calibration + eval for KeywordGRU")
    p.add_argument("--feature-config", choices=list(FEATURE_CONFIGS.keys()), required=True)
    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to FP32 state_dict (.pt)")
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--batch-size-eval", type=int, default=256)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--percentile", type=float, default=99.99)
    p.add_argument("--max-eval-batches", type=int, default=-1, help="-1 = full eval; otherwise limit eval to N batches (sanity-check mode)")
    p.add_argument("--spec-augment", action="store_true", help="Build KeywordGRU with SpecAugment submodules (training-config parity; no-op at eval).")
    p.add_argument("--freq-mask-param", type=int, default=8)
    p.add_argument("--time-mask-param", type=int, default=30)
    p.add_argument("--dropout", type=float, default=0.0, help="Constructor dropout (no-op at eval).")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FEATURE_CONFIGS[args.feature_config]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Feature config: {args.feature_config} -> {cfg}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Checkpoint: {args.checkpoint}")

    fp32_model = KeywordGRU(
        n_mels=cfg["n_mels"],
        hidden_size=args.hidden_size,
        use_delta=cfg["use_delta"],
        use_delta_delta=cfg["use_delta_delta"],
        spec_augment=args.spec_augment,
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
        use_new_gru=True,
        dropout=args.dropout,
        precomputed_features=True,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    fp32_model.load_state_dict(state)
    fp32_model = fp32_model.to(device).eval()

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size_train=args.batch_size_eval,
        batch_size_eval=args.batch_size_eval,
        pin_memory=(device.type == "cuda"),
        n_mels=cfg["n_mels"],
        use_delta=cfg["use_delta"],
        use_delta_delta=cfg["use_delta_delta"],
        seed=args.seed,
    )

    fp32_acc = evaluate(fp32_model, test_loader, device, split_name="FP32 Test", max_batches=args.max_eval_batches)

    q_model = from_fp32(fp32_model).to(device).eval()

    for m in q_model.modules():
        if hasattr(m, "observer") and hasattr(m.observer, "percentile"):
            m.observer.percentile = args.percentile

    print(f"Calibrating on {args.calib_batches} val batches (percentile={args.percentile})...")
    t_calib = time.perf_counter()
    calibrate(q_model, val_loader, device, args.calib_batches)
    print(f"Calibration done in {time.perf_counter() - t_calib:.1f}s")

    int8_acc = evaluate(q_model, test_loader, device, split_name="INT8 Test", max_batches=args.max_eval_batches)
    delta = fp32_acc - int8_acc
    print(f"Accuracy delta (FP32 - INT8): {delta:.3f} pp")

    scales = collect_scales(q_model)
    scales_path = out_dir / "scales.json"
    with open(scales_path, "w") as f:
        json.dump(scales, f, indent=2, sort_keys=True)
    print(f"Wrote scales to {scales_path} ({len(scales)} entries)")

    from ptq.report_size import build_report, format_report
    size_report = build_report(state, scales)
    print(format_report(size_report))
    with open(out_dir / "size_report.json", "w") as f:
        json.dump(size_report, f, indent=2, sort_keys=True)

    input_multiplier = 1 + int(cfg["use_delta"]) + int(cfg["use_delta"] and cfg["use_delta_delta"])
    macs = compute_macs(
        n_mels=cfg["n_mels"],
        input_multiplier=input_multiplier,
        hidden_size=args.hidden_size,
        num_layers=2,
        num_classes=NUM_CLASSES,
        n_frames=N_FRAMES,
    )
    params = sum(p.numel() for p in fp32_model.parameters() if p.requires_grad)
    print(f"Params: {params:,}  MACs: {macs:,}")

    summary = {
        "feature_config": args.feature_config,
        "hidden_size": args.hidden_size,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "fp32_test_acc": fp32_acc,
        "int8_test_acc": int8_acc,
        "delta_pp": delta,
        "params": params,
        "macs": macs,
        "calib_batches": args.calib_batches,
        "percentile": args.percentile,
        "int8_total_kib": size_report["int8_total_kib"],
        "compression_total": size_report["compression_total"],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
