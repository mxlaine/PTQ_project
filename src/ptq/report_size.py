import argparse
import json
from pathlib import Path

import torch


WEIGHT_KEY_HINTS = ("weight_ih", "weight_hh", "classifier.weight")
BIAS_KEY_HINTS = ("bias_ih", "bias_hh", "classifier.bias")


def classify_param(name: str) -> str:
    if any(h in name for h in WEIGHT_KEY_HINTS):
        return "weight"
    if any(h in name for h in BIAS_KEY_HINTS):
        return "bias"
    return "other"


def build_report(state_dict, scales, *, keep_bias_fp32=False, n_frames=97) -> dict:
    fp32_weight_bytes = 0
    fp32_bias_bytes = 0
    fp32_other_bytes = 0
    n_params = 0
    by_layer = {}

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or not torch.is_floating_point(tensor):
            continue
        bytes_fp32 = tensor.numel() * 4
        kind = classify_param(name)
        if kind == "weight":
            fp32_weight_bytes += bytes_fp32
        elif kind == "bias":
            fp32_bias_bytes += bytes_fp32
        else:
            fp32_other_bytes += bytes_fp32
        n_params += tensor.numel()
        by_layer[name] = {"kind": kind, "n_params": tensor.numel(), "fp32_bytes": bytes_fp32}

    bias_bytes_int8 = fp32_bias_bytes if keep_bias_fp32 else fp32_bias_bytes // 4
    int8_weight_bytes = fp32_weight_bytes // 4
    scales_bytes = len(scales) * 4
    int8_total_bytes = int8_weight_bytes + bias_bytes_int8 + scales_bytes
    fp32_total_bytes = fp32_weight_bytes + fp32_bias_bytes + fp32_other_bytes

    fp32_bw = fp32_weight_bytes + fp32_bias_bytes
    int8_bw = int8_weight_bytes + bias_bytes_int8

    return {
        "n_params": n_params,
        "fp32_weight_kib": fp32_weight_bytes / 1024,
        "fp32_bias_kib": fp32_bias_bytes / 1024,
        "fp32_other_kib": fp32_other_bytes / 1024,
        "fp32_total_kib": fp32_total_bytes / 1024,
        "int8_weight_kib": int8_weight_bytes / 1024,
        "bias_kib": bias_bytes_int8 / 1024,
        "keep_bias_fp32": keep_bias_fp32,
        "n_scales": len(scales),
        "scales_kib": scales_bytes / 1024,
        "int8_total_kib": int8_total_bytes / 1024,
        "compression_weights": fp32_weight_bytes / int8_weight_bytes,
        "compression_total": fp32_total_bytes / int8_total_bytes,
        "fp32_bw_kib": fp32_bw / 1024,
        "int8_bw_kib": int8_bw / 1024,
        "bw_reduction": fp32_bw / int8_bw,
        "by_layer": by_layer,
    }


def format_report(report: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("Static model footprint")
    lines.append("=" * 60)
    lines.append(f"Parameters total           : {report['n_params']:,}")
    lines.append(f"FP32 weights               : {report['fp32_weight_kib']:8.2f} KiB")
    lines.append(f"FP32 biases                : {report['fp32_bias_kib']:8.2f} KiB")
    if report["fp32_other_kib"]:
        lines.append(f"FP32 other (buffers etc.)  : {report['fp32_other_kib']:8.2f} KiB")
    lines.append(f"FP32 TOTAL                 : {report['fp32_total_kib']:8.2f} KiB")
    lines.append("")
    lines.append(f"INT8 weights               : {report['int8_weight_kib']:8.2f} KiB")
    lines.append(f"Biases ({'FP32' if report['keep_bias_fp32'] else 'INT8'} kept)         : {report['bias_kib']:8.2f} KiB")
    lines.append(f"Scales metadata ({report['n_scales']} entries): {report['scales_kib']:8.2f} KiB")
    lines.append(f"INT8 TOTAL                 : {report['int8_total_kib']:8.2f} KiB")
    lines.append("")
    lines.append(f"Compression (weights only) : {report['compression_weights']:6.2f}x")
    lines.append(f"Compression (total)        : {report['compression_total']:6.2f}x")
    lines.append("")
    lines.append("=" * 60)
    lines.append("Memory bandwidth per single inference (analytical)")
    lines.append("=" * 60)
    lines.append(f"Weight+bias bytes loaded (FP32) : {report['fp32_bw_kib']:8.2f} KiB / inference")
    lines.append(f"Weight+bias bytes loaded (INT8) : {report['int8_bw_kib']:8.2f} KiB / inference")
    lines.append(f"Bandwidth reduction             : {report['bw_reduction']:6.2f}x")
    lines.append("")
    lines.append("=" * 60)
    lines.append("Layer-by-layer parameter breakdown")
    lines.append("=" * 60)
    for name, info in sorted(report["by_layer"].items()):
        lines.append(
            f"  {name:50s}  kind={info['kind']:6s}  n={info['n_params']:>8,}  fp32={info['fp32_bytes']/1024:7.2f} KiB"
        )
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Report FP32 vs INT8 size & bandwidth for a quantized KeywordGRU.")
    p.add_argument("--checkpoint", type=str, required=True, help="FP32 state_dict (.pt).")
    p.add_argument("--scales", type=str, required=True, help="scales.json produced by calibrate_ptq.py.")
    p.add_argument("--keep-bias-fp32", action="store_true", help="Count biases as FP32 (matches a common FPGA accumulator design).")
    p.add_argument("--n-frames", type=int, default=97, help="Sequence length for the activation-bandwidth estimate.")
    return p.parse_args()


def main():
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")
    with open(args.scales) as f:
        scales = json.load(f)
    report = build_report(state, scales, keep_bias_fp32=args.keep_bias_fp32, n_frames=args.n_frames)
    print(format_report(report))


if __name__ == "__main__":
    main()
