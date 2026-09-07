from pathlib import Path
import warnings
import importlib
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from model import KeywordGRU
from utils import LABEL_TO_IDX, NUM_CLASSES, build_dataloaders, compute_macs
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR


warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step\\(\\)`", category=UserWarning)


FEATURE_CONFIGS = {
    "16_mels": {"n_mels": 16, "use_delta": False, "use_delta_delta": False},
    "16_mels_delta_delta": {"n_mels": 16, "use_delta": True, "use_delta_delta": True},
    "48_mels": {"n_mels": 48, "use_delta": False, "use_delta_delta": False},
}


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_cuda = device.type == 'cuda'
    time_data_loading = 0.0
    time_forward = 0.0
    time_backward = 0.0

    if use_cuda:
        torch.cuda.synchronize()
    epoch_start_time = time.perf_counter()
    batch_start_time = epoch_start_time

    for batch_idx, (data, target) in enumerate(loader):
        time_data_loading += time.perf_counter() - batch_start_time

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if use_cuda:
            torch.cuda.synchronize()
        forward_start_time = time.perf_counter()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if use_cuda:
            torch.cuda.synchronize()
        backward_start_time = time.perf_counter()
        time_forward += backward_start_time - forward_start_time

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if use_cuda:
            torch.cuda.synchronize()
        time_backward += time.perf_counter() - backward_start_time

        total_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

        if batch_idx % 500 == 0 and epoch is not None:
            print(
                f"Epoch {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

        batch_start_time = time.perf_counter()

    train_time = time.perf_counter() - epoch_start_time
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    if epoch is not None:
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.6f} | Train Acc: {accuracy:.2f}%")

    return accuracy, train_time, time_data_loading, time_forward, time_backward, total


def evaluate(model, loader, criterion, device, split_name="Val"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    use_cuda = device.type == 'cuda'
    if use_cuda:
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * target.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    if use_cuda:
        torch.cuda.synchronize()
    eval_time = time.perf_counter() - t_start

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    print(f"{split_name}: loss={avg_loss:.4f}, acc={correct}/{total} ({acc:.2f}%)")
    return acc, avg_loss, eval_time


def parse_args():
    parser = argparse.ArgumentParser(description="Train KeywordGRU on Speech Commands")
    parser.add_argument("--epochs", type=int, default=325)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model init and sampler shuffle")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-eval", type=int, default=1024)
    parser.add_argument("--spec-augment", action="store_true", help="Enable SpecAugment (freq/time masking) during training")
    parser.add_argument("--freq-mask-param", type=int, default=8, help="Frequency mask param for SpecAugment")
    parser.add_argument("--time-mask-param", type=int, default=30, help="Time mask param for SpecAugment")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing for CrossEntropyLoss")
    parser.add_argument(
        "--feature-config",
        choices=list(FEATURE_CONFIGS.keys()),
        default="16_mels_delta_delta",
        help="Input feature setup",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use-new-gru", action="store_true", help="Use the from-scratch NewGRU instead of nn.GRU")
    parser.add_argument("--lr-warmup-epochs", type=int, default=0, help="Linear LR warmup epochs before cosine decay")
    parser.add_argument("--lr-scheduler", choices=["cosine", "cosine-warm-restarts"], default="cosine")
    parser.add_argument("--lr-t0", type=int, default=200, help="T_0 (epochs per first cycle) for cosine-warm-restarts")
    parser.add_argument("--lr-t-mult", type=int, default=1, help="T_mult (cycle length multiplier) for cosine-warm-restarts")
    parser.add_argument("--lr-restart-ceiling-decay", type=float, default=1.0, help="Multiplier applied to base_lrs at each warm-restart cycle boundary (1.0 = no decay)")
    parser.add_argument("--balanced-sampler", action="store_true", help="BC-ResNet style under-sampler that balances classes per epoch")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_config = FEATURE_CONFIGS[args.feature_config]

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size_train=args.batch_size_train,
        batch_size_eval=args.batch_size_eval,
        pin_memory=(device.type == "cuda"),
        balanced_sampler=args.balanced_sampler,
        n_mels=feature_config["n_mels"],
        use_delta=feature_config["use_delta"],
        use_delta_delta=feature_config["use_delta_delta"],
        seed=args.seed,
    )

    model = KeywordGRU(
        n_mels=feature_config["n_mels"],
        hidden_size=args.hidden_size,
        use_delta=feature_config["use_delta"],
        use_delta_delta=feature_config["use_delta_delta"],
        spec_augment=args.spec_augment,
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
        use_new_gru=args.use_new_gru,
        dropout=args.dropout,
        precomputed_features=True,
    ).to(device)

    epochs = args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = min(args.lr_warmup_epochs, epochs - 1)

    if args.lr_scheduler == "cosine-warm-restarts":
        main_sched = CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_t0, T_mult=args.lr_t_mult)
    else:
        main_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs if warmup_epochs > 0 else epochs)

    if warmup_epochs > 0:
        warmup_sched = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs])
    else:
        scheduler = main_sched

    wr_sched = main_sched if args.lr_scheduler == "cosine-warm-restarts" else None

    # When the under-sampler is on, classes are already balanced per epoch — adding
    # class weights on top would double-count. Apply weights only without the sampler.
    class_weights = torch.ones(NUM_CLASSES, device=device)
    if not args.balanced_sampler:
        class_weights[LABEL_TO_IDX["unknown"]] = 1.20
        class_weights[LABEL_TO_IDX["silence"]] = 1.10

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Feature config: {args.feature_config} -> {feature_config}")
    print(f"SpecAugment: {args.spec_augment} freq_mask={args.freq_mask_param} time_mask={args.time_mask_param}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Epochs: {epochs}")
    print(f"LR: {args.lr}")
    print(f"LR warmup epochs: {warmup_epochs}")
    if args.lr_scheduler == "cosine-warm-restarts":
        print(f"LR scheduler: cosine-warm-restarts T0={args.lr_t0} T_mult={args.lr_t_mult} ceiling_decay={args.lr_restart_ceiling_decay}")
    else:
        print(f"LR scheduler: cosine")
    print(f"Balanced sampler: {args.balanced_sampler}")
    print(f"Dropout: {args.dropout}")
    print(f"New GRU: {args.use_new_gru}")
    print(f"CPU count: {os.cpu_count()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    from utils import FeatureExtractor, TARGET_LENGTH
    _probe = FeatureExtractor(
        n_mels=feature_config["n_mels"],
        use_delta=feature_config["use_delta"],
        use_delta_delta=feature_config["use_delta_delta"],
    )
    _dummy = torch.zeros(1, 1, TARGET_LENGTH)
    n_frames = _probe(_dummy).shape[-1]
    input_multiplier = 1 + int(feature_config["use_delta"]) + int(feature_config["use_delta"] and feature_config["use_delta_delta"])
    total_macs = compute_macs(
        n_mels=feature_config["n_mels"],
        input_multiplier=input_multiplier,
        hidden_size=args.hidden_size,
        num_layers=2,
        num_classes=NUM_CLASSES,
        n_frames=n_frames,
    )
    print(f"Total MACs: {total_macs:,}")

    best_val = -1.0
    train_acc_history = []
    val_acc_history = []

    # Plots are grouped by job and task similar to .err/.out naming in slurm.
    # Allow SLURM or explicit PLOT_* env vars set by the submit script.
    job_id = os.environ.get("PLOT_JOB") or os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    task_id = os.environ.get("PLOT_TASK") or os.environ.get("SLURM_ARRAY_TASK_ID") or "0"

    artifact_dir = Path(__file__).resolve().parents[1] / "models" / job_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = artifact_dir / f"best_keyword_gru_{args.feature_config}_{job_id}_{task_id}.pt"

    plots_dir = Path(__file__).resolve().parents[1] / "plots" / f"{job_id}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"train_val_{args.feature_config}_{job_id}_{task_id}.png"
    log_plot_path = plots_dir / f"train_val_{args.feature_config}_{job_id}_{task_id}_log.png"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_acc, train_time, time_data_loading, time_forward, time_backward, train_samples = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch=epoch,
        )
        val_acc, _, eval_time = evaluate(model, val_loader, criterion, device, split_name="Val")

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"saved best: {best_val:.2f}%")

        prev_t_cur = wr_sched.T_cur if wr_sched is not None else None
        scheduler.step()
        if wr_sched is not None and prev_t_cur is not None and wr_sched.T_cur < prev_t_cur:
            wr_sched.base_lrs = [lr * args.lr_restart_ceiling_decay for lr in wr_sched.base_lrs]
            for pg, base_lr in zip(optimizer.param_groups, wr_sched.base_lrs):
                pg['lr'] = base_lr
            print(f"cycle restart at epoch {epoch}: new base_lrs={wr_sched.base_lrs}")
        epoch_total = time.perf_counter() - epoch_start
        samples_per_sec = train_samples / train_time if train_time > 0 else 0.0
        print(
            f"Epoch {epoch} timing: total={epoch_total:.1f}s train={train_time:.1f}s eval={eval_time:.1f}s "
            f"| data_wait={time_data_loading:.1f}s fwd={time_forward:.1f}s bwd={time_backward:.1f}s ({samples_per_sec:.0f} samples/s)"
        )

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_acc, _, _ = evaluate(model, test_loader, criterion, device, split_name="Test")

    pyplot = importlib.import_module("matplotlib.pyplot")

    line1 = [
        f"feature={args.feature_config}",
        f"n_mels={feature_config['n_mels']}",
        f"use_delta={feature_config['use_delta']}",
        f"use_delta_delta={feature_config['use_delta_delta']}",
    ]
    if args.spec_augment:
        line1.append(f"SpecAugment(freq={args.freq_mask_param},time={args.time_mask_param})")
    if args.balanced_sampler:
        line1.append("balanced_sampler=True")

    line2 = [
        f"hidden={args.hidden_size}",
        f"dropout={args.dropout}",
        f"lr={args.lr}",
        f"wd={args.weight_decay}",
        f"label_smooth={args.label_smoothing}",
        f"warmup={warmup_epochs}",
        f"scheduler={args.lr_scheduler}" + (f"(T0={args.lr_t0},Tm={args.lr_t_mult})" if args.lr_scheduler == "cosine-warm-restarts" else ""),
    ]
    meta_txt = " | ".join(line1) + "\n" + " | ".join(line2)

    epochs_ran = list(range(1, len(val_acc_history) + 1))

    def _draw(ax, train_y, val_y, test_y, test_label):
        # Curves
        ax.plot(epochs_ran, train_y, marker="^", markersize=1, linewidth=1.0,
                label="Training", alpha=0.8)
        ax.plot(epochs_ran, val_y, marker="o", markersize=1, linewidth=1.0,
                label="Validation", alpha=0.8)
        # Test marker: a short dash at the right edge + label outside the axes,
        # so it never obstructs the training/validation curves.
        ax.axhline(y=test_y, xmin=0.97, xmax=1.0, color="r", linewidth=2.0)
        ax.annotate(
            test_label,
            xy=(1.0, test_y), xycoords=("axes fraction", "data"),
            xytext=(5, 0), textcoords="offset points",
            va="center", ha="left", fontsize=8, color="r",
            annotation_clip=False,
        )
        ax.legend(loc="best")

    # Linear accuracy plot
    fig, ax = pyplot.subplots(figsize=(10, 5))
    _draw(ax, train_acc_history, val_acc_history, test_acc, f"Test {test_acc:.2f}%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Training and Validation Accuracy ({job_id}_{task_id})")
    ax.grid(True, alpha=0.3)
    fig.suptitle(meta_txt, fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    try:
        fig.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"Failed to save plot: {e}")
    pyplot.close(fig)

    # Log-scale error-rate plot: error = 100 - accuracy on a log y-axis, which
    # spreads out near-ceiling differences for comparison with SotA KWS methods
    # (e.g. BC-ResNet). EPS keeps perfect-accuracy points loggable.
    EPS = 0.05
    train_err = [max(100.0 - a, EPS) for a in train_acc_history]
    val_err = [max(100.0 - a, EPS) for a in val_acc_history]
    test_err = max(100.0 - test_acc, EPS)
    fig, ax = pyplot.subplots(figsize=(10, 5))
    _draw(ax, train_err, val_err, test_err, f"Test {test_err:.2f}%")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error rate (%)")
    ax.set_title(f"Training and Validation Error — log scale ({job_id}_{task_id})")
    ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(meta_txt, fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    try:
        fig.savefig(log_plot_path)
        print(f"Saved plot to {log_plot_path}")
    except Exception as e:
        print(f"Failed to save plot: {e}")
    pyplot.close(fig)


if __name__ == "__main__":
    main()
