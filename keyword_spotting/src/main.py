from pathlib import Path
import warnings
import importlib
import argparse

import torch
import torch.nn as nn

from model import KeywordGRU
from utils import LABEL_TO_IDX, NUM_CLASSES, build_dataloaders


warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


FEATURE_CONFIGS = {
    "16_mels_delta_delta": {"n_mels": 16, "use_delta": True, "use_delta_delta": True},
    "24_mels_delta": {"n_mels": 24, "use_delta": True, "use_delta_delta": False},
    "40_mels": {"n_mels": 40, "use_delta": False, "use_delta_delta": False},
    "48_mels": {"n_mels": 48, "use_delta": False, "use_delta_delta": False},
}


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

        if batch_idx % 500 == 0 and epoch is not None:
            print(
                f"Epoch {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    if epoch is not None:
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.6f} | Train Acc: {accuracy:.2f}%")

    return accuracy


def evaluate(model, loader, criterion, device, split_name="Val"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * target.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    print(f"{split_name}: loss={avg_loss:.4f}, acc={correct}/{total} ({acc:.2f}%)")
    return acc, avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train KeywordGRU on Speech Commands")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-eval", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--feature-config",
        choices=list(FEATURE_CONFIGS.keys()),
        default="16_mels_delta_delta",
        help="Input feature setup",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size_train=args.batch_size_train,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_config = FEATURE_CONFIGS[args.feature_config]
    model = KeywordGRU(
        n_mels=feature_config["n_mels"],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        use_delta=feature_config["use_delta"],
        use_delta_delta=feature_config["use_delta_delta"],
    ).to(device)

    epochs = args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs + 1,
        pct_start=0.3,
    )

    class_weights = torch.ones(NUM_CLASSES, device=device)
    class_weights[LABEL_TO_IDX["unknown"]] = 1.20
    class_weights[LABEL_TO_IDX["silence"]] = 1.10

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    print(f"Device: {device}")
    print(f"Feature config: {args.feature_config} -> {feature_config}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    best_val = -1.0
    train_acc_history = []
    val_acc_history = []

    artifact_dir = Path(__file__).resolve().parents[1] / "notebooks"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = artifact_dir / f"best_keyword_gru_{args.feature_config}.pt"

    for epoch in range(1, epochs + 1):
        train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch=epoch,
        )
        val_acc, _ = evaluate(model, val_loader, criterion, device, split_name="Val")

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"saved best: {best_val:.2f}%")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_acc, _ = evaluate(model, test_loader, criterion, device, split_name="Test")

    pyplot = importlib.import_module("matplotlib.pyplot")

    epochs_ran = range(1, len(val_acc_history) + 1)
    pyplot.figure(figsize=(8, 5))
    pyplot.plot(epochs_ran, train_acc_history, marker="^", linewidth=2, label="Training Accuracy")
    pyplot.plot(epochs_ran, val_acc_history, marker="o", linewidth=2, label="Validation Accuracy")
    pyplot.axhline(y=test_acc, color="r", linestyle="--", label="Final Test Accuracy")
    pyplot.annotate(
        f"Test: {test_acc:.2f}%",
        (len(val_acc_history), test_acc),
        textcoords="offset points",
        xytext=(10, 6),
    )
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Accuracy (%)")
    pyplot.title(f"Training and Validation Accuracy ({args.feature_config})")
    pyplot.grid(True, alpha=0.3)
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    main()
