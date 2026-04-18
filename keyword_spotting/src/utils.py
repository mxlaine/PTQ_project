from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torchaudio import datasets


COMMANDS_10 = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = COMMANDS_10 + ["unknown", "silence"]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

N_MELS = 16
TARGET_LENGTH = 16000
SAMPLE_RATE = 16000


def collate_fn(batch, training=False):
    waveforms, labels = [], []

    for waveform, sample_rate, label, *_ in batch:
        del sample_rate

        if waveform.shape[-1] < TARGET_LENGTH:
            waveform = F.pad(waveform, (0, TARGET_LENGTH - waveform.shape[-1]))
        else:
            waveform = waveform[..., :TARGET_LENGTH]

        if training:
            shift = random.randint(-800, 800)
            if shift > 0:
                waveform = F.pad(waveform[..., :-shift], (shift, 0))
            elif shift < 0:
                waveform = F.pad(waveform[..., -shift:], (0, -shift))

            noise_amp = random.uniform(0.0, 0.005)
            waveform = waveform + noise_amp * torch.randn_like(waveform)

        mapped_label = LABEL_TO_IDX[label] if label in COMMANDS_10 else LABEL_TO_IDX["unknown"]
        waveforms.append(waveform)
        labels.append(mapped_label)

    num_silence = int(0.1 * len(waveforms))
    for _ in range(num_silence):
        noise_amp = random.uniform(0.0, 0.002)
        silent = torch.randn(1, TARGET_LENGTH) * noise_amp
        waveforms.append(silent)
        labels.append(LABEL_TO_IDX["silence"])

    return torch.stack(waveforms), torch.tensor(labels)


def train_collate(batch):
    return collate_fn(batch, training=True)


def eval_collate(batch):
    return collate_fn(batch, training=False)


def get_data_root():
    return Path(__file__).resolve().parents[1] / "data"


def build_datasets(data_root=None, download=True):
    root = str(data_root or get_data_root())

    train_set = datasets.SPEECHCOMMANDS(
        root=root,
        subset="training",
        download=download,
        url="speech_commands_v0.02",
    )
    val_set = datasets.SPEECHCOMMANDS(
        root=root,
        subset="validation",
        download=download,
        url="speech_commands_v0.02",
    )
    test_set = datasets.SPEECHCOMMANDS(
        root=root,
        subset="testing",
        download=download,
        url="speech_commands_v0.02",
    )

    return train_set, val_set, test_set


def build_dataloaders(batch_size_train=64, batch_size_eval=1024, num_workers=4, pin_memory=True):
    train_set, val_set, test_set = build_datasets()

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        shuffle=True,
        collate_fn=train_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size_eval,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        shuffle=False,
        collate_fn=eval_collate,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size_eval,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        shuffle=False,
        collate_fn=eval_collate,
    )

    return train_loader, val_loader, test_loader
