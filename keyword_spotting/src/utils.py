from pathlib import Path
import random
from functools import lru_cache

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import datasets


COMMANDS_10 = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = COMMANDS_10 + ["unknown", "silence"]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

N_MELS = 16
TARGET_LENGTH = 16000
SAMPLE_RATE = 16000

NOISE_MIX_PROB = 0.5
NOISE_SNR_DB_RANGE = (0.0, 15.0)
GAIN_RANGE = (0.7, 1.3)
SILENCE_GAIN_RANGE = (0.0, 0.1)


def get_data_root():
    return Path(__file__).resolve().parents[1] / "data"


@lru_cache(maxsize=1)
def load_background_noises():
    noise_dir = get_data_root() / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"
    clips = []
    for wav_path in sorted(noise_dir.glob("*.wav")):
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        clips.append(waveform)
    if not clips:
        raise RuntimeError(f"No background noise wavs found in {noise_dir}")
    return clips


def _random_noise_crop(length: int) -> torch.Tensor:
    clips = load_background_noises()
    clip = random.choice(clips)
    total = clip.shape[-1]
    if total < length:
        reps = (length // total) + 1
        clip = clip.repeat(1, reps)
        total = clip.shape[-1]
    start = random.randint(0, total - length)
    return clip[..., start:start + length].clone()


def _mix_noise(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    noise = _random_noise_crop(signal.shape[-1])
    sig_power = signal.pow(2).mean().clamp(min=1e-10)
    noise_power = noise.pow(2).mean().clamp(min=1e-10)
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power).sqrt()
    return signal + scale * noise


# Discrete rates chosen so that new_freq = round(16000 * rate) has GCD >= 800 with
# 16000, keeping the sinc filter length under ~300 taps.  Continuous uniform sampling
# can produce GCD=2 and filter lengths > 96,000, which makes each batch take seconds.
_SPEED_RATES = (0.9, 0.95, 1.05, 1.1)


@lru_cache(maxsize=8)
def _get_speed_resampler(new_freq: int) -> torchaudio.transforms.Resample:
    return torchaudio.transforms.Resample(SAMPLE_RATE, new_freq)


def _random_speed_perturb(waveform: torch.Tensor) -> torch.Tensor:
    rate = random.choice(_SPEED_RATES)
    new_freq = int(round(SAMPLE_RATE * rate))
    waveform = _get_speed_resampler(new_freq)(waveform)
    if waveform.shape[-1] < TARGET_LENGTH:
        waveform = F.pad(waveform, (0, TARGET_LENGTH - waveform.shape[-1]))
    else:
        waveform = waveform[..., :TARGET_LENGTH]
    return waveform


def collate_fn(batch, training=False, speed_perturb=False):
    waveforms, labels = [], []

    for waveform, sample_rate, label, *_ in batch:
        del sample_rate

        if waveform.shape[-1] < TARGET_LENGTH:
            waveform = F.pad(waveform, (0, TARGET_LENGTH - waveform.shape[-1]))
        else:
            waveform = waveform[..., :TARGET_LENGTH]

        if training:
            if speed_perturb:
                waveform = _random_speed_perturb(waveform)

            shift = random.randint(-800, 800)
            if shift > 0:
                waveform = F.pad(waveform[..., :-shift], (shift, 0))
            elif shift < 0:
                waveform = F.pad(waveform[..., -shift:], (0, -shift))

            gain = random.uniform(*GAIN_RANGE)
            waveform = waveform * gain

            if random.random() < NOISE_MIX_PROB:
                snr_db = random.uniform(*NOISE_SNR_DB_RANGE)
                waveform = _mix_noise(waveform, snr_db)

        mapped_label = LABEL_TO_IDX[label] if label in COMMANDS_10 else LABEL_TO_IDX["unknown"]
        waveforms.append(waveform)
        labels.append(mapped_label)

    num_silence = int(0.1 * len(waveforms))
    for _ in range(num_silence):
        silent = _random_noise_crop(TARGET_LENGTH)
        silent = silent * random.uniform(*SILENCE_GAIN_RANGE)
        waveforms.append(silent)
        labels.append(LABEL_TO_IDX["silence"])

    return torch.stack(waveforms), torch.tensor(labels)


def make_train_collate(speed_perturb=False):
    def _collate(batch):
        return collate_fn(batch, training=True, speed_perturb=speed_perturb)
    return _collate


def make_sample_weights(dataset) -> torch.Tensor:
    """Return a per-sample weight tensor (inverse class frequency) without loading audio.

    Uses dataset._walker (list of file paths); label is the parent directory name.
    Silence is injected synthetically in collate_fn and not present here — that's fine.
    """
    labels = [
        LABEL_TO_IDX[Path(p).parent.name] if Path(p).parent.name in COMMANDS_10
        else LABEL_TO_IDX["unknown"]
        for p in dataset._walker
    ]
    label_tensor = torch.tensor(labels)
    counts = torch.bincount(label_tensor, minlength=NUM_CLASSES).float()
    weights = 1.0 / counts.clamp(min=1)
    return weights[label_tensor]


def train_collate(batch):
    return collate_fn(batch, training=True)


def eval_collate(batch):
    return collate_fn(batch, training=False)


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


def build_dataloaders(batch_size_train=64, batch_size_eval=1024, num_workers=4, pin_memory=True, speed_perturb=False, balanced_sampler=False):
    train_set, val_set, test_set = build_datasets()

    if balanced_sampler:
        sample_weights = make_sample_weights(train_set)
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=make_train_collate(speed_perturb=speed_perturb),
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
