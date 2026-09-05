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

N_MELS = 16 # OR 48 only mels
TARGET_LENGTH = 16000
SAMPLE_RATE = 16000


def compute_macs(n_mels, input_multiplier, hidden_size, num_layers, num_classes, n_frames):
    """Analytical MAC count for a stacked GRU + linear classifier head.

    Per timestep, each GRU layer needs 3 gates * (input_size * hidden + hidden * hidden) MACs.
    Layer 1 input_size = n_mels * input_multiplier; subsequent layers input_size = hidden_size.
    Only the final hidden state is fed to the classifier, so its MACs aren't multiplied by n_frames.
    """
    feat_dim = n_mels * input_multiplier
    macs = 0
    for layer_idx in range(num_layers):
        in_size = feat_dim if layer_idx == 0 else hidden_size
        macs += n_frames * 3 * hidden_size * (in_size + hidden_size)
    macs += hidden_size * num_classes
    return macs

NOISE_MIX_PROB = 0.5
NOISE_SNR_DB_RANGE = (0.0, 15.0)
GAIN_RANGE = (0.7, 1.3)
SILENCE_GAIN_RANGE = (0.0, 0.1)


def build_mel_spectrogram(n_mels=N_MELS):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        hop_length=160,
        win_length=480,
        n_mels=n_mels,
        f_min=0.0,
        f_max=8000.0,
    )


class FeatureExtractor:
    """Stateless mel+dB+delta extractor for use in dataloader workers (no autograd needed)."""

    def __init__(self, n_mels=N_MELS, use_delta=True, use_delta_delta=True):
        self.mel = build_mel_spectrogram(n_mels=n_mels)
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.use_delta = use_delta
        self.use_delta_delta = use_delta_delta
        self.compute_deltas = torchaudio.transforms.ComputeDeltas()

    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        x = self.mel(waveforms)
        x = self.db(x)
        features = [x]
        if self.use_delta:
            deltas = self.compute_deltas(x)
            features.append(deltas)
            if self.use_delta_delta:
                ddeltas = self.compute_deltas(deltas)
                features.append(ddeltas)
        x = torch.cat(features, dim=2)
        return x.squeeze(1)  # (B, 1, F, T) -> (B, F, T)


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


def _synthesize_silence_batch(num_silence, target_length):
    waveforms = []
    for _ in range(num_silence):
        silent = _random_noise_crop(target_length)
        silent = silent * random.uniform(*SILENCE_GAIN_RANGE)
        waveforms.append(silent)
    return waveforms


def collate_fn(batch, training=False, feature_extractor=None):
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

            gain = random.uniform(*GAIN_RANGE)
            waveform = waveform * gain

            if random.random() < NOISE_MIX_PROB:
                snr_db = random.uniform(*NOISE_SNR_DB_RANGE)
                waveform = _mix_noise(waveform, snr_db)

        mapped_label = LABEL_TO_IDX[label] if label in COMMANDS_10 else LABEL_TO_IDX["unknown"]
        waveforms.append(waveform)
        labels.append(mapped_label)

    num_silence = int(0.1 * len(waveforms))
    silence_waveforms = _synthesize_silence_batch(num_silence, TARGET_LENGTH)
    waveforms.extend(silence_waveforms)
    labels.extend([LABEL_TO_IDX["silence"]] * num_silence)

    stacked = torch.stack(waveforms)
    if feature_extractor is not None:
        stacked = feature_extractor(stacked)
    return stacked, torch.tensor(labels)


def make_train_collate(feature_extractor=None):
    def _collate(batch):
        return collate_fn(batch, training=True, feature_extractor=feature_extractor)
    return _collate


class BalancedUnderSampler(torch.utils.data.Sampler):
    """BC-ResNet style class-balanced under-sampler.

    Each epoch samples the same number of indices per class (size of the smallest class),
    drawn without replacement, then concatenates and shuffles the result.
    """

    def __init__(self, dataset):
        labels = [
            LABEL_TO_IDX[Path(p).parent.name] if Path(p).parent.name in COMMANDS_10
            else LABEL_TO_IDX["unknown"]
            for p in dataset._walker
        ]
        self._indices_by_class = {}
        for idx, label in enumerate(labels):
            self._indices_by_class.setdefault(label, []).append(idx)
        non_empty_counts = [len(v) for v in self._indices_by_class.values() if v]
        self._target_count = min(non_empty_counts)
        self._length = self._target_count * len(self._indices_by_class)

    def __iter__(self):
        sampled = []
        for class_indices in self._indices_by_class.values():
            sampled.extend(random.sample(class_indices, self._target_count))
        random.shuffle(sampled)
        return iter(sampled)

    def __len__(self):
        return self._length


def make_eval_collate(feature_extractor=None):
    def _collate(batch):
        return collate_fn(batch, training=False, feature_extractor=feature_extractor)
    return _collate


def build_datasets(data_root=None, download=True):
    root = str(data_root or get_data_root())
    if download:
        Path(root).mkdir(parents=True, exist_ok=True)

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


def build_dataloaders(
    batch_size_train=64,
    batch_size_eval=1024,
    pin_memory=True,
    balanced_sampler=False,
    n_mels=None,
    use_delta=None,
    use_delta_delta=None,
    seed=None,
):
    train_set, val_set, test_set = build_datasets()

    if n_mels is not None:
        feature_extractor = FeatureExtractor(
            n_mels=n_mels,
            use_delta=use_delta if use_delta is not None else True,
            use_delta_delta=use_delta_delta if use_delta_delta is not None else True,
        )
    else:
        feature_extractor = None

    if balanced_sampler:
        sampler = BalancedUnderSampler(train_set)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_generator = None
    if seed is not None and shuffle:
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size_train,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=True,
        shuffle=shuffle,
        sampler=sampler,
        generator=train_generator,
        collate_fn=make_train_collate(feature_extractor=feature_extractor),
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size_eval,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=True,
        shuffle=False,
        collate_fn=make_eval_collate(feature_extractor=feature_extractor),
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size_eval,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=True,
        shuffle=False,
        collate_fn=make_eval_collate(feature_extractor=feature_extractor),
    )

    return train_loader, val_loader, test_loader
