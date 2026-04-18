import torch
import torch.nn as nn
import torchaudio

from utils import N_MELS, NUM_CLASSES, SAMPLE_RATE


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


class KeywordGRU(nn.Module):
    def __init__(
        self,
        n_mels=N_MELS,
        hidden_size=64,
        num_layers=2,
        num_classes=NUM_CLASSES,
        use_delta=True,
        use_delta_delta=True,
    ):
        super().__init__()
        self.mel = build_mel_spectrogram(n_mels=n_mels)
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.use_delta = use_delta
        self.use_delta_delta = use_delta_delta
        self.compute_deltas = torchaudio.transforms.ComputeDeltas()

        input_multiplier = 1
        if self.use_delta:
            input_multiplier += 1
        if self.use_delta and self.use_delta_delta:
            input_multiplier += 1

        self.gru = nn.GRU(
            input_size=n_mels * input_multiplier,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.mel(x)
        x = self.db(x)

        features = [x]
        if self.use_delta:
            deltas = self.compute_deltas(x)
            features.append(deltas)
            if self.use_delta_delta:
                ddeltas = self.compute_deltas(deltas)
                features.append(ddeltas)

        x = torch.cat(features, dim=2)

        std = x.std(dim=-1, keepdim=True).clamp(min=0.1)
        x = (x - x.mean(dim=-1, keepdim=True)) / std

        x = x.squeeze(1)
        x = x.permute(0, 2, 1)

        _, h_n = self.gru(x)
        last_hidden = h_n[-1]
        return self.classifier(last_hidden)
