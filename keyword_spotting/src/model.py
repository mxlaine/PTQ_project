import torch
import torch.nn as nn
import torchaudio

from new_gru import NewGRU
from utils import N_MELS, NUM_CLASSES, build_mel_spectrogram


class KeywordGRU(nn.Module):
    def __init__(
        self,
        n_mels=N_MELS,
        hidden_size=64,
        num_layers=2,
        num_classes=NUM_CLASSES,
        use_delta=True,
        use_delta_delta=True,
        spec_augment=False,
        freq_mask_param=8,
        time_mask_param=30,
        use_new_gru=False,
        dropout=0.3,
        precomputed_features=False,
    ):
        super().__init__()
        self.precomputed_features = precomputed_features
        self.mel = build_mel_spectrogram(n_mels=n_mels)
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.use_delta = use_delta
        self.use_delta_delta = use_delta_delta
        self.compute_deltas = torchaudio.transforms.ComputeDeltas()
        self.apply_specaugment = spec_augment
        if self.apply_specaugment:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)

        input_multiplier = 1
        if self.use_delta:
            input_multiplier += 1
        if self.use_delta and self.use_delta_delta:
            input_multiplier += 1

        self.use_new_gru = use_new_gru
        gru_cls = NewGRU if use_new_gru else nn.GRU
        self.gru = gru_cls(
            input_size=n_mels * input_multiplier,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def _apply_spec_augment(self, x):
        x = self.freq_mask(x)
        x = self.time_mask(x)
        return x

    def forward(self, x):
        if self.precomputed_features:
            if x.dim() == 4:
                x = x.squeeze(1)
        else:
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
            x = x.squeeze(1)

        if self.training and self.apply_specaugment:
            x = self._apply_spec_augment(x)

        std = x.std(dim=-1, keepdim=True).clamp(min=0.1)
        x = (x - x.mean(dim=-1, keepdim=True)) / std

        x = x.permute(0, 2, 1)

        if self.use_new_gru:
            _, h_n = self.gru(x, return_sequences=False)
        else:
            _, h_n = self.gru(x)
        last_hidden = h_n[-1]
        return self.classifier(last_hidden)
