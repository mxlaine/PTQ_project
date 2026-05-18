import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from model import KeywordGRU
from new_gru import NewGRU
from .quant_new_gru import QuantizedNewGRU, from_new_gru
from .quant_utils import Quantizer
from utils import N_MELS, NUM_CLASSES, build_mel_spectrogram


class QuantizedKeywordGRU(nn.Module):
    def __init__(
        self,
        n_mels: int = N_MELS,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        use_delta: bool = True,
        use_delta_delta: bool = True,
        dropout: float = 0.0,
        precomputed_features: bool = True,
    ):
        super().__init__()
        self.precomputed_features = precomputed_features
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

        self.gru = QuantizedNewGRU(
            input_size=n_mels * input_multiplier,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier_w_quant = Quantizer(observer_type="max")
        self.classifier_in_quant = Quantizer(observer_type="percentile")
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        std = x.std(dim=-1, keepdim=True).clamp(min=0.1)
        x = (x - x.mean(dim=-1, keepdim=True)) / std

        x = x.permute(0, 2, 1)

        _, h_n = self.gru(x, return_sequences=False)
        last_hidden = h_n[-1]

        last_hidden = self.classifier_in_quant(last_hidden)
        w_q = self.classifier_w_quant(self.classifier.weight)
        return F.linear(last_hidden, w_q, self.classifier.bias)


def from_fp32(fp32_model: KeywordGRU) -> QuantizedKeywordGRU:
    if not fp32_model.use_new_gru:
        raise ValueError("from_fp32 expects a KeywordGRU built with use_new_gru=True")

    src_gru: NewGRU = fp32_model.gru
    n_mels = src_gru.input_size // (1 + int(fp32_model.use_delta) + int(fp32_model.use_delta and fp32_model.use_delta_delta))

    q_model = QuantizedKeywordGRU(
        n_mels=n_mels,
        hidden_size=src_gru.hidden_size,
        num_layers=src_gru.num_layers,
        num_classes=fp32_model.classifier.out_features,
        use_delta=fp32_model.use_delta,
        use_delta_delta=fp32_model.use_delta_delta,
        dropout=0.0,
        precomputed_features=fp32_model.precomputed_features,
    )

    q_model.gru = from_new_gru(src_gru)

    with torch.no_grad():
        q_model.classifier.weight.copy_(fp32_model.classifier.weight)
        q_model.classifier.bias.copy_(fp32_model.classifier.bias)

    return q_model
