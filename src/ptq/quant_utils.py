"""Symmetric per-tensor INT8 fake-quantization primitives for PTQ."""

import torch
import torch.nn as nn

INT8_QMIN = -127
INT8_QMAX = 127


def fake_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if scale.item() <= 0.0:
        return x
    return torch.round(x / scale).clamp(INT8_QMIN, INT8_QMAX) * scale


class MaxObserver(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("max_abs", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.detach().abs().max()
        if m.item() > self.max_abs.item():
            self.max_abs.copy_(m)
        return x

    def compute_scale(self) -> torch.Tensor:
        return self.max_abs.detach().clone() / INT8_QMAX


class PercentileObserver(nn.Module):
    def __init__(self, num_bins: int = 2048, percentile: float = 99.99):
        super().__init__()
        self.num_bins = num_bins
        self.percentile = percentile
        self.register_buffer("histogram", torch.zeros(num_bins, dtype=torch.float64))
        self.register_buffer("upper", torch.tensor(0.0))

    def _rebin_to_new_upper(self, new_upper: float) -> None:
        old_upper = self.upper.item()
        if old_upper <= 0.0:
            return
        ratio = old_upper / new_upper
        idxs = torch.arange(self.num_bins, dtype=torch.long, device=self.histogram.device)
        new_idxs = (idxs.double() * ratio).long().clamp(max=self.num_bins - 1)
        new_hist = torch.zeros_like(self.histogram)
        new_hist.scatter_add_(0, new_idxs, self.histogram)
        self.histogram.copy_(new_hist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = x.detach().abs().flatten().float()
        cur_max = abs_x.max().item()
        if cur_max <= 0.0:
            return x
        if cur_max > self.upper.item():
            self._rebin_to_new_upper(cur_max)
            self.upper.fill_(cur_max)
        hist = torch.histc(abs_x.cpu(), bins=self.num_bins, min=0.0, max=self.upper.item())
        self.histogram.add_(hist.double().to(self.histogram.device))
        return x

    def compute_scale(self) -> torch.Tensor:
        upper = self.upper.item()
        if upper <= 0.0:
            return torch.tensor(0.0)
        cum = self.histogram.cumsum(0)
        total = cum[-1].item()
        if total <= 0.0:
            return torch.tensor(0.0)
        target = total * (self.percentile / 100.0)
        target_t = torch.tensor(target, dtype=cum.dtype, device=cum.device)
        idx = int(torch.searchsorted(cum, target_t).item())
        idx = min(idx, self.num_bins - 1)
        bin_upper_edge = (idx + 1) * (upper / self.num_bins)
        return torch.tensor(bin_upper_edge / INT8_QMAX)


class Quantizer(nn.Module):
    def __init__(self, observer_type: str = "percentile", **observer_kwargs):
        super().__init__()
        if observer_type == "percentile":
            self.observer = PercentileObserver(**observer_kwargs)
        elif observer_type == "max":
            self.observer = MaxObserver()
        else:
            raise ValueError(f"unknown observer_type: {observer_type}")
        self.register_buffer("scale", torch.tensor(0.0))
        self.mode: str = "off"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return x
        if self.mode == "observe":
            return self.observer(x)
        if self.mode == "quantize":
            return fake_quant(x, self.scale)
        raise ValueError(f"unknown mode: {self.mode}")

    def freeze(self) -> None:
        self.scale.copy_(self.observer.compute_scale().to(self.scale.device))
        self.mode = "quantize"


class FixedScaleQuantizer(nn.Module):
    def __init__(self, range_max: float):
        super().__init__()
        self.register_buffer("scale", torch.tensor(range_max / INT8_QMAX))
        self.mode: str = "off"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "quantize":
            return fake_quant(x, self.scale)
        return x

    def freeze(self) -> None:
        self.mode = "quantize"


def set_quant_mode(model: nn.Module, mode: str) -> None:
    for m in model.modules():
        if isinstance(m, (Quantizer, FixedScaleQuantizer)):
            m.mode = mode


def freeze_quantizers(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (Quantizer, FixedScaleQuantizer)):
            m.freeze()


def collect_scales(model: nn.Module) -> dict:
    scales = {}
    for name, m in model.named_modules():
        if isinstance(m, (Quantizer, FixedScaleQuantizer)):
            scales[name] = float(m.scale.item())
    return scales
