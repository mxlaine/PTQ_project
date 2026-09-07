from pathlib import Path

import pytest
import torch
import torch.nn as nn

from new_gru import NewGRU, from_torch_gru
from model import KeywordGRU
from utils import build_dataloaders


def test_random_input_equivalence():
    torch.manual_seed(0)
    input_size, hidden_size, num_layers = 48, 64, 2
    batch, seq_len = 4, 97

    torch_gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=0.0,
    ).eval()

    new_gru = from_torch_gru(torch_gru).eval()

    x = torch.randn(batch, seq_len, input_size)
    with torch.no_grad():
        out_t, h_t = torch_gru(x)
        out_n, h_n = new_gru(x)

    out_diff = (out_t - out_n).abs().max().item()
    h_diff = (h_t - h_n).abs().max().item()
    print(f"random-input max |Δ| output={out_diff:.2e} hidden={h_diff:.2e}")
    assert out_diff < 1e-5, f"output mismatch: {out_diff}"
    assert h_diff < 1e-5, f"hidden mismatch: {h_diff}"
    print("[PASS] random-input numerical parity")


def test_checkpoint_accuracy():
    ckpt_path = Path(__file__).resolve().parents[1] / "notebooks" / "best_keyword_gru_24_mels_delta.pt"
    if not ckpt_path.exists():
        pytest.skip(f"checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_kwargs = dict(
        n_mels=24,
        hidden_size=64,
        num_layers=2,
        use_delta=True,
        use_delta_delta=False,
        spec_augment=False,
        dropout=0.0,
    )

    state = torch.load(ckpt_path, map_location=device)

    torch_model = KeywordGRU(use_new_gru=False, **common_kwargs).to(device).eval()
    torch_model.load_state_dict(state)

    new_model = KeywordGRU(use_new_gru=True, **common_kwargs).to(device).eval()
    new_model.load_state_dict(state)

    _, _, test_loader = build_dataloaders(
        batch_size_train=64, batch_size_eval=256, pin_memory=False
    )

    correct_t = correct_n = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pt = torch_model(data).argmax(dim=1)
            pn = new_model(data).argmax(dim=1)
            correct_t += (pt == target).sum().item()
            correct_n += (pn == target).sum().item()
            total += target.size(0)

    acc_t = 100.0 * correct_t / total
    acc_n = 100.0 * correct_n / total
    print(f"test acc nn.GRU={acc_t:.4f}%  NewGRU={acc_n:.4f}%  Δ={abs(acc_t - acc_n):.4f}%")
    assert abs(acc_t - acc_n) < 0.05, "checkpoint accuracy mismatch between nn.GRU and NewGRU"
    print("[PASS] trained-checkpoint accuracy match")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
