import torch

from model import KeywordGRU
from new_gru import NewGRU
from ptq.quant_model import QuantizedKeywordGRU, from_fp32
from ptq.quant_new_gru import QuantizedNewGRU, from_new_gru
from ptq.quant_utils import (
    INT8_QMAX,
    fake_quant,
    freeze_quantizers,
    set_quant_mode,
)


def test_fake_quant_round_trip():
    torch.manual_seed(0)
    x = torch.randn(128) * 3.0
    scale = x.abs().max() / INT8_QMAX
    xq = fake_quant(x, scale)
    err = (x - xq).abs().max().item()
    assert err <= scale.item() * 1.001, f"fake_quant error {err} exceeds half-scale {scale.item()/2}"
    assert torch.all(xq.abs() <= INT8_QMAX * scale + 1e-6), "fake_quant exceeded clamp range"
    print(f"[PASS] fake_quant round-trip (max |err|={err:.3e}, scale={scale.item():.3e})")


def test_quantized_gru_off_matches_fp32():
    torch.manual_seed(0)
    input_size, hidden_size, num_layers = 12, 16, 2
    batch, seq_len = 2, 8

    fp = NewGRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.0).eval()
    q = from_new_gru(fp).eval()
    set_quant_mode(q, "off")

    x = torch.randn(batch, seq_len, input_size)
    with torch.no_grad():
        out_fp, h_fp = fp(x, return_sequences=False)
        out_q, h_q = q(x, return_sequences=False)

    diff = (out_fp - out_q).abs().max().item()
    h_diff = (h_fp - h_q).abs().max().item()
    assert diff < 1e-5 and h_diff < 1e-5, f"off-mode mismatch: out_diff={diff} h_diff={h_diff}"
    print(f"[PASS] off-mode parity NewGRU vs QuantizedNewGRU (out_diff={diff:.2e}, h_diff={h_diff:.2e})")


def test_pipeline_runs_random_weights():
    torch.manual_seed(0)
    fp = KeywordGRU(
        n_mels=8,
        hidden_size=8,
        num_layers=2,
        use_delta=False,
        use_delta_delta=False,
        use_new_gru=True,
        dropout=0.0,
        precomputed_features=True,
    ).eval()

    q = from_fp32(fp).eval()

    n_frames = 20
    feat_dim = 8
    x = torch.randn(4, feat_dim, n_frames)

    set_quant_mode(q, "observe")
    with torch.no_grad():
        for _ in range(2):
            _ = q(x)
    freeze_quantizers(q)
    set_quant_mode(q, "quantize")

    with torch.no_grad():
        out_fp = fp(x)
        out_q = q(x)

    assert torch.isfinite(out_q).all(), "INT8 output contains NaN/Inf"
    fp_pred = out_fp.argmax(dim=1)
    q_pred = out_q.argmax(dim=1)
    agreement = (fp_pred == q_pred).float().mean().item()
    print(f"[PASS] pipeline runs end-to-end; FP32 vs INT8 top-1 agreement = {agreement*100:.1f}%")


if __name__ == "__main__":
    test_fake_quant_round_trip()
    test_quantized_gru_off_matches_fp32()
    test_pipeline_runs_random_weights()
    print("All quant PTQ sanity tests passed.")
