"""Check the published trained weights without downloading Speech Commands."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from new_gru import from_torch_gru

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = json.loads((ROOT / "checkpoints/published-sweep/manifest.json").read_text())


def load_runner():
    spec = importlib.util.spec_from_file_location("reproduce_results", ROOT / "scripts/reproduce_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_uses_bundled_checkpoint(monkeypatch, tmp_path):
    runner = load_runner()
    calls = []
    monkeypatch.setattr(sys, "argv", ["reproduce_results", "--hidden-size", "32", "--seed", "2",
                                     "--out-dir", str(tmp_path)])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    runner.main()
    assert len(calls) == 1
    command, kwargs = calls[0]
    options = dict(zip(command[2:], command[3:]))
    entry = next(e for e in ENTRIES if (e["hidden_size"], e["seed"]) == (32, 2))
    assert options["--checkpoint"] == str(ROOT / entry["checkpoint"])
    assert options["--batch-size-eval"] == "256"
    assert options["--calib-batches"] == "16"
    assert options["--percentile"] == "99.99"
    assert options["--out-dir"] == str(tmp_path / "16_mels_delta_delta_h32_s2")
    assert kwargs == {"check": True, "cwd": ROOT}


def test_corrupt_checkpoint_stops_before_evaluation(monkeypatch, tmp_path):
    runner = load_runner()
    (tmp_path / "bad.pt").write_bytes(b"corrupt")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([dict(ENTRIES[-1], checkpoint="bad.pt")]))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["reproduce_results", "--all"])
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.main()


def test_verify_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/reproduce_results.py"), "--all", "--verify-only"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert result.stdout.count("Verified ") == 12


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: f"h{e['hidden_size']}-s{e['seed']}")
def test_trained_checkpoint_parity(entry):
    path = ROOT / entry["checkpoint"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    summary = json.loads((ROOT / entry["summary"]).read_text())
    assert summary["checkpoint"] == entry["source_checkpoint"]
    assert (summary["hidden_size"], summary["seed"]) == (entry["hidden_size"], entry["seed"])
    assert (ROOT / entry["training_log"]).is_file()
    assert (ROOT / entry["evaluation_log"]).is_file()
    state = torch.load(path, map_location="cpu", weights_only=True)
    gru = torch.nn.GRU(48, entry["hidden_size"], num_layers=2, batch_first=True).eval()
    gru.load_state_dict({k.removeprefix("gru."): v for k, v in state.items() if k.startswith("gru.")})
    custom = from_torch_gru(gru).eval()
    torch.manual_seed(entry["seed"])
    features = torch.randn(2, 101, 48)
    with torch.no_grad():
        expected, expected_hidden = gru(features)
        actual, actual_hidden = custom(features)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_hidden, expected_hidden, atol=1e-5, rtol=1e-5)
    assert state["classifier.weight"].shape == (12, entry["hidden_size"])
    assert state["mel.mel_scale.fb"].shape == (257, 16)
