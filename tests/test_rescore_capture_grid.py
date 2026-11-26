"""Grid re-scorer: gate replay over a quiet-headset-like capture, and the summary counts."""

from __future__ import annotations

import numpy as np

from scripts.rescore_capture_grid import gate_decisions, grid_starts, summarize
from src.audio_utils import TARGET_SR


def _at_dbfs(x: np.ndarray, dbfs: float) -> np.ndarray:
    return (x * (10.0 ** (dbfs / 20.0) / np.sqrt(np.mean(x.astype(np.float64) ** 2)))).astype(np.float32)


def test_gate_replay_passes_quiet_speech_and_skips_floor() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(int(4.0 * TARGET_SR)) / TARGET_SR
    speech = _at_dbfs((0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) * np.sin(2 * np.pi * 150 * t), -42.0)
    noise = lambda sec: _at_dbfs(rng.normal(size=int(sec * TARGET_SR)), -55.0)
    wav = np.concatenate([noise(4.0), speech + noise(4.0), noise(4.0)])

    starts = grid_starts(wav.shape[0], 0.5)
    assert starts[0] == 0.0 and starts[-1] == 9.5 and len(starts) == 20
    gates = gate_decisions(wav, starts)
    by_start = dict(zip(starts, gates))
    assert by_start[0.0]["gate_speech"] is False          # noise floor only
    assert by_start[4.5]["gate_speech"] is True           # -42 dBFS speech
    assert by_start[4.5]["old_rms_vad_speech"] is False   # old fixed gate missed it
    assert by_start[9.5]["gate_speech"] is False          # back to the floor

    rows = [{"spoof": 0.99 if g["gate_speech"] else 0.01, "baseline": 0.1, **g} for g in gates]
    s = summarize(rows, threshold=0.95)
    assert s["n_windows"] == 20
    assert s["gate_n_speech"] == s["n_ge_threshold"] == s["gate_n_ge_threshold"]
    assert s["old_rms_vad_n_speech"] == 0
    assert s["gate_frac_ge_threshold"] == 1.0
    assert s["baseline_n_ge_0.5"] == 0
