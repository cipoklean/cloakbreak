"""Unit tests for scripts/replay_capture.py using a stub detector runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.replay_capture import replay_capture
from src.audio_utils import TARGET_SR, save_wav_mono
from src.model import DetectionResult


class _StubRuntime:
    backend_name = "stub"

    def __init__(self, prob: float = 0.42) -> None:
        self.prob = prob
        self.seen_lengths: list[int] = []

    def predict(self, waveform: np.ndarray) -> DetectionResult:
        self.seen_lengths.append(int(waveform.shape[0]))
        return DetectionResult(
            spoof_prob=self.prob,
            attention_focus=[0.0] * 32,
            latency_ms=1.0,
            backend="stub",
        )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    n = int(3.0 * TARGET_SR)
    t = np.arange(n, dtype=np.float32) / TARGET_SR
    sig = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.normal(size=n)).astype(np.float32)
    wav_path = tmp_path / "cap.wav"
    save_wav_mono(wav_path, sig, TARGET_SR)
    sidecar = {
        "windows": [
            {
                "audio_start_sec": 0.0,
                "audio_end_sec": 2.5,
                "spoof_prob": 0.42,
                "baseline_prob": 0.10,
                "verdict": "bonafide",
                "window_verdict": "bonafide",
                "condition": "clean",
                "threshold": 0.5,
            },
            {
                "audio_start_sec": 0.25,
                "audio_end_sec": 2.75,
                "spoof_prob": 0.42,
                "baseline_prob": 0.10,
                "verdict": "bonafide",
                "window_verdict": "bonafide",
                "condition": "clean",
                "threshold": 0.5,
            },
        ]
    }
    json_path = tmp_path / "cap.json"
    json_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return wav_path, json_path


def test_replay_capture_stub_runtime_deterministic(tmp_path: Path) -> None:
    """Offline probs equal the logged live probs when the runtime is
    deterministic; windows are scored on the exact [start, end) samples."""
    wav_path, json_path = _write_fixture(tmp_path)
    stub = _StubRuntime(prob=0.42)
    summary = replay_capture(wav_path, sidecar_path=json_path, runtime=stub)

    assert summary["n_windows"] == 2
    assert summary["max_abs_diff"] == 0.0
    assert summary["mean_abs_diff"] == 0.0
    for row in summary["windows"]:
        assert row["offline_prob"] == 0.42
        assert row["abs_diff"] == 0.0
        assert row["offline_ge_threshold"] is False
    # exact 2.5 s windows -> 40000 samples each (degrade clean preserves length)
    assert stub.seen_lengths[:2] == [int(2.5 * TARGET_SR), int(2.5 * TARGET_SR)]
    # upload path also ran on the whole 3 s clip
    assert "spoof_prob" in summary["upload_path"]
    assert summary["upload_path"]["spoof_prob"] == 0.42
