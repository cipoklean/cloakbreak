"""SpeechGate unit tests + LiveScorer end-to-end gate behavior.

The gate replaces the fixed rms_vad threshold on the live path: it tracks the
rolling noise floor (10th percentile of 0.25 s block levels) and opens
``margin_db`` above it. These tests cover the gate with and without history,
floor tracking, the rms_vad regression, and a LiveScorer run over a quiet
headset-like stream (speech near -42 dBFS over a -60 dBFS floor).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src.audio_utils import TARGET_SR, SpeechGate, audio_rms, rms_dbfs, rms_vad
from src.live import LiveScorer


def _at_dbfs(x: np.ndarray, dbfs: float) -> np.ndarray:
    """Scale x so its RMS is exactly ``dbfs`` dBFS."""
    rms = audio_rms(x)
    if rms <= 0.0:
        return x.astype(np.float32)
    return (x * (10.0 ** (dbfs / 20.0) / rms)).astype(np.float32)


def _noise_dbfs(dbfs: float, dur_s: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.05, size=int(round(dur_s * TARGET_SR))).astype(np.float32)
    return _at_dbfs(x, dbfs)


def _speech_like_dbfs(dbfs: float, dur_s: float, seed: int = 0) -> np.ndarray:
    """150 Hz harmonic stack with syllable-rate amplitude modulation."""
    rng = np.random.default_rng(seed)
    n = int(round(dur_s * TARGET_SR))
    t = np.arange(n, dtype=np.float32) / TARGET_SR
    env = 0.5 + 0.5 * np.sin(2.0 * np.pi * 4.0 * t + 0.7)  # ~4 Hz syllable rate
    sig = env * (
        np.sin(2.0 * np.pi * 150.0 * t)
        + 0.5 * np.sin(2.0 * np.pi * 300.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 450.0 * t)
    )
    sig += rng.normal(0.0, 0.002, size=n).astype(np.float32)
    return _at_dbfs(sig.astype(np.float32), dbfs)


def _push_seconds(gate: SpeechGate, x: np.ndarray, chunk: int = 1365) -> None:
    for i in range(0, x.size, chunk):
        gate.push(x[i : i + chunk])


def test_gate_with_history_noise_floor() -> None:
    """3 s of -60 dBFS noise history: -42 speech passes, -60 noise does not."""
    gate = SpeechGate()
    _push_seconds(gate, _noise_dbfs(-60.0, 3.0))
    floor = gate.noise_floor_dbfs()
    assert floor is not None and floor == pytest.approx(-60.0, abs=1.5)
    assert gate.gate_dbfs() == pytest.approx(max(-60.0, floor + 6.0))

    speech_win = _speech_like_dbfs(-42.0, 2.5, seed=1)
    noise_win = _noise_dbfs(-60.0, 2.5, seed=2)
    assert rms_dbfs(speech_win) == pytest.approx(-42.0, abs=0.2)
    assert gate.is_speech(speech_win) is True
    assert gate.is_speech(noise_win) is False


def test_gate_without_history_falls_back() -> None:
    """No history -> fallback gate -50: -42 passes, -60 does not, zeros do not."""
    gate = SpeechGate()
    assert gate.noise_floor_dbfs() is None
    assert gate.gate_dbfs() == pytest.approx(-50.0)
    assert gate.is_speech(_speech_like_dbfs(-42.0, 2.5, seed=3)) is True
    assert gate.is_speech(_noise_dbfs(-60.0, 2.5, seed=4)) is False
    assert gate.is_speech(np.zeros(4000, dtype=np.float32)) is False


def test_gate_all_zeros_stream_is_not_speech() -> None:
    """Digital silence never enters the floor; gate stays at fallback."""
    gate = SpeechGate()
    _push_seconds(gate, np.zeros(int(3.0 * TARGET_SR), dtype=np.float32))
    assert gate.noise_floor_dbfs() is None
    assert gate.gate_dbfs() == pytest.approx(-50.0)
    assert gate.is_speech(np.zeros(4000, dtype=np.float32)) is False


def test_gate_floor_tracks_louder_noise() -> None:
    """10 s of -45 dBFS noise lifts the floor; a -47 window is then rejected."""
    gate = SpeechGate()
    _push_seconds(gate, _noise_dbfs(-45.0, 10.0, seed=5))
    floor = gate.noise_floor_dbfs()
    assert floor is not None and floor == pytest.approx(-45.0, abs=1.5)
    assert gate.gate_dbfs() == pytest.approx(-39.0, abs=1.5)
    assert gate.is_speech(_noise_dbfs(-47.0, 2.5, seed=6)) is False
    # ...while clearly louder signal still passes.
    assert gate.is_speech(_speech_like_dbfs(-30.0, 2.5, seed=7)) is True


def test_rms_vad_regression_fixed_threshold_rejects_quiet_speech() -> None:
    """The old bug: rms_vad's fixed 0.012 (-38.4 dBFS) rejects -42 dBFS speech."""
    speech_win = _speech_like_dbfs(-42.0, 2.5, seed=8)
    assert rms_vad(speech_win) is False

    gate = SpeechGate()
    _push_seconds(gate, _noise_dbfs(-60.0, 3.0, seed=9))
    assert gate.is_speech(speech_win) is True


def test_gate_min_gate_floor_clamp() -> None:
    """A very low noise floor cannot pull the gate below min_gate_dbfs (-60)."""
    gate = SpeechGate()
    _push_seconds(gate, _noise_dbfs(-80.0, 3.0, seed=10))
    floor = gate.noise_floor_dbfs()
    assert floor is not None
    assert gate.gate_dbfs() == pytest.approx(-60.0)


def test_live_scorer_gate_end_to_end() -> None:
    """3 s of -60 dBFS noise then 4 s of -42 dBFS speech-like signal through a
    LiveScorer with a stub score_fn: speech windows get scored, noise-only
    windows do not, and every prediction carries the gate fields."""

    async def main():
        emitted: list[dict] = []

        async def emit(msg: dict) -> None:
            emitted.append(msg)

        def score_fn(window: np.ndarray) -> dict:
            return {
                "spoof_prob": 0.5,
                "attention_focus": [0.0] * 32,
                "latency_ms": 1.0,
                "backend": "stub",
            }

        scorer = LiveScorer(score_fn, threshold=0.9, sr=TARGET_SR)
        task = asyncio.create_task(scorer.run(emit))

        noise = _noise_dbfs(-60.0, 3.0, seed=11)
        speech = _speech_like_dbfs(-42.0, 4.0, seed=12)
        stream = np.concatenate([noise, speech])
        chunk = 1365
        for i in range(0, stream.size, chunk):
            scorer.push(stream[i : i + chunk])
            await asyncio.sleep(0.002)
        await asyncio.sleep(0.5)
        scorer.stop()
        await asyncio.wait_for(task, timeout=10.0)
        return emitted

    emitted = asyncio.run(main())
    preds = [m for m in emitted if m.get("type") == "prediction"]
    assert preds, "expected prediction messages"
    for m in preds:
        assert "input_dbfs" in m
        assert "noise_floor_dbfs" in m
        assert "speech_gate_dbfs" in m

    scored = [m for m in preds if m.get("spoof_prob") is not None]
    assert scored, "expected at least one scored window"
    # No scored window may end inside the noise-only part (t <= 3.0 s).
    assert all(m["audio_end_sec"] > 3.0 for m in scored), [
        m["audio_end_sec"] for m in scored
    ]
    # At least one scored window ends during the speech part.
    assert any(m["audio_end_sec"] > 3.0 for m in scored)
    # The speech part was actually reached: some scored windows end well inside it.
    assert any(m["audio_end_sec"] >= 5.0 for m in scored), [
        m["audio_end_sec"] for m in scored
    ]
    # Gate fields are sane on scored windows.
    for m in scored:
        assert m["input_dbfs"] > m["speech_gate_dbfs"] - 1.0
        assert m["is_speech"] is True
