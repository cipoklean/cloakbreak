"""LiveScorer unit tests + FastAPI websocket lag tests.

The LiveScorer decouples PCM ingest from a single-worker inference loop that
always scores the NEWEST window; these tests verify bounded lag, the k-of-n
smoothing rule, and the end-to-end websocket contract with a slow detector.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import AppState, create_app
from src.audio_utils import TARGET_SR, float32_to_pcm16_bytes, load_wav_mono, pcm16_bytes_to_float32
from src.live import LiveScorer
from src.model import DetectionResult

REPO_ROOT = Path(__file__).resolve().parents[1]


def _speech_chunk(n: int, seed: int = 0) -> np.ndarray:
    """Loud speech-like noise; RMS far above the 0.012 rms_vad threshold."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / TARGET_SR
    env = 0.6 + 0.4 * np.sin(2.0 * np.pi * 3.0 * t)
    sig = env * (0.35 * np.sin(2.0 * np.pi * 220.0 * t) + 0.12 * np.sin(2.0 * np.pi * 550.0 * t))
    sig += rng.normal(0.0, 0.01, size=n).astype(np.float32)
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


def _slow_score_fn(window: np.ndarray) -> dict:
    time.sleep(0.7)
    return {
        "spoof_prob": 0.9,
        "attention_focus": [0.0] * 32,
        "latency_ms": 700.0,
        "backend": "stub",
    }


def test_live_scorer_bounds_lag_single_inflight() -> None:
    """60 s of speech at 5x real time through a 0.7 s scorer: lag stays bounded,
    at most one inference in flight, stale strides are dropped not queued."""

    async def main():
        emitted: list[dict] = []

        async def emit(msg: dict) -> None:
            emitted.append(msg)

        scorer = LiveScorer(_slow_score_fn, threshold=0.5, sr=TARGET_SR)
        task = asyncio.create_task(scorer.run(emit))

        chunk = _speech_chunk(1365)
        chunk_sec = 1365 / TARGET_SR
        speed = 5.0
        interval = chunk_sec / speed
        n_chunks = int(round(60.0 / chunk_sec))
        t_feed0 = time.monotonic()
        for _ in range(n_chunks):
            scorer.push(chunk)
            await asyncio.sleep(interval)
        feed_wall = time.monotonic() - t_feed0
        await asyncio.sleep(1.5)
        total_wall = time.monotonic() - t_feed0
        scorer.stop()
        await asyncio.wait_for(task, timeout=15.0)
        return scorer, emitted, feed_wall, total_wall

    scorer, emitted, feed_wall, total_wall = asyncio.run(main())

    speech = [m for m in emitted if m.get("is_speech") is True and m.get("spoof_prob") is not None]
    assert speech, "expected speech verdicts"
    assert scorer.inflight_max == 1
    assert scorer.max_pending_windows <= 1
    assert all(m["lag_ms"] < 2000 for m in speech)
    # scoring is bounded by model speed (~0.7 s/window), not by the 240+ strides
    lo = int(total_wall / 0.75) - 2
    hi = int(np.ceil(total_wall / 0.7)) + 2
    assert lo <= scorer.n_scored <= hi, (scorer.n_scored, lo, hi)
    assert scorer.dropped_windows > 0
    ends = [m["audio_end_sec"] for m in speech]
    assert all(b > a for a, b in zip(ends, ends[1:])), "audio_end_sec must increase"
    # newest-window policy: the last verdict scored audio ending within
    # (5x speed * 0.7 s inference + 0.5 s slack) of what had been received.
    last = speech[-1]
    assert last["stream_sec_received"] - last["audio_end_sec"] <= (5.0 * 0.7 + 0.5)
    # push never blocked: the ~12 s feed (60 s audio at 5x) finished on
    # schedule. asyncio.sleep overshoot under load accumulates ~1.5 s here,
    # so allow 35% headroom — a blocked push() would add seconds per chunk.
    assert feed_wall < 12.0 * 1.35
    assert total_wall < feed_wall + 1.5 + 3.0


def test_live_scorer_smoothing_verdicts() -> None:
    """k-of-n smoothing: verdict needs 2 of last 3 window scores >= threshold."""

    async def main():
        emitted: list[dict] = []

        async def emit(msg: dict) -> None:
            emitted.append(msg)

        probs = iter([0.99, 0.10, 0.99, 0.99, 0.10, 0.10])

        def score_fn(window: np.ndarray) -> dict:
            return {
                "spoof_prob": next(probs),
                "attention_focus": [0.0] * 32,
                "latency_ms": 1.0,
                "backend": "stub",
            }

        scorer = LiveScorer(score_fn, threshold=0.9526, sr=TARGET_SR)
        task = asyncio.create_task(scorer.run(emit))
        chunk = _speech_chunk(40000)  # one full 2.5 s window per push
        speech_seen = 0
        for _ in range(6):
            scorer.push(chunk)
            for _ in range(2000):
                await asyncio.sleep(0.005)
                if sum(1 for m in emitted if m.get("is_speech") is True) > speech_seen:
                    speech_seen += 1
                    break
            else:
                raise AssertionError("timed out waiting for speech verdict")
        scorer.stop()
        await asyncio.wait_for(task, timeout=5.0)
        return emitted

    emitted = asyncio.run(main())
    speech = [m for m in emitted if m.get("is_speech") is True]
    assert [m["spoof_prob"] for m in speech] == [0.99, 0.10, 0.99, 0.99, 0.10, 0.10]
    assert [m["window_verdict"] for m in speech] == [
        "spoof", "bonafide", "spoof", "spoof", "bonafide", "bonafide",
    ]
    assert [m["verdict"] for m in speech] == [
        "bonafide", "bonafide", "spoof", "spoof", "spoof", "bonafide",
    ]
    for m in speech:
        assert len(m["recent_scores"]) <= 3
    assert speech[0]["recent_scores"] == [0.99]
    assert speech[2]["recent_scores"] == [0.99, 0.10, 0.99]
    assert speech[3]["recent_scores"] == [0.10, 0.99, 0.99]


class _SlowDetector:
    """DetectorRuntime-shaped stub whose predict sleeps like the fp32 ONNX path."""

    backend_name = "slow_stub"

    def predict(self, waveform):
        time.sleep(0.7)
        return DetectionResult(
            spoof_prob=0.9,
            attention_focus=[0.0] * 32,
            latency_ms=700.0,
            backend=self.backend_name,
        )


def test_ws_live_scorer_lag_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: 60 s PCM16 at 5x through /ws/detect with a 0.7 s detector."""
    monkeypatch.setenv("CLOAKBREAK_BACKEND", "mock")
    monkeypatch.setenv("ENABLE_CLONE_ENDPOINT", "false")
    monkeypatch.setenv("CLOAKBREAK_THRESHOLD_FILE", "/nonexistent/threshold.json")
    state = AppState()
    state.detector = _SlowDetector()
    client = TestClient(create_app(state))

    chunk = _speech_chunk(1365)
    pcm = float32_to_pcm16_bytes(chunk)
    n_chunks = int(round(60.0 * TARGET_SR / 1365))
    interval = (1365 / TARGET_SR) / 5.0

    preds: list[dict] = []
    stats: dict | None = None
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_json({"condition": "clean"})
        ack = ws.receive_json()
        assert ack["type"] == "ack" and ack["condition"] == "clean"

        t0 = time.monotonic()
        for _ in range(n_chunks):
            ws.send_bytes(pcm)
            time.sleep(interval)
        time.sleep(1.5)
        ws.send_json({"type": "stats"})

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            msg = ws.receive_json()
            if msg.get("type") == "stats":
                stats = msg
                break
            preds.append(msg)
        total_wall = time.monotonic() - t0

    assert stats is not None, "no stats reply"
    speech = [
        m for m in preds
        if m.get("is_speech") is True and m.get("spoof_prob") is not None
    ]
    assert speech, "expected speech verdicts"
    assert all(m["lag_ms"] < 2000 for m in speech)
    assert stats["inflight_max"] == 1
    assert stats["max_pending_windows"] <= 1
    assert stats["dropped_windows"] > 0
    lo = int(total_wall / 0.75) - 2
    hi = int(np.ceil(total_wall / 0.7)) + 2
    assert lo <= stats["n_scored"] <= hi, (stats["n_scored"], lo, hi)
    for m in speech:
        for key in (
            "audio_end_sec", "inference_ms", "lag_ms", "verdict",
            "window_verdict", "recent_scores", "stream_sec_received",
            "dropped_windows", "threshold",
        ):
            assert key in m, (key, m)
        assert m["verdict"] in {"spoof", "bonafide"}
        assert m["window_verdict"] in {"spoof", "bonafide"}
        assert len(m["recent_scores"]) <= 3


def test_live_scorer_stops_after_repeated_failures() -> None:
    """A detector that always raises must not be retried forever: after 3
    consecutive failures the scorer emits an error and its loop exits."""

    def boom(window: np.ndarray) -> dict:
        raise RuntimeError("model crashed")

    async def main():
        emitted: list[dict] = []

        async def emit(msg: dict) -> None:
            emitted.append(msg)

        scorer = LiveScorer(boom, threshold=0.5, sr=TARGET_SR)
        task = asyncio.create_task(scorer.run(emit))
        chunk = _speech_chunk(1365)
        for _ in range(200):
            scorer.push(chunk)
            await asyncio.sleep(0.005)
            if task.done():
                break
        await asyncio.wait_for(task, timeout=5.0)
        return scorer, emitted

    scorer, emitted = asyncio.run(main())
    errors = [m for m in emitted if m.get("type") == "error"]
    assert scorer.consecutive_failures == 3
    assert len(errors) == 1 and "3 times in a row" in errors[0]["detail"]
    assert scorer.n_scored == 0


def test_bench_send_time_for_audio_end() -> None:
    """Pure helper: audio_end_sec -> send time of first chunk ending >= it."""
    from scripts.bench_live_lag import send_time_for_audio_end

    # 3 chunks of 1000 samples each at sr=1000 -> ends at samples 1000/2000/3000
    ends = [1000, 2000, 3000]
    times = [10.0, 10.1, 10.2]
    # audio_end 0.5s -> end_idx 500 -> chunk 0 (end 1000 >= 500)
    assert send_time_for_audio_end(0.5, 1000, ends, times) == 10.0
    # exactly on a boundary -> that chunk (end 1000 >= 1000)
    assert send_time_for_audio_end(1.0, 1000, ends, times) == 10.0
    # just past (beyond the 3-decimal rounding slack) -> next chunk
    assert send_time_for_audio_end(1.002, 1000, ends, times) == 10.1
    # server-rounded value must map back to the chunk it came from:
    # 9 chunks of 1365 samples -> total 12285 -> 0.767812 s -> sent as 0.768
    ends16 = [1365 * (k + 1) for k in range(12)]
    times16 = [float(k) for k in range(12)]
    assert send_time_for_audio_end(round(12285 / 16000, 3), 16000, ends16, times16) == 8.0
    assert send_time_for_audio_end(round(13650 / 16000, 3), 16000, ends16, times16) == 9.0
    assert send_time_for_audio_end(2.5, 1000, ends, times) == 10.2
    # beyond the stream -> last send time
    assert send_time_for_audio_end(9.9, 1000, ends, times) == 10.2
    assert send_time_for_audio_end(0.5, 1000, ends, []) is None


def test_ws_debug_capture_wav_and_sidecar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLOAKBREAK_DEBUG_CAPTURE=1: WAV is byte-exact what the scorer saw and the
    sidecar lists every emitted speech verdict; nothing is written when off."""
    capdir = tmp_path / "captures"
    monkeypatch.setenv("CLOAKBREAK_BACKEND", "mock")
    monkeypatch.setenv("CLOAKBREAK_DEBUG_CAPTURE", "1")
    monkeypatch.setenv("CLOAKBREAK_CAPTURE_DIR", str(capdir))
    client = TestClient(create_app(AppState()))
    assert client.get("/api/health").json()["debug_capture"] is True

    speech = _speech_chunk(int(3.0 * TARGET_SR), seed=1)
    pcm = float32_to_pcm16_bytes(speech)
    chunk_bytes = 1365 * 2
    sent = bytearray()
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_json({"condition": "clean"})
        assert ws.receive_json()["type"] == "ack"
        for i in range(0, len(pcm), chunk_bytes):
            part = pcm[i : i + chunk_bytes]
            ws.send_bytes(part)
            sent += part
        saw_speech = False
        for _ in range(40):
            m = ws.receive_json()
            if m.get("type") == "prediction" and m.get("is_speech") and m.get("spoof_prob") is not None:
                saw_speech = True
                break
        assert saw_speech, "expected at least one speech verdict before disconnect"

    # The WAV+JSON are written in the handler's finally — poll briefly.
    wavs: list[Path] = []
    jsons: list[Path] = []
    for _ in range(40):
        wavs = sorted(capdir.glob("*.wav"))
        jsons = sorted(capdir.glob("*.json"))
        if wavs and jsons:
            break
        time.sleep(0.25)
    assert len(wavs) == 1 and len(jsons) == 1, f"capture files missing in {capdir}"
    wav, sr = load_wav_mono(wavs[0], target_sr=TARGET_SR)
    expected = pcm16_bytes_to_float32(bytes(sent))
    assert sr == TARGET_SR
    assert wav.shape[0] == expected.shape[0]
    np.testing.assert_array_equal(wav, expected)  # PCM16 round-trip is exact

    meta = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert meta["truncated"] is False
    assert meta["sample_rate"] == TARGET_SR and meta["channels"] == 1
    assert meta["windows"], "sidecar must list the emitted speech verdicts"
    for w in meta["windows"]:
        for key in (
            "audio_start_sec", "audio_end_sec", "spoof_prob", "baseline_prob",
            "verdict", "window_verdict", "condition", "threshold",
        ):
            assert key in w, f"missing {key} in {w}"
        assert w["audio_end_sec"] > w["audio_start_sec"]
        assert 0.0 <= w["spoof_prob"] <= 1.0

    # Flag off -> no capture files at all.
    monkeypatch.delenv("CLOAKBREAK_DEBUG_CAPTURE")
    other = tmp_path / "other"
    monkeypatch.setenv("CLOAKBREAK_CAPTURE_DIR", str(other))
    client2 = TestClient(create_app(AppState()))
    assert client2.get("/api/health").json()["debug_capture"] is False
    with client2.websocket_connect("/ws/detect") as ws:
        ws.send_bytes(pcm[: chunk_bytes * 4])
        ws.receive_json()
    assert not other.exists() or not list(other.glob("*.wav"))
