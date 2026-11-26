"""Tests for the job-queue analysis API: all-window scoring, uploads, analyze-url,
facts/honest-limits, the one-at-a-time live-mic rule, and FIFO backpressure.

Everything runs on the deterministic zero-download mock backend
(CLOAKBREAK_BACKEND=mock); detector latency is simulated by monkeypatching
MockDetectorRuntime.predict where ordering matters.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import AppState, create_app
from src.analysis import analyze_clip
from src.audio_utils import TARGET_SR, audio_rms, save_wav_mono, wav_to_bytes
from src.jobs import (
    BLOCKED_ERROR_MESSAGE,
    DECODE_ERROR_MESSAGE,
    SSRF_ERROR_MESSAGE,
    TIMEOUT_ERROR_MESSAGE,
    JobQueue,
)
from src.model import DetectionResult, MockDetectorRuntime

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# -- helpers ------------------------------------------------------------------


def _speech(dur_s: float, freq_hz: float = 220.0, seed: int = 0) -> np.ndarray:
    """Loud speech-like harmonic clip (~-15 dBFS), clearly above any gate."""
    rng = np.random.default_rng(seed)
    n = int(round(TARGET_SR * dur_s))
    t = np.arange(n, dtype=np.float32) / TARGET_SR
    env = 0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
    sig = env * (0.3 * np.sin(2.0 * np.pi * freq_hz * t) + 0.15 * np.sin(2.0 * np.pi * 2.5 * freq_hz * t))
    sig += rng.normal(0.0, 0.01, size=n).astype(np.float32)
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


def _at_dbfs(x: np.ndarray, dbfs: float) -> np.ndarray:
    rms = audio_rms(x)
    if rms <= 0.0:
        return x.astype(np.float32)
    return (x * (10.0 ** (dbfs / 20.0) / rms)).astype(np.float32)


def _noise_dbfs(dbfs: float, dur_s: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.05, size=int(round(dur_s * TARGET_SR))).astype(np.float32)
    return _at_dbfs(x, dbfs)


class _ScriptedDetector:
    """DetectorRuntime-shaped stub returning scripted spoof probs in call order."""

    backend_name = "scripted"

    def __init__(self, probs: list[float]) -> None:
        self.probs = list(probs)
        self.calls = 0

    def predict(self, waveform: np.ndarray) -> DetectionResult:
        prob = self.probs[self.calls % len(self.probs)]
        self.calls += 1
        return DetectionResult(prob, [0.0] * 32, 1.0, self.backend_name)


def _analyze(wav: np.ndarray, probs: list[float], threshold: float = 0.5, condition: str = "clean") -> dict:
    det = _ScriptedDetector(probs)
    out = analyze_clip(
        wav,
        condition,
        detector=det,
        baseline=None,
        threshold=threshold,
        min_window_sec=2.0,
        input_type="upload",
    )
    out["_detector_calls"] = det.calls
    return out


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CLOAKBREAK_BACKEND", "mock")
    monkeypatch.setenv("CLOAKBREAK_WEB_DIST", "/nonexistent-web-dist")
    return TestClient(create_app(AppState()))


def _wait_job(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        view = r.json()
        if view["status"] in ("done", "error"):
            return view
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout} s")


def _slow_mock_predict(monkeypatch: pytest.MonkeyPatch, delay: float = 0.3) -> None:
    orig = MockDetectorRuntime.predict

    def slow(self, waveform: np.ndarray) -> DetectionResult:
        time.sleep(delay)
        return orig(self, waveform)

    monkeypatch.setattr(MockDetectorRuntime, "predict", slow)


# -- analysis pipeline (unit level) --------------------------------------------


def test_analysis_windows_six_second_clip() -> None:
    out = _analyze(_speech(6.0), [0.1, 0.1])
    assert out["n_windows"] == 2
    assert [w["start"] for w in out["windows"]] == [0.0, 2.5]
    assert [w["end"] for w in out["windows"]] == [2.5, 5.0]
    assert out["n_speech"] == 2
    assert out["duration_s"] == pytest.approx(6.0, abs=0.01)
    for w in out["windows"]:
        assert w["is_speech"] is True
        assert w["spoof_prob"] == pytest.approx(0.1)
        assert w["window_verdict"] == "bonafide"
        assert w["attention_focus"] is not None
        assert isinstance(w["dbfs"], float)


def test_analysis_short_clip_single_partial_window() -> None:
    out = _analyze(_speech(2.2), [0.1])
    assert out["n_windows"] == 1
    w = out["windows"][0]
    assert w["start"] == 0.0
    assert w["end"] == pytest.approx(2.2, abs=0.01)


def test_analysis_rejects_clips_under_minimum() -> None:
    from src.analysis import ClipTooShortError

    with pytest.raises(ClipTooShortError) as exc:
        _analyze(_speech(1.9), [0.1])
    assert "need at least 2.0 s of audio" in str(exc.value)
    assert "clips under 1 s are at chance" in str(exc.value)


def test_analysis_truncates_to_60_seconds() -> None:
    # 62 s of speech -> exactly 60 s analyzed -> 24 non-overlapping windows
    out = _analyze(_speech(62.0), [0.1] * 30)
    assert out["duration_s"] == pytest.approx(60.0, abs=0.01)
    assert out["n_windows"] == 24
    assert out["_detector_calls"] == 24


def test_verdict_rule_two_of_three_consecutive_is_spoof() -> None:
    wav = _speech(7.5)  # 3 full windows
    # hits at speech positions 0 and 2: inside one span of 3 consecutive windows
    out = _analyze(wav, [0.99, 0.10, 0.99])
    assert out["verdict"] == "spoof"
    assert out["n_over_threshold"] == 2
    assert out["frac_over_threshold"] == pytest.approx(2 / 3, abs=0.001)
    assert out["max_spoof_prob"] == pytest.approx(0.99)
    assert out["verdict_rule"] == "2 of any 3 consecutive speech windows at or above the threshold"

    out_adjacent = _analyze(wav, [0.99, 0.99, 0.10])
    assert out_adjacent["verdict"] == "spoof"


def test_verdict_rule_one_of_three_is_bonafide() -> None:
    wav = _speech(7.5)
    for probs in ([0.99, 0.10, 0.10], [0.10, 0.99, 0.10], [0.10, 0.10, 0.99]):
        out = _analyze(wav, list(probs))
        assert out["verdict"] == "bonafide", probs
        assert out["n_over_threshold"] == 1


def test_verdict_rule_single_speech_window_decides_alone() -> None:
    wav = _speech(2.5)  # exactly one window
    assert _analyze(wav, [0.99])["verdict"] == "spoof"
    assert _analyze(wav, [0.10])["verdict"] == "bonafide"


def test_verdict_rule_hits_beyond_span_of_three_do_not_count() -> None:
    # 10 s -> 4 speech windows; hits at positions 0 and 3 are never inside the
    # same 3-consecutive span, so the verdict must stay bonafide.
    out = _analyze(_speech(10.0), [0.99, 0.10, 0.10, 0.99])
    assert out["n_over_threshold"] == 2
    assert out["verdict"] == "bonafide"
    # but hits at 0 and 2 are inside a span -> spoof
    out2 = _analyze(_speech(10.0), [0.99, 0.10, 0.99, 0.10])
    assert out2["verdict"] == "spoof"


def test_silent_windows_are_skipped_not_scored() -> None:
    speech = _speech(2.5, seed=1)
    quiet = _noise_dbfs(-60.0, 2.5, seed=2)
    wav = np.concatenate([speech, quiet, speech])
    assert wav.shape[0] == int(7.5 * TARGET_SR)

    out = _analyze(wav, [0.99, 0.99])
    assert out["n_windows"] == 3
    assert out["n_speech"] == 2
    # the detector only ran for the two speech windows
    assert out["_detector_calls"] == 2
    mid = out["windows"][1]
    assert mid["is_speech"] is False
    assert mid["spoof_prob"] is None
    assert mid["baseline_prob"] is None
    assert mid["window_verdict"] is None
    assert mid["attention_focus"] is None
    assert mid["dbfs"] < -55.0
    # the two speech windows are consecutive in SPEECH sequence -> spoof
    assert out["verdict"] == "spoof"


def test_all_silence_clip_is_bonafide_with_no_scores() -> None:
    out = _analyze(_noise_dbfs(-60.0, 5.0, seed=3), [0.99])
    assert out["n_speech"] == 0
    assert out["_detector_calls"] == 0
    assert out["verdict"] == "bonafide"
    assert out["max_spoof_prob"] is None
    assert out["frac_over_threshold"] == 0.0


def test_response_shape_top_level_keys() -> None:
    out = _analyze(_speech(2.5), [0.4])
    for key in (
        "type", "condition", "duration_s", "threshold", "verdict", "verdict_rule",
        "n_windows", "n_speech", "n_over_threshold", "frac_over_threshold",
        "max_spoof_prob", "latency_ms", "backend", "windows",
    ):
        assert key in out, key
    assert out["type"] == "upload"
    assert out["backend"] == "scripted"
    assert out["latency_ms"] >= 0.0


# -- job queue unit behavior ----------------------------------------------------


def test_job_queue_expiry_with_injected_clock() -> None:
    now = [1000.0]
    q = JobQueue(expire_sec=600.0, clock=lambda: now[0])
    jid = q.submit(lambda: {"ok": True}, kind="upload")
    assert q.wait(jid, timeout=5.0) is True
    assert q.status_view(jid)["status"] == "done"
    # not yet expired
    now[0] += 599.0
    q.submit(lambda: {"ok": True}, kind="upload")  # triggers purge pass
    assert q.status_view(jid) is not None
    # past the 10-minute expiry -> purged on next access
    now[0] += 2.0
    assert q.status_view(jid) is None


def test_job_queue_capacity_counts_unfinished() -> None:
    from src.jobs import QueueFullError

    q = JobQueue(max_unfinished=2)
    gate = threading.Event()

    def blocker() -> dict:
        gate.wait(timeout=5.0)
        return {"ok": True}

    j1 = q.submit(blocker)
    j2 = q.submit(blocker)  # 1 running + 1 queued = 2 unfinished
    with pytest.raises(QueueFullError):
        q.submit(blocker)
    gate.set()
    assert q.wait(j1, timeout=5.0) and q.wait(j2, timeout=5.0)
    # once finished, capacity frees up again
    j3 = q.submit(lambda: {"ok": True})
    assert q.wait(j3, timeout=5.0)


# -- API: analyze/upload + jobs endpoints ---------------------------------------


def test_upload_job_windows_and_status_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    wav = _speech(6.0)
    r = client.post(
        "/api/analyze/upload",
        files={"file": ("clip.wav", wav_to_bytes(wav, TARGET_SR), "audio/wav")},
        data={"condition": "clean"},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    view = _wait_job(client, job_id)
    assert view["status"] == "done"
    assert "position" not in view  # position only exists while queued
    res = view["result"]
    assert res["type"] == "upload"
    assert res["n_windows"] == 2
    assert [w["start"] for w in res["windows"]] == [0.0, 2.5]
    assert res["verdict"] in {"spoof", "bonafide"}
    assert res["condition"] == "clean"

    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_upload_rejects_bad_extension_and_empty_file(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    wav_bytes = wav_to_bytes(_speech(3.0), TARGET_SR)
    r = client.post(
        "/api/analyze/upload",
        files={"file": ("clip.txt", wav_bytes, "text/plain")},
        data={"condition": "clean"},
    )
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]

    r_empty = client.post(
        "/api/analyze/upload",
        files={"file": ("clip.wav", b"", "audio/wav")},
        data={"condition": "clean"},
    )
    assert r_empty.status_code == 400


def test_upload_short_clip_job_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    wav = _speech(1.5)
    r = client.post(
        "/api/analyze/upload",
        files={"file": ("clip.wav", wav_to_bytes(wav, TARGET_SR), "audio/wav")},
    )
    assert r.status_code == 202
    view = _wait_job(client, r.json()["job_id"])
    assert view["status"] == "error"
    assert "need at least 2.0 s of audio" in view["error"]


def test_upload_mp3_decode_and_garbage_rejection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("soundfile")
    if not __import__("shutil").which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    client = _client(monkeypatch)

    # build a real mp3 with ffmpeg from a 6 s speech-like wav
    wav_path = tmp_path / "speech.wav"
    save_wav_mono(wav_path, _speech(6.0), TARGET_SR)
    mp3_path = tmp_path / "speech.mp3"
    conv = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-q:a", "4", str(mp3_path)],
        capture_output=True,
    )
    if conv.returncode != 0:
        pytest.skip("libmp3lame encoder unavailable")
    mp3_bytes = mp3_path.read_bytes()
    assert len(mp3_bytes) > 1000

    r = client.post(
        "/api/analyze/upload",
        files={"file": ("clip.mp3", mp3_bytes, "audio/mpeg")},
        data={"condition": "clean"},
    )
    assert r.status_code == 202
    view = _wait_job(client, r.json()["job_id"])
    assert view["status"] == "done", view.get("error")
    res = view["result"]
    # mp3 codec padding shifts duration slightly but cannot change the window count
    assert res["n_windows"] == 2
    assert [w["start"] for w in res["windows"]] == [0.0, 2.5]
    assert res["n_speech"] == 2
    assert res["type"] == "upload"

    # garbage bytes with an mp3 name must fail decode inside the job
    r_bad = client.post(
        "/api/analyze/upload",
        files={"file": ("clip.mp3", b"not-audio-at-all" * 100, "audio/mpeg")},
    )
    assert r_bad.status_code == 202
    view_bad = _wait_job(client, r_bad.json()["job_id"])
    assert view_bad["status"] == "error"
    assert view_bad["error"] == DECODE_ERROR_MESSAGE
    assert view_bad["error"] == "Could not decode this file. Try a WAV or MP3."


def test_sync_detect_file_matches_async_shape_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    wav = _speech(2.5)
    r = client.post(
        "/api/detect-file",
        files={"file": ("clip.wav", wav_to_bytes(wav, TARGET_SR), "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "upload"
    assert body["n_windows"] == 1
    assert body["verdict_rule"] == "2 of any 3 consecutive speech windows at or above the threshold"

    short = client.post(
        "/api/detect-file",
        files={"file": ("clip.wav", wav_to_bytes(_speech(1.0), TARGET_SR), "audio/wav")},
    )
    assert short.status_code == 400
    assert "need at least 2.0 s of audio" in short.json()["detail"]

    garbage = client.post(
        "/api/detect-file",
        files={"file": ("clip.mp3", b"garbage" * 200, "audio/mpeg")},
    )
    assert garbage.status_code == 400
    assert garbage.json()["detail"] == DECODE_ERROR_MESSAGE


def test_sync_detect_file_lenient_extension_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy detect-file contract never inspected the filename, only the
    bytes — valid audio with a missing/odd extension must still be analyzed,
    while the async upload endpoint enforces the extension allowlist."""
    client = _client(monkeypatch)
    wav_bytes = wav_to_bytes(_speech(2.5), TARGET_SR)

    for name in ("clip", "clip.bin", "clip.txt"):
        r = client.post(
            "/api/detect-file",
            files={"file": (name, wav_bytes, "application/octet-stream")},
        )
        assert r.status_code == 200, (name, r.json())
        assert r.json()["type"] == "upload"

        r_async = client.post(
            "/api/analyze/upload",
            files={"file": (name, wav_bytes, "application/octet-stream")},
        )
        assert r_async.status_code == 400, name


def test_analyze_sample_async_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLOAKBREAK_BACKEND", "mock")
    monkeypatch.setenv("CLOAKBREAK_WEB_DIST", "/nonexistent-web-dist")
    sdir = tmp_path / "samples"
    from app.main import SAMPLE_CLIPS

    for i, (sid, meta) in enumerate(SAMPLE_CLIPS.items()):
        p = sdir / meta["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        save_wav_mono(p, _speech(3.0, freq_hz=180 + i * 20, seed=i), TARGET_SR)
    monkeypatch.setenv("CLOAKBREAK_SAMPLES_DIR", str(sdir))
    client = TestClient(create_app(AppState()))

    sid = next(iter(SAMPLE_CLIPS))
    # json condition
    r = client.post(f"/api/analyze/sample/{sid}", json={"condition": "noise_10db"})
    assert r.status_code == 202
    view = _wait_job(client, r.json()["job_id"])
    assert view["status"] == "done"
    res = view["result"]
    assert res["type"] == "sample"
    assert res["sample_id"] == sid
    assert res["truth"] == SAMPLE_CLIPS[sid]["truth"]
    assert res["condition"] == "noise_10db"
    assert res["n_windows"] == 1

    # form condition also accepted
    r2 = client.post(f"/api/analyze/sample/{sid}", data={"condition": "clean"})
    assert r2.status_code == 202
    assert _wait_job(client, r2.json()["job_id"])["status"] == "done"

    # unknown sample -> 404 before any job is created
    assert client.post("/api/analyze/sample/nope").status_code == 404
    # invalid condition -> 422
    assert client.post(f"/api/analyze/sample/{sid}", json={"condition": "bogus"}).status_code == 422


# -- queue ordering / capacity ---------------------------------------------------


def test_queue_fifo_positions_and_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    _slow_mock_predict(monkeypatch, delay=0.3)
    wav_bytes = wav_to_bytes(_speech(6.0), TARGET_SR)  # 2 windows -> ~0.6 s of scoring per job

    def submit() -> str:
        r = client.post(
            "/api/analyze/upload",
            files={"file": ("clip.wav", wav_bytes, "audio/wav")},
        )
        assert r.status_code == 202
        return r.json()["job_id"]

    j1 = submit()
    # wait until the worker has actually picked up job 1
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if client.get(f"/api/jobs/{j1}").json()["status"] == "running":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("job 1 never started running")

    j2 = submit()
    j3 = submit()
    v2 = client.get(f"/api/jobs/{j2}").json()
    v3 = client.get(f"/api/jobs/{j3}").json()
    assert v2["status"] == "queued"
    assert v2["position"] >= 1  # next in line behind the running job
    assert v3["position"] == v2["position"] + 1

    done_order = []
    deadline = time.monotonic() + 30.0
    remaining = {j1, j2, j3}
    while remaining and time.monotonic() < deadline:
        for jid in list(remaining):
            if client.get(f"/api/jobs/{jid}").json()["status"] == "done":
                done_order.append(jid)
                remaining.discard(jid)
        time.sleep(0.05)
    assert done_order == [j1, j2, j3], "single worker must complete jobs in FIFO order"


def test_queue_eleventh_job_gets_429(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    _slow_mock_predict(monkeypatch, delay=0.3)
    wav_bytes = wav_to_bytes(_speech(2.5), TARGET_SR)

    statuses = []
    for _ in range(11):
        r = client.post(
            "/api/analyze/upload",
            files={"file": ("clip.wav", wav_bytes, "audio/wav")},
        )
        statuses.append(r.status_code)
    assert statuses[:10] == [202] * 10
    assert statuses[10] == 429


def test_sync_endpoint_waits_behind_running_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """FIFO enforcement for the synchronous compat path: while a slow async job
    runs, a detect-file call must wait behind it — total elapsed covers both."""
    client = _client(monkeypatch)
    _slow_mock_predict(monkeypatch, delay=0.3)
    wav_bytes = wav_to_bytes(_speech(6.0), TARGET_SR)  # ~0.6 s scoring + decode per job

    r = client.post(
        "/api/analyze/upload",
        files={"file": ("a.wav", wav_bytes, "audio/wav")},
    )
    assert r.status_code == 202

    t0 = time.monotonic()
    r_sync = client.post(
        "/api/detect-file",
        files={"file": ("b.wav", wav_bytes, "audio/wav")},
    )
    elapsed = time.monotonic() - t0
    assert r_sync.status_code == 200
    # one job alone costs ~0.6-0.8 s (decode + 2 x 0.3 s windows); if detect-file
    # did not queue behind the async job it would finish in that time. Waiting
    # behind the first job pushes it well past 1.0 s.
    assert elapsed >= 1.0, f"detect-file finished in {elapsed:.2f} s — did not wait behind the running job"


# -- analyze-url ------------------------------------------------------------------


def _gai_for(ip: str):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 443))]

    return fake_getaddrinfo


def _patch_yt_dlp(monkeypatch: pytest.MonkeyPatch, behavior: str, wav_bytes: bytes = b"") -> list:
    """Monkeypatch subprocess.run so yt-dlp invocations are faked; everything
    else (ffmpeg decode) passes through to the real subprocess.run."""
    real_run = subprocess.run
    calls: list[list] = []

    def fake_run(cmd, *args, **kwargs):
        if list(cmd[:3]) == [sys.executable, "-m", "yt_dlp"]:
            calls.append(list(cmd))
            if behavior == "timeout":
                raise subprocess.TimeoutExpired(cmd, 60)
            if behavior == "blocked":
                return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"blocked")
            # success: write the fake downloaded media at the -o template path
            out_tmpl = cmd[cmd.index("-o") + 1]
            out_path = out_tmpl.replace("%(ext)s", "wav")
            Path(out_path).write_bytes(wav_bytes)
            return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_analyze_url_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    monkeypatch.setattr(socket, "getaddrinfo", _gai_for("93.184.216.34"))
    wav_bytes = wav_to_bytes(_speech(6.0), TARGET_SR)
    calls = _patch_yt_dlp(monkeypatch, "success", wav_bytes)

    r = client.post("/api/analyze-url", json={"url": "https://video.example.com/watch?v=1", "condition": "clean"})
    assert r.status_code == 202, r.json()
    view = _wait_job(client, r.json()["job_id"])
    assert view["status"] == "done", view.get("error")
    res = view["result"]
    assert res["type"] == "url"
    assert res["url"] == "https://video.example.com/watch?v=1"
    assert res["n_windows"] == 2
    # the yt-dlp invocation shape is the documented one
    cmd = calls[0]
    assert cmd[:3] == [sys.executable, "-m", "yt_dlp"]
    for flag in ("--no-playlist", "--no-warnings", "-f", "ba/b", "--max-filesize", "50M", "--socket-timeout", "15", "--retries", "1"):
        assert flag in cmd, flag

    # form-encoded posts are accepted too
    calls.clear()
    r2 = client.post("/api/analyze-url", data={"url": "https://video.example.com/x", "condition": "clean"})
    assert r2.status_code == 202
    assert _wait_job(client, r2.json()["job_id"])["status"] == "done"


def test_analyze_url_blocked_site_exact_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    monkeypatch.setattr(socket, "getaddrinfo", _gai_for("93.184.216.34"))
    _patch_yt_dlp(monkeypatch, "blocked")

    r = client.post("/api/analyze-url", json={"url": "https://video.example.com/watch?v=1"})
    assert r.status_code == 202
    view = _wait_job(client, r.json()["job_id"])
    assert view["status"] == "error"
    assert view["error"] == "This site blocked our server. Download the video and upload the file instead."
    assert view["error"] == BLOCKED_ERROR_MESSAGE


def test_analyze_url_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    monkeypatch.setattr(socket, "getaddrinfo", _gai_for("93.184.216.34"))
    _patch_yt_dlp(monkeypatch, "timeout")

    r = client.post("/api/analyze-url", json={"url": "https://video.example.com/watch?v=1"})
    assert r.status_code == 202
    view = _wait_job(client, r.json()["job_id"])
    assert view["status"] == "error"
    assert view["error"] == TIMEOUT_ERROR_MESSAGE
    assert "timed out" in view["error"]


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5"])
def test_analyze_url_ssrf_private_ip_rejected(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    client = _client(monkeypatch)
    monkeypatch.setattr(socket, "getaddrinfo", _gai_for(ip))
    calls = _patch_yt_dlp(monkeypatch, "success", wav_to_bytes(_speech(3.0), TARGET_SR))

    r = client.post("/api/analyze-url", json={"url": "https://evil.example.com/watch"})
    assert r.status_code == 422
    assert r.json()["detail"] == "That address is not allowed."
    assert r.json()["detail"] == SSRF_ERROR_MESSAGE
    # the job is never created and the download is never attempted
    assert calls == []


def test_analyze_url_ssrf_scheme_userinfo_and_dns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    monkeypatch.setattr(socket, "getaddrinfo", _gai_for("93.184.216.34"))
    calls = _patch_yt_dlp(monkeypatch, "success", b"")

    for bad in (
        "ftp://video.example.com/x",
        "http://user@video.example.com/x",
        "http://user:pw@video.example.com/x",
        "not-a-url",
        "",
    ):
        r = client.post("/api/analyze-url", json={"url": bad, "condition": "clean"})
        assert r.status_code == 422, bad
    assert calls == []

    def gai_fail(host, port, *args, **kwargs):
        raise socket.gaierror("name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", gai_fail)
    r = client.post("/api/analyze-url", json={"url": "https://nonexistent.invalid/x"})
    assert r.status_code == 422
    assert r.json()["detail"] == "That address is not allowed."
    assert calls == []

    # missing url entirely
    assert client.post("/api/analyze-url", json={"condition": "clean"}).status_code == 422


# -- facts / honest-limits ---------------------------------------------------------


def test_facts_endpoint_values_from_results_files(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    r = client.get("/api/facts")
    assert r.status_code == 200
    facts = r.json()
    # values come from results/session2/eval_results.json — never hardcoded
    assert facts["headline_cb_eer"] == pytest.approx(5.67, abs=0.01)
    assert facts["headline_baseline_eer"] == pytest.approx(58.17, abs=0.01)
    assert facts["noise_10db_eer"] == pytest.approx(16.61, abs=0.01)
    assert facts["short_eer_0_5"] == pytest.approx(52.54, abs=0.01)
    assert facts["short_eer_0_8"] == pytest.approx(35.59, abs=0.01)
    # the 21.8%/26.7% headset false-alarm rates are derived from these fractions
    assert facts["livemic_frac_all"] * 100 == pytest.approx(21.8, abs=0.1)
    assert facts["livemic_frac_gated"] * 100 == pytest.approx(26.7, abs=0.1)


def test_honest_limits_endpoint_returns_readme_section(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    r = client.get("/api/honest-limits")
    assert r.status_code == 200
    md = r.json()["markdown"]
    assert isinstance(md, str)
    assert "Short clips are at chance" in md
    assert md.startswith("## Honest Limits")
    # stops at the next '## ' heading
    assert "## Repository Structure" not in md


# -- live-mic single session --------------------------------------------------------


def test_ws_second_concurrent_session_gets_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    with client.websocket_connect("/ws/detect") as ws1:
        # handshake with the first session so its connect handler is definitely
        # past the busy-check increment before the second client connects
        ws1.send_json({"condition": "clean"})
        assert ws1.receive_json()["type"] == "ack"

        with client.websocket_connect("/ws/detect") as ws2:
            msg = ws2.receive_json()
            assert msg["type"] == "busy"
            assert msg["detail"] == "Someone else is using the mic. Try again in a moment."

    # after the first session closes, a new one is accepted again (retry while
    # the server-side teardown of session 1 settles)
    deadline = time.monotonic() + 5.0
    while True:
        with client.websocket_connect("/ws/detect") as ws3:
            ws3.send_json({"condition": "clean"})
            msg = ws3.receive_json()
            if msg.get("type") == "ack":
                break
            assert msg.get("type") == "busy"
        if time.monotonic() > deadline:
            raise AssertionError("session slot was never released after disconnect")
        time.sleep(0.1)


# -- static SPA mount + CORS ---------------------------------------------------------


def test_static_mount_serves_web_dist_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLOAKBREAK_BACKEND", "mock")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>react-shell</body></html>", encoding="utf-8")
    (dist / "asset.js").write_text("console.log(1)", encoding="utf-8")
    monkeypatch.setenv("CLOAKBREAK_WEB_DIST", str(dist))
    client = TestClient(create_app(AppState()))

    r = client.get("/")
    assert r.status_code == 200
    assert "react-shell" in r.text
    # static assets serve under the mount
    assert client.get("/asset.js").status_code == 200
    # API routes still win over the catch-all mount
    assert client.get("/api/health").json()["backend"] == "mock"
    assert client.get("/api/facts").status_code == 200


def test_cors_only_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOAKBREAK_BACKEND", "mock")
    monkeypatch.delenv("CLOAKBREAK_CORS_ORIGIN", raising=False)
    plain = TestClient(create_app(AppState()))
    r = plain.get("/api/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in r.headers

    monkeypatch.setenv("CLOAKBREAK_CORS_ORIGIN", "https://demo.example.com")
    client = TestClient(create_app(AppState()))
    r2 = client.get("/api/health", headers={"Origin": "https://demo.example.com"})
    assert r2.headers.get("access-control-allow-origin") == "https://demo.example.com"
    preflight = client.options(
        "/api/health",
        headers={
            "Origin": "https://demo.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers.get("access-control-allow-origin") == "https://demo.example.com"
