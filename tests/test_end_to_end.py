from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import AppState, create_app
from src.audio_utils import (
    TARGET_SR,
    RollingAudioBuffer,
    float32_to_pcm16_bytes,
    load_wav_mono,
    pcm16_bytes_to_float32,
    rms_vad,
    save_wav_mono,
    wav_to_bytes,
)
from src.augment import TrainCodecAugmenter, apply_opus_codec, apply_phone_8k, degrade_for_eval
from src.eval import compute_eer_and_auroc, format_markdown_table, run_evaluation
from src.generate_spoofs import build_from_local_dir
from src.model import (
    CloakBreakDetector,
    MockDetectorRuntime,
    OnnxDetectorRuntime,
    TorchDetectorRuntime,
    export_to_onnx,
    quantize_onnx_int8,
)
from src.train import train_model


def _make_dummy_speech(duration_s: float = 2.5, freq_hz: float = 220.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(TARGET_SR * duration_s)
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float32)
    env = 0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
    sig = env * (0.3 * np.sin(2.0 * np.pi * freq_hz * t) + 0.15 * np.sin(2.0 * np.pi * (2.5 * freq_hz) * t))
    sig += rng.normal(0.0, 0.01, size=n).astype(np.float32)
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


@pytest.fixture()
def dummy_real_dir(tmp_path: Path) -> Path:
    real_dir = tmp_path / "real_wavs"
    real_dir.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        wav = _make_dummy_speech(duration_s=2.5, freq_hz=150.0 + i * 15.0, seed=i)
        save_wav_mono(real_dir / f"spk{i:02d}_utt0.wav", wav, TARGET_SR)
    return real_dir


def test_audio_utils_and_rolling_buffer(tmp_path: Path) -> None:
    wav = _make_dummy_speech(1.0)
    p = tmp_path / "sample.wav"
    save_wav_mono(p, wav, TARGET_SR)
    loaded, sr = load_wav_mono(p, TARGET_SR)
    assert sr == TARGET_SR
    assert loaded.shape == wav.shape
    assert rms_vad(loaded) is True
    assert rms_vad(np.zeros(TARGET_SR, dtype=np.float32)) is False

    pcm = float32_to_pcm16_bytes(wav[:4000])
    roundtrip = pcm16_bytes_to_float32(pcm)
    assert roundtrip.shape == (4000,)
    assert np.allclose(roundtrip, wav[:4000], atol=1e-3)

    ring = RollingAudioBuffer(window_sec=2.5, stride_sec=0.25, sr=TARGET_SR)
    assert ring.push(wav[:2000]) is False
    assert ring.push(wav[2000:4000]) is True
    assert ring.get_window().shape == (int(2.5 * TARGET_SR),)


def test_augmentations_including_opus_and_phone() -> None:
    wav = _make_dummy_speech(2.0)
    phone = apply_phone_8k(wav, TARGET_SR)
    assert phone.shape == wav.shape
    assert np.isfinite(phone).all()

    opus = apply_opus_codec(wav, TARGET_SR, bitrate_kbps=16)
    assert opus.shape == wav.shape
    assert np.isfinite(opus).all()

    short = degrade_for_eval(wav, "short_0.5s", TARGET_SR)
    assert short.shape == (int(0.5 * TARGET_SR),)

    aug = TrainCodecAugmenter(p_codec=1.0, p_noise=1.0, p_clip=1.0, seed=7)
    out = aug(wav, TARGET_SR)
    assert out.shape == wav.shape
    assert float(np.max(np.abs(out))) <= 1.0


def test_generate_train_export_onnx_and_eval_end_to_end(dummy_real_dir: Path, tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    manifest = build_from_local_dir(
        real_dir=dummy_real_dir,
        out_dir=data_dir,
        max_chatterbox=12,
        max_kokoro=6,
        mock_generator=True,
    )
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 12 + 12 + 6
    kokoro_rows = [r for r in rows if "kokoro" in r["generator"]]
    assert len(kokoro_rows) == 6
    assert all(r["split"] == "test" for r in kokoro_rows)

    # Verify resumability: re-running skips existing WAVs without rewriting them
    first_cb = data_dir / "chatterbox" / "cb_00000.wav"
    mtime_before = first_cb.stat().st_mtime_ns
    manifest_resumed = build_from_local_dir(
        real_dir=dummy_real_dir,
        out_dir=data_dir,
        max_chatterbox=12,
        max_kokoro=6,
        mock_generator=True,
    )
    assert manifest_resumed == manifest
    assert first_cb.stat().st_mtime_ns == mtime_before

    ckpt_dir = tmp_path / "checkpoints"
    summary = train_model(
        manifest_path=manifest,
        out_dir=ckpt_dir,
        epochs_stage1=1,
        epochs_stage2=1,
        batch_size=4,
        use_tiny_test_backbone=True,
        allow_opus=False,
        device_str="cpu",
    )
    assert Path(summary["best_ckpt"]).exists()
    assert Path(summary["fp32_onnx"]).exists()
    assert Path(summary["int8_onnx"]).exists()

    torch_rt = TorchDetectorRuntime(summary["best_ckpt"], device="cpu")
    onnx_rt = OnnxDetectorRuntime(summary["int8_onnx"])
    sample = _make_dummy_speech(2.5)
    t_res = torch_rt.predict(sample)
    o_res = onnx_rt.predict(sample)
    assert 0.0 <= t_res.spoof_prob <= 1.0
    assert 0.0 <= o_res.spoof_prob <= 1.0
    assert len(o_res.attention_focus) == 32
    assert o_res.latency_ms >= 0.0

    eval_json = tmp_path / "eval_results.json"
    results = run_evaluation(manifest, detector=onnx_rt, max_per_slice=12, out_json=eval_json)
    assert eval_json.exists()
    assert "test_clean" in results
    assert "test_opus_16k" in results
    assert "test_short_0.5s" in results
    assert "unseen_kokoro_clean" in results
    md = format_markdown_table(results)
    assert "unseen_kokoro_clean" in md


def test_eer_and_auroc_math() -> None:
    perfect = compute_eer_and_auroc([0, 0, 0, 1, 1, 1], [0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    assert perfect["eer_pct"] == 0.0
    assert perfect["auroc"] == 1.0

    inverted = compute_eer_and_auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert inverted["auroc"] == 0.0


def test_fastapi_mock_ws_and_clone_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOAKBREAK_BACKEND", "mock")
    monkeypatch.setenv("ENABLE_CLONE_ENDPOINT", "false")
    client = TestClient(create_app(AppState()))

    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "CloakBreak" in r_root.text
    assert "Model Attention Focus" in r_root.text

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    body = r_health.json()
    assert body["backend"] == "mock"
    assert body["clone_enabled"] is False

    wav = _make_dummy_speech(2.5)
    wav_bytes = wav_to_bytes(wav, TARGET_SR)
    r_det = client.post(
        "/api/detect-file",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        data={"condition": "opus_16k"},
    )
    assert r_det.status_code == 200
    det = r_det.json()
    assert 0.0 <= det["spoof_prob"] <= 1.0
    assert len(det["attention_focus"]) == 32
    assert det["condition"] == "opus_16k"

    # Clone endpoint must be disabled by default
    r_clone_disabled = client.post(
        "/api/clone",
        files={"reference_wav": ("ref.wav", wav_bytes, "audio/wav")},
        data={"text": "Hello test"},
    )
    assert r_clone_disabled.status_code == 403

    # WebSocket streaming: silence then voiced speech
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_json({"condition": "phone_8k"})
        ack = ws.receive_json()
        assert ack["type"] == "ack" and ack["condition"] == "phone_8k"

        silence_pcm = float32_to_pcm16_bytes(np.zeros(4000, dtype=np.float32))
        ws.send_bytes(silence_pcm)
        msg_sil = ws.receive_json()
        assert msg_sil["is_speech"] is False
        assert msg_sil["spoof_prob"] is None

        speech_pcm = float32_to_pcm16_bytes(wav[:4000])
        ws.send_bytes(speech_pcm)
        msg_sp = ws.receive_json()
        assert msg_sp["is_speech"] is True
        assert 0.0 <= msg_sp["spoof_prob"] <= 1.0
        assert len(msg_sp["attention_focus"]) == 32


def test_fastapi_private_clone_auth_and_onnx_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = CloakBreakDetector(use_tiny_test_backbone=True)
    fp32 = export_to_onnx(model, tmp_path / "m_fp32.onnx")
    int8 = quantize_onnx_int8(fp32, tmp_path / "m_int8.onnx")

    monkeypatch.setenv("CLOAKBREAK_BACKEND", "onnx")
    monkeypatch.setenv("CLOAKBREAK_MODEL_PATH", str(int8))
    monkeypatch.setenv("ENABLE_CLONE_ENDPOINT", "true")
    monkeypatch.setenv("CLOAKBREAK_CLONE_TOKEN", "secret-owner-token")
    client = TestClient(create_app(AppState()))

    h = client.get("/api/health").json()
    assert h["backend"].startswith("onnx:")
    assert h["clone_enabled"] is True

    wav_bytes = wav_to_bytes(_make_dummy_speech(2.5), TARGET_SR)
    # Missing or wrong token rejected
    assert client.post(
        "/api/clone",
        files={"reference_wav": ("ref.wav", wav_bytes, "audio/wav")},
        data={"text": "Owner self test"},
    ).status_code == 401
    assert client.post(
        "/api/clone",
        files={"reference_wav": ("ref.wav", wav_bytes, "audio/wav")},
        data={"text": "Owner self test"},
        headers={"Authorization": "Bearer wrong-token"},
    ).status_code == 401

    # Valid ONNX detection via FastAPI
    r_det = client.post(
        "/api/detect-file",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        data={"condition": "clean"},
    )
    assert r_det.status_code == 200
    assert r_det.json()["backend"].startswith("onnx:")

    # Valid authorized clone in mock mode returns decodable 16 kHz WAV
    monkeypatch.setenv("CLOAKBREAK_BACKEND", "mock")
    mock_client = TestClient(create_app(AppState()))
    r_ok = mock_client.post(
        "/api/clone",
        files={"reference_wav": ("ref.wav", wav_bytes, "audio/wav")},
        data={"text": "Owner self test", "condition": "phone_8k"},
        headers={"Authorization": "Bearer secret-owner-token"},
    )
    assert r_ok.status_code == 200
    assert r_ok.headers["content-type"] == "audio/wav"
    assert len(r_ok.content) > 1000
