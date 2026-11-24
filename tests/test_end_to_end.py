from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import AppState, create_app
from src.audio_utils import (
    TARGET_RMS,
    TARGET_SR,
    RollingAudioBuffer,
    audio_rms,
    float32_to_pcm16_bytes,
    load_wav_mono,
    normalize_loudness,
    pcm16_bytes_to_float32,
    rms_vad,
    save_wav_mono,
    wav_to_bytes,
)
from src.augment import TrainCodecAugmenter, apply_opus_codec, apply_phone_8k, degrade_for_eval, precompute_opus_cache
from src.eval import compute_eer_and_auroc, format_markdown_table, run_evaluation, run_shortcut_audit
from src.generate_spoofs import (
    DEMO_PROMPTS,
    assert_disjoint_text_splits,
    build_from_local_dir,
    normalize_sentence_key,
    summarize_manifest_counts,
)
from src.model import (
    CloakBreakDetector,
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
    for i in range(16):
        wav = _make_dummy_speech(duration_s=2.5, freq_hz=150.0 + i * 15.0, seed=i)
        save_wav_mono(real_dir / f"spk{i:02d}_utt0.wav", wav, TARGET_SR)
    return real_dir


def test_audio_utils_loudness_and_rolling_buffer(tmp_path: Path) -> None:
    wav = _make_dummy_speech(1.0)
    quiet = normalize_loudness(wav * 0.05)
    loud = normalize_loudness(wav * 0.9)
    assert abs(audio_rms(quiet) - TARGET_RMS) < 0.005
    assert abs(audio_rms(loud) - TARGET_RMS) < 0.005

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


def test_augmentations_and_opus_precompute_cache(tmp_path: Path) -> None:
    wav = _make_dummy_speech(2.0)
    phone = apply_phone_8k(wav, TARGET_SR)
    assert phone.shape == wav.shape
    assert np.isfinite(phone).all()

    opus = apply_opus_codec(wav, TARGET_SR, bitrate_kbps=16)
    assert opus.shape == wav.shape
    assert np.isfinite(opus).all()

    short = degrade_for_eval(wav, "short_0.5s", TARGET_SR)
    assert short.shape == (int(0.5 * TARGET_SR),)

    src_wav = tmp_path / "src.wav"
    save_wav_mono(src_wav, wav, TARGET_SR)
    cache_dir = tmp_path / "opus_cache"
    mapping = precompute_opus_cache([str(src_wav)], cache_dir=cache_dir, bitrate_kbps=16, num_workers=1)
    cached_path = Path(mapping[str(src_wav)])
    assert cached_path.exists()
    mtime_1 = cached_path.stat().st_mtime_ns
    # Re-running cache must skip already-cached file
    precompute_opus_cache([str(src_wav)], cache_dir=cache_dir, bitrate_kbps=16, num_workers=1)
    assert cached_path.stat().st_mtime_ns == mtime_1

    cached_wav, _ = load_wav_mono(cached_path, TARGET_SR)
    aug = TrainCodecAugmenter(p_codec=1.0, p_noise=1.0, p_clip=1.0, seed=7)
    out = aug(wav, TARGET_SR, cached_opus_waveform=cached_wav)
    assert out.shape == wav.shape
    assert abs(audio_rms(out) - TARGET_RMS) < 0.01


def test_disjoint_kokoro_and_chatterbox_sentences(dummy_real_dir: Path, tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    manifest = build_from_local_dir(
        real_dir=dummy_real_dir,
        out_dir=data_dir,
        max_chatterbox=12,
        max_kokoro=8,
        mock_generator=True,
    )
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_val_texts = {
        normalize_sentence_key(r["text"])
        for r in rows
        if r.get("split") in {"train", "val"} and r.get("text")
    }
    cb_texts = {
        normalize_sentence_key(r["text"])
        for r in rows
        if "chatterbox" in r["generator"] and r.get("text")
    }
    kokoro_texts = {
        normalize_sentence_key(r["text"])
        for r in rows
        if "kokoro" in r["generator"] and r.get("text")
    }
    demo_texts = {normalize_sentence_key(p) for p in DEMO_PROMPTS}

    assert len(cb_texts) == 12
    assert len(kokoro_texts) == 8
    assert kokoro_texts.isdisjoint(train_val_texts)
    assert kokoro_texts.isdisjoint(cb_texts)
    assert cb_texts.isdisjoint(demo_texts)
    assert kokoro_texts.isdisjoint(demo_texts)
    assert all(r["watermarked"] is False for r in rows if "chatterbox" in r["generator"])

    # Inject a leaked sentence and confirm the guard catches it
    leaked_rows = [dict(r) for r in rows]
    sample_train_text = next(r["text"] for r in rows if "chatterbox" in r["generator"])
    for r in leaked_rows:
        if "kokoro" in r["generator"]:
            r["text"] = sample_train_text
            break
    with pytest.raises(ValueError, match="Sentence leakage"):
        assert_disjoint_text_splits(leaked_rows)


def test_shortcut_audit_catches_biased_and_clears_normalized(tmp_path: Path) -> None:
    # 1. Deliberately biased duration/RMS dataset must trigger warning
    biased_dir = tmp_path / "biased"
    biased_rows = []
    for split in ("train", "test"):
        for i in range(10):
            r_wav = _make_dummy_speech(4.0, freq_hz=180 + i * 5, seed=i) * 0.9
            f_wav = _make_dummy_speech(1.5, freq_hz=180 + i * 5, seed=100 + i) * 0.08
            rp = biased_dir / f"{split}_real_{i}.wav"
            fp = biased_dir / f"{split}_fake_{i}.wav"
            save_wav_mono(rp, r_wav, TARGET_SR)
            save_wav_mono(fp, f_wav, TARGET_SR)
            biased_rows.append({"path": str(rp.relative_to(biased_dir)), "label": 0, "generator": "bonafide_wild", "split": split})
            biased_rows.append({"path": str(fp.relative_to(biased_dir)), "label": 1, "generator": "spoof", "split": split})
    biased_manifest = biased_dir / "manifest.jsonl"
    biased_manifest.write_text("\n".join(json.dumps(r) for r in biased_rows) + "\n", encoding="utf-8")

    biased_audit = run_shortcut_audit(biased_manifest)
    assert biased_audit["raw_acoustic_shortcuts"]["shortcut_warning"] is True
    assert biased_audit["raw_acoustic_shortcuts"]["auroc"] >= 0.90

    # 2. Channel-mismatch shortcut (identical 2.5s duration + identical RMS, but noisy/narrow real vs clean/wide fake)
    chan_dir = tmp_path / "channel_mismatch"
    chan_rows = []
    for split_idx, split in enumerate(("train", "test")):
        for i in range(10):
            base_seed = split_idx * 1000 + i
            rng = np.random.default_rng(base_seed)
            # Noisy narrowband real (phone_8k + noise floor)
            r_raw = apply_phone_8k(_make_dummy_speech(2.5, freq_hz=200 + i * 5, seed=base_seed), TARGET_SR)
            r_wav = normalize_loudness(r_raw + rng.normal(0.0, 0.04, size=r_raw.shape[0]).astype(np.float32))
            # Clean wideband fake (zero noise floor, high-freq harmonic up to 6 kHz)
            t = np.linspace(0.0, 2.5, int(TARGET_SR * 2.5), endpoint=False, dtype=np.float32)
            gate = (np.sin(2.0 * np.pi * 1.5 * t) > 0.0).astype(np.float32)
            f_wav = normalize_loudness(gate * (0.25 * np.sin(2.0 * np.pi * 240.0 * t) + 0.25 * np.sin(2.0 * np.pi * 5200.0 * t)))
            rp = chan_dir / f"{split}_real_{i}.wav"
            fp = chan_dir / f"{split}_fake_{i}.wav"
            save_wav_mono(rp, r_wav, TARGET_SR)
            save_wav_mono(fp, f_wav, TARGET_SR)
            chan_rows.append({"path": str(rp.relative_to(chan_dir)), "label": 0, "generator": "bonafide_wild", "split": split})
            chan_rows.append({"path": str(fp.relative_to(chan_dir)), "label": 1, "generator": "spoof", "split": split})
    chan_manifest = chan_dir / "manifest.jsonl"
    chan_manifest.write_text("\n".join(json.dumps(r) for r in chan_rows) + "\n", encoding="utf-8")

    chan_audit = run_shortcut_audit(chan_manifest)
    assert chan_audit["model_window_shortcuts"]["shortcut_warning"] is True
    assert chan_audit["model_window_shortcuts"]["auroc"] >= 0.90

    # 3. Matched acoustic distribution (same duration, RMS, SNR, bandwidth) must NOT trigger shortcut
    clean_dir = tmp_path / "clean"
    clean_rows = []
    for split_idx, split in enumerate(("train", "test")):
        for i in range(12):
            base_seed = split_idx * 1000 + i
            r_wav = normalize_loudness(_make_dummy_speech(2.5, freq_hz=170 + i * 7, seed=base_seed))
            f_wav = normalize_loudness(_make_dummy_speech(2.5, freq_hz=170 + i * 7, seed=base_seed + 500))
            rp = clean_dir / f"{split}_real_{i}.wav"
            fp = clean_dir / f"{split}_fake_{i}.wav"
            save_wav_mono(rp, r_wav, TARGET_SR)
            save_wav_mono(fp, f_wav, TARGET_SR)
            clean_rows.append({"path": str(rp.relative_to(clean_dir)), "label": 0, "generator": "bonafide_clean", "split": split})
            clean_rows.append({"path": str(fp.relative_to(clean_dir)), "label": 1, "generator": "spoof", "split": split})
    clean_manifest = clean_dir / "manifest.jsonl"
    clean_manifest.write_text("\n".join(json.dumps(r) for r in clean_rows) + "\n", encoding="utf-8")

    clean_audit = run_shortcut_audit(clean_manifest)
    assert clean_audit["model_window_shortcuts"]["shortcut_warning"] is False


def test_generate_train_export_onnx_and_eval_end_to_end(dummy_real_dir: Path, tmp_path: Path) -> None:
    clean_real_dir = tmp_path / "clean_real_wavs"
    clean_real_dir.mkdir(parents=True, exist_ok=True)
    for i in range(16):
        wav = _make_dummy_speech(duration_s=2.5, freq_hz=160.0 + i * 12.0, seed=500 + i)
        save_wav_mono(clean_real_dir / f"cln{i:02d}_utt0.wav", wav, TARGET_SR)

    data_dir = tmp_path / "dataset"
    manifest = build_from_local_dir(
        real_dir=dummy_real_dir,
        clean_real_dir=clean_real_dir,
        out_dir=data_dir,
        max_chatterbox=12,
        max_kokoro=6,
        mock_generator=True,
    )
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 16 + 16 + 12 + 6
    counts = summarize_manifest_counts(rows)
    assert sum(counts[s]["total"] for s in ("train", "val", "test")) == len(rows)
    kokoro_rows = [r for r in rows if "kokoro" in r["generator"]]
    assert len(kokoro_rows) == 6
    assert all(r["split"] == "test" for r in kokoro_rows)

    # Verify resumability: re-running skips existing WAVs without rewriting them
    first_cb = data_dir / "chatterbox" / "cb_00000.wav"
    mtime_before = first_cb.stat().st_mtime_ns
    manifest_resumed = build_from_local_dir(
        real_dir=dummy_real_dir,
        clean_real_dir=clean_real_dir,
        out_dir=data_dir,
        max_chatterbox=12,
        max_kokoro=6,
        mock_generator=True,
    )
    assert manifest_resumed == manifest
    assert first_cb.stat().st_mtime_ns == mtime_before

    # Train using precomputed Opus cache + multi-worker DataLoader (num_workers=2)
    ckpt_dir = tmp_path / "checkpoints"
    summary = train_model(
        manifest_path=manifest,
        out_dir=ckpt_dir,
        epochs_stage1=1,
        epochs_stage2=1,
        batch_size=4,
        use_tiny_test_backbone=True,
        precompute_opus=True,
        num_workers=2,
        device_str="cpu",
    )
    assert Path(summary["best_ckpt"]).exists()
    assert Path(summary["fp32_onnx"]).exists()
    assert Path(summary["int8_onnx"]).exists()
    assert summary["history"][0]["num_workers"] == 2
    assert summary["history"][0]["steps_per_sec"] > 0

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
    assert "shortcut_audit" in results
    assert results["shortcut_audit"]["disjoint_text_check"] == "PASSED"
    assert "test_clean" in results
    assert "test_opus_16k" in results
    assert "test_short_0.5s" in results
    assert "unseen_kokoro_clean" in results
    assert "clean_real_vs_kokoro_clean" in results
    assert "clean_real_vs_kokoro_opus_16k" in results
    assert "clean_real_fpr" in results
    assert 0.0 <= results["clean_real_fpr"]["cloakbreak_fpr_at_05_pct"] <= 100.0
    md = format_markdown_table(results)
    assert "clean_real_vs_kokoro_clean" in md
    assert "Clean human speech FPR@0.5" in md
    assert "shortcut_audit" not in md


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

    assert client.post(
        "/api/clone",
        files={"reference_wav": ("ref.wav", wav_bytes, "audio/wav")},
        data={"text": "Hello test"},
    ).status_code == 403

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

    r_det = client.post(
        "/api/detect-file",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        data={"condition": "clean"},
    )
    assert r_det.status_code == 200
    assert r_det.json()["backend"].startswith("onnx:")

    monkeypatch.setenv("CLOAKBREAK_BACKEND", "mock")
    mock_client = TestClient(create_app(AppState()))
    # Short 2.5s reference must be rejected (< 5.5s for Chatterbox-Turbo)
    r_short = mock_client.post(
        "/api/clone",
        files={"reference_wav": ("ref.wav", wav_bytes, "audio/wav")},
        data={"text": "Owner self test", "condition": "phone_8k"},
        headers={"Authorization": "Bearer secret-owner-token"},
    )
    assert r_short.status_code == 400

    # Valid 6.0s reference succeeds
    ref6_bytes = wav_to_bytes(_make_dummy_speech(6.0), TARGET_SR)
    r_ok = mock_client.post(
        "/api/clone",
        files={"reference_wav": ("ref6.wav", ref6_bytes, "audio/wav")},
        data={"text": "Owner self test", "condition": "phone_8k"},
        headers={"Authorization": "Bearer secret-owner-token"},
    )
    assert r_ok.status_code == 200
    assert r_ok.headers["content-type"] == "audio/wav"
    assert len(r_ok.content) > 1000


def test_cpu_smoke_pipeline_mirrors_vast_smoke(tmp_path: Path) -> None:
    """Runs the exact Step 2b smoke pipeline on CPU with synthetic audio (40/20/6/4, 20 steps, ONNX eval)."""
    wild_dir = tmp_path / "wild_40"
    clean_dir = tmp_path / "clean_20"
    wild_dir.mkdir(parents=True)
    clean_dir.mkdir(parents=True)
    for i in range(40):
        # Mix 6s refs matching Chatterbox-Turbo > 5.0s requirement
        dur = 6.0 if i % 2 == 0 else 3.0
        save_wav_mono(wild_dir / f"wspk{i:02d}_utt0.wav", _make_dummy_speech(dur, freq_hz=150 + i * 3, seed=i), TARGET_SR)
    for i in range(20):
        dur = 6.0 if i % 2 == 0 else 3.0
        save_wav_mono(clean_dir / f"cspk{i:02d}_utt0.wav", _make_dummy_speech(dur, freq_hz=160 + i * 4, seed=500 + i), TARGET_SR)

    data_dir = tmp_path / "smoke_v1"
    manifest = build_from_local_dir(
        real_dir=wild_dir,
        clean_real_dir=clean_dir,
        out_dir=data_dir,
        max_chatterbox=6,
        max_kokoro=4,
        mock_generator=True,
    )
    audit = run_shortcut_audit(manifest, max_per_split=100)
    assert audit["disjoint_text_check"] == "PASSED"

    ckpt_dir = tmp_path / "smoke_ckpt"
    summary = train_model(
        manifest_path=manifest,
        out_dir=ckpt_dir,
        epochs_stage1=1,
        epochs_stage2=2,
        batch_size=4,
        max_steps=20,
        use_tiny_test_backbone=True,
        precompute_opus=False,
        num_workers=0,
        device_str="cpu",
    )
    assert summary["history"][-1]["global_steps"] == 20
    assert Path(summary["fp32_onnx"]).exists()
    assert Path(summary["int8_onnx"]).exists()

    eval_out = tmp_path / "smoke_results.json"
    onnx_rt = OnnxDetectorRuntime(summary["int8_onnx"])
    res = run_evaluation(manifest, detector=onnx_rt, max_per_slice=20, out_json=eval_out)
    assert eval_out.exists()
    assert "clean_real_vs_kokoro_clean" in res
    assert "clean_real_fpr" in res


def test_eval_cli_fail_on_shortcut_flag(tmp_path: Path) -> None:
    biased_dir = tmp_path / "biased_cli"
    rows = []
    for split in ("train", "test"):
        for i in range(8):
            r_wav = _make_dummy_speech(4.0, freq_hz=190 + i * 4, seed=i) * 0.9
            f_wav = _make_dummy_speech(1.2, freq_hz=190 + i * 4, seed=100 + i) * 0.08
            rp = biased_dir / f"{split}_r_{i}.wav"
            fp = biased_dir / f"{split}_f_{i}.wav"
            save_wav_mono(rp, r_wav, TARGET_SR)
            save_wav_mono(fp, f_wav, TARGET_SR)
            rows.append({"path": str(rp.relative_to(biased_dir)), "label": 0, "generator": "bonafide_wild", "split": split})
            rows.append({"path": str(fp.relative_to(biased_dir)), "label": 1, "generator": "spoof", "split": split})
    manifest = biased_dir / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    env_strict = {k: v for k, v in os.environ.items() if k != "ALLOW_SHORTCUT_WARNING"}
    p_fail = subprocess.run(
        [sys.executable, "-m", "src.eval", "--shortcut-only", "--fail-on-shortcut", "--manifest", str(manifest)],
        env=env_strict,
        capture_output=True,
        text=True,
    )
    assert p_fail.returncode == 2

    env_allow = {**env_strict, "ALLOW_SHORTCUT_WARNING": "1"}
    p_ok = subprocess.run(
        [sys.executable, "-m", "src.eval", "--shortcut-only", "--fail-on-shortcut", "--manifest", str(manifest)],
        env=env_allow,
        capture_output=True,
        text=True,
    )
    assert p_ok.returncode == 0
