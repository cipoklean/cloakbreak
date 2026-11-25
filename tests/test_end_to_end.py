from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import AppState, create_app
from scripts import hf_auth_check
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
    spectral_rolloff_hz,
    wav_to_bytes,
)
from src.augment import (
    TrainCodecAugmenter,
    _sample_spectral_params,
    apply_opus_codec,
    apply_phone_8k,
    apply_spectral_augment,
    degrade_for_eval,
    precompute_opus_cache,
)
from src.eval import (
    build_rolloff_matched_pair,
    compute_eer_and_auroc,
    format_markdown_table,
    run_augmented_shortcut_audit,
    run_evaluation,
    run_shortcut_audit,
)
from src.generate_spoofs import (
    DEMO_PROMPTS,
    assert_disjoint_text_splits,
    build_from_local_dir,
    normalize_sentence_key,
    summarize_manifest_counts,
)
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

        # Below the 2.0 s minimum window the detector must not run: every
        # stride emits warming_up with spoof_prob null until 2.0 s buffered.
        silence_pcm = float32_to_pcm16_bytes(np.zeros(4000, dtype=np.float32))
        ws.send_bytes(silence_pcm)
        msg_sil = ws.receive_json()
        assert msg_sil["warming_up"] is True
        assert msg_sil["spoof_prob"] is None
        assert msg_sil["threshold"] == 0.5
        assert msg_sil["verdict"] is None
        assert msg_sil["window_sec"] == pytest.approx(0.25)

        speech_pcm = float32_to_pcm16_bytes(wav[:4000])
        for _ in range(6):  # 0.25s silence + 6*0.25s speech = 1.75s -> still warming
            ws.send_bytes(speech_pcm)
            msg_w = ws.receive_json()
            assert msg_w["warming_up"] is True
            assert msg_w["spoof_prob"] is None

        ws.send_bytes(speech_pcm)  # 8th chunk -> 2.0s buffered -> prediction
        msg_sp = ws.receive_json()
        assert msg_sp.get("warming_up") is not True
        assert msg_sp["is_speech"] is True
        assert 0.0 <= msg_sp["spoof_prob"] <= 1.0
        assert msg_sp["verdict"] in {"spoof", "bonafide"}
        assert msg_sp["threshold"] == 0.5
        assert len(msg_sp["attention_focus"]) == 32


def test_app_threshold_precedence_verdict_and_min_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hashlib

    monkeypatch.setenv("CLOAKBREAK_BACKEND", "mock")
    monkeypatch.setenv("ENABLE_CLONE_ENDPOINT", "false")
    monkeypatch.delenv("CLOAKBREAK_THRESHOLD", raising=False)
    monkeypatch.delenv("CLOAKBREAK_THRESHOLD_FILE", raising=False)
    monkeypatch.delenv("CLOAKBREAK_MODEL_PATH", raising=False)

    # calibrated file pinned to a model sha, but no model loaded (mock) -> not applied
    pinned = tmp_path / "pinned.json"
    pinned.write_text(json.dumps({"operating_threshold": 0.9, "model_sha256": "a" * 64}))
    monkeypatch.setenv("CLOAKBREAK_THRESHOLD_FILE", str(pinned))
    st = AppState()
    assert st.threshold == 0.5 and st.threshold_source == "default_0.5_model_mismatch"

    # default: no env, no file -> 0.5 / default_0.5
    monkeypatch.setenv("CLOAKBREAK_THRESHOLD_FILE", str(tmp_path / "missing.json"))
    st = AppState()
    assert st.threshold == 0.5 and st.threshold_source == "default_0.5"

    # env wins over file
    tfile = tmp_path / "thr.json"
    tfile.write_text(json.dumps({"operating_threshold": 0.42, "model_sha256": None}))
    monkeypatch.setenv("CLOAKBREAK_THRESHOLD_FILE", str(tfile))
    monkeypatch.setenv("CLOAKBREAK_THRESHOLD", "0.33")
    st = AppState()
    assert st.threshold == 0.33 and st.threshold_source == "env"

    # file used when env absent
    monkeypatch.delenv("CLOAKBREAK_THRESHOLD")
    st = AppState()
    assert st.threshold == 0.42 and st.threshold_source == "file"

    # sha mismatch vs CLOAKBREAK_MODEL_PATH -> 0.5 default_0.5_model_mismatch
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"fake-model-bytes")
    real_sha = hashlib.sha256(b"fake-model-bytes").hexdigest()
    tfile.write_text(json.dumps({"operating_threshold": 0.42, "model_sha256": "0" * 64}))
    monkeypatch.setenv("CLOAKBREAK_MODEL_PATH", str(model_file))
    st = AppState()
    assert st.threshold == 0.5 and st.threshold_source == "default_0.5_model_mismatch"

    # sha match -> file threshold honored
    tfile.write_text(json.dumps({"operating_threshold": 0.42, "model_sha256": real_sha}))
    st = AppState()
    assert st.threshold == 0.42 and st.threshold_source == "file"

    # min window clamps to >= 2.0 and honors larger env values
    monkeypatch.setenv("CLOAKBREAK_MIN_WINDOW_SEC", "1.0")
    assert AppState().min_window_sec == 2.0
    monkeypatch.setenv("CLOAKBREAK_MIN_WINDOW_SEC", "3.5")
    assert AppState().min_window_sec == 3.5
    monkeypatch.delenv("CLOAKBREAK_MIN_WINDOW_SEC")

    # health exposes threshold + source; detect-file adds threshold + verdict
    monkeypatch.setenv("CLOAKBREAK_THRESHOLD", "0.33")
    client = TestClient(create_app(AppState()))
    h = client.get("/api/health").json()
    assert h["threshold"] == 0.33 and h["threshold_source"] == "env"

    wav_bytes = wav_to_bytes(_make_dummy_speech(2.5), TARGET_SR)
    r = client.post(
        "/api/detect-file",
        files={"file": ("t.wav", wav_bytes, "audio/wav")},
        data={"condition": "clean"},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["threshold"] == 0.33
    assert d["verdict"] in {"spoof", "bonafide"}
    # verdict must be consistent with the reported prob and threshold
    assert d["verdict"] == ("spoof" if d["spoof_prob"] >= d["threshold"] else "bonafide")

    # mock prob is in [0.02, 0.98]: 0.99 -> bonafide, 0.01 -> spoof
    monkeypatch.setenv("CLOAKBREAK_THRESHOLD", "0.99")
    hi = TestClient(create_app(AppState())).post(
        "/api/detect-file",
        files={"file": ("t.wav", wav_bytes, "audio/wav")},
        data={"condition": "clean"},
    ).json()
    assert hi["verdict"] == "bonafide"
    monkeypatch.setenv("CLOAKBREAK_THRESHOLD", "0.01")
    lo = TestClient(create_app(AppState())).post(
        "/api/detect-file",
        files={"file": ("t.wav", wav_bytes, "audio/wav")},
        data={"condition": "clean"},
    ).json()
    assert lo["verdict"] == "spoof"

    # detect-file rejects audio under the 2.0 s minimum window
    short_bytes = wav_to_bytes(_make_dummy_speech(1.0), TARGET_SR)
    r_short = client.post(
        "/api/detect-file",
        files={"file": ("s.wav", short_bytes, "audio/wav")},
        data={"condition": "clean"},
    )
    assert r_short.status_code == 400
    assert "need at least 2.0 s of audio" in r_short.json()["detail"]


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
    results_dir = tmp_path / "cli_results"
    p_fail = subprocess.run(
        [
            sys.executable, "-m", "src.eval", "--shortcut-only", "--fail-on-shortcut",
            "--manifest", str(manifest), "--results-dir", str(results_dir),
        ],
        env=env_strict,
        capture_output=True,
        text=True,
    )
    assert p_fail.returncode == 2
    assert list(results_dir.glob("shortcut_audit_*.json"))

    env_allow = {**env_strict, "ALLOW_SHORTCUT_WARNING": "1"}
    p_ok = subprocess.run(
        [
            sys.executable, "-m", "src.eval", "--shortcut-only", "--fail-on-shortcut",
            "--manifest", str(manifest), "--results-dir", str(results_dir),
        ],
        env=env_allow,
        capture_output=True,
        text=True,
    )
    assert p_ok.returncode == 0


def _make_harmonic_clip(beta: float, f0: float, dur: float, seed: int, noise_lvl: float = 0.002) -> np.ndarray:
    """Speech-like harmonic stack; ``beta`` sets spectral-envelope slope -> rolloff_85."""
    rng = np.random.default_rng(seed)
    n = int(round(TARGET_SR * dur))
    t = np.arange(n, dtype=np.float32) / TARGET_SR
    ks = np.arange(1, int(7000.0 // f0) + 1, dtype=np.float64)
    amps = ks ** (-beta)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=ks.size)
    ang = (2.0 * np.pi * f0) * np.outer(t.astype(np.float64), ks)
    y = np.sin(ang) @ (amps * np.cos(phi)) + np.cos(ang) @ (amps * np.sin(phi))
    y = y / (float(np.max(np.abs(y))) + 1e-9)
    y += rng.normal(0.0, noise_lvl, size=n)
    y *= 0.6 + 0.4 * np.sin(2.0 * np.pi * 4.0 * t + rng.uniform(0.0, 2.0 * np.pi))
    return normalize_loudness(y.astype(np.float32))


def _write_rolloff_proxy_manifest(
    base: Path,
    n_per: int = 40,
    sigma_beta: float = 0.06,
    b_real: float = 0.795,
    b_spoof: float = 0.852,
) -> Path:
    """Manifest whose model-window rolloff_85 mirrors the real failure (~1290 real vs ~970 spoof).

    Harmonic beta jitter is calibrated so the UNaugmented trivial AUROC lands ~0.65
    (real Session-1 value: 0.6459); augmentation must then push it below 0.60.
    """
    wav_dir = base / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for si, split in enumerate(("train", "test")):
        for cls, (bmean, label, gen) in enumerate(((b_real, 0, "bonafide_wild"), (b_spoof, 1, "spoof"))):
            for i in range(n_per):
                seed = 100000 * si + 1000 * cls + i
                rng = np.random.default_rng(seed)
                beta = bmean + rng.normal(0.0, sigma_beta)
                f0 = rng.uniform(100.0, 200.0)
                dur = rng.uniform(2.5, 4.0)
                wav = _make_harmonic_clip(beta, f0, dur, seed + 7)
                p = wav_dir / f"{split}_{'real' if label == 0 else 'fake'}_{i:03d}.wav"
                save_wav_mono(p, wav, TARGET_SR)
                rows.append({"path": str(p.relative_to(base)), "label": label, "generator": gen, "split": split})
    manifest = base / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return manifest


def test_spectral_augmentation_deterministic_and_label_free() -> None:
    wav = _make_dummy_speech(2.5, seed=3)
    wav_b = _make_dummy_speech(2.5, freq_hz=330.0, seed=99)

    def make(seed: int) -> TrainCodecAugmenter:
        return TrainCodecAugmenter(p_codec=0.0, p_noise=0.0, p_clip=0.0, seed=seed)

    o1 = make(11)(wav, TARGET_SR)
    o2 = make(11)(wav, TARGET_SR)
    o3 = make(12)(wav, TARGET_SR)
    assert o1.shape == wav.shape
    assert o1.dtype == np.float32
    assert np.isfinite(o1).all()
    assert np.array_equal(o1, o2)
    assert not np.array_equal(o1, o3)
    assert abs(audio_rms(o1) - TARGET_RMS) < 0.01

    # __call__ must never see a label
    params = inspect.signature(TrainCodecAugmenter.__call__).parameters
    assert not any("label" in name.lower() or name.lower() in {"y", "cls", "class", "is_real", "is_spoof"} for name in params)

    # RNG consumption is content-independent: different inputs, same seed -> same sampled params + same rng state
    a1, a2 = make(11), make(11)
    a1(wav, TARGET_SR)
    a2(wav_b, TARGET_SR)
    assert a1.last_spectral_params == a2.last_spectral_params
    assert a1.rng.random() == a2.rng.random()

    # _sample_spectral_params is reproducible per seed
    pa = _sample_spectral_params(np.random.default_rng(0), TARGET_SR)
    pb = _sample_spectral_params(np.random.default_rng(0), TARGET_SR)
    assert pa == pb
    assert _sample_spectral_params(np.random.default_rng(1), TARGET_SR) != pa

    # Forced low-pass measurably lowers rolloff_85 of a wideband signal
    wide = normalize_loudness(np.random.default_rng(0).standard_normal(int(TARGET_SR * 2.5)).astype(np.float32))
    r0 = spectral_rolloff_hz(wide, TARGET_SR)
    off = {"apply": False, "gain_db": 0.0, "corner_hz": 300.0}
    forced_lp = {
        "lowpass": {"apply": True, "fc_hz": 3400.0},
        "low_shelf": dict(off),
        "high_shelf": {"apply": False, "gain_db": 0.0, "corner_hz": 2000.0},
        "tilt": {"apply": False, "tilt_db_per_octave": 0.0},
    }
    out_lp = apply_spectral_augment(wide, TARGET_SR, params=forced_lp)
    assert out_lp.shape == wide.shape and out_lp.dtype == np.float32 and np.isfinite(out_lp).all()
    assert spectral_rolloff_hz(out_lp, TARGET_SR) < r0 - 100.0

    # Forced negative high shelf also lowers rolloff measurably
    forced_hs = {
        "lowpass": {"apply": False, "fc_hz": 5000.0},
        "low_shelf": dict(off),
        "high_shelf": {"apply": True, "gain_db": -6.0, "corner_hz": 1000.0},
        "tilt": {"apply": False, "tilt_db_per_octave": 0.0},
    }
    out_hs = apply_spectral_augment(wide, TARGET_SR, params=forced_hs)
    assert spectral_rolloff_hz(out_hs, TARGET_SR) < r0 - 100.0


def test_augmented_shortcut_audit_hides_rolloff_proxy(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    manifest = _write_rolloff_proxy_manifest(tmp_path / "proxy")
    audit = run_augmented_shortcut_audit(manifest, max_per_split=1000, seed=42, opus_cache_dir=tmp_path / "opus_cache")
    unaug = audit["model_window_shortcuts"]
    aug = audit["augmented_train_window_shortcuts"]
    print(
        f"[proxy] unaugmented model-window AUROC={unaug['auroc']} (EER {unaug['eer_pct']}%) | "
        f"augmented AUROC={aug['auroc']} (EER {aug['eer_pct']}%) | real-data ref 0.6459"
    )
    # mirrors the real Session-1 failure band, then augmentation must hide it
    assert unaug["auroc"] >= 0.60
    assert unaug["auroc"] <= 0.75
    assert aug["auroc"] < 0.60
    assert aug["shortcut_warning"] is False
    # sanity: rolloff means reproduce the real ~1290 vs ~967 gap before augmentation
    means = unaug["test_feature_means"]
    assert means["real"]["rolloff_85_hz"] > means["spoof"]["rolloff_85_hz"] + 150.0
    assert abs(means["real"]["duration_s"] - means["spoof"]["duration_s"]) < 0.01
    assert audit["augmentation"]["p_lowpass"] == 0.4
    assert audit["augmentation"]["p_tilt"] == 0.6
    assert audit["seed"] == 42
    capsys.readouterr()


def _write_rolloff_matched_manifest(base: Path, n_per: int = 15) -> tuple[Path, list[dict]]:
    """Train+test manifest with overlapping-but-separated rolloff distributions per class."""
    wav_dir = base / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    specs = (
        ("train", "bonafide_wild", 0, 0.80),
        ("train", "spoof", 1, 0.90),
        ("test", "bonafide_wild", 0, 0.80),
        ("test", "spoof", 1, 0.90),
        ("test", "kokoro_unseen", 1, 0.86),
    )
    for si, (split, gen, label, bmean) in enumerate(specs):
        for i in range(n_per):
            seed = 7000 + 1000 * si + i
            rng = np.random.default_rng(seed)
            beta = bmean + rng.normal(0.0, 0.06)
            f0 = rng.uniform(100.0, 200.0)
            dur = rng.uniform(2.5, 4.0)
            wav = _make_harmonic_clip(beta, f0, dur, seed + 3)
            p = wav_dir / f"{split}_{gen}_{i:03d}.wav"
            save_wav_mono(p, wav, TARGET_SR)
            rows.append({"path": str(p), "label": label, "generator": gen, "split": split})
    manifest = base / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return manifest, rows


def test_rolloff_matched_pair_and_eval_slices(tmp_path: Path) -> None:
    manifest, rows = _write_rolloff_matched_manifest(tmp_path / "rm")
    test_rows = [r for r in rows if r["split"] == "test"]
    reals = [r for r in test_rows if int(r["label"]) == 0]
    spoofs = [r for r in test_rows if int(r["label"]) == 1 and "kokoro" not in r["generator"]]

    pair, summary = build_rolloff_matched_pair(reals, spoofs, n_bins=10, max_total=16, seed=0)
    n_real = sum(1 for r in pair if int(r["label"]) == 0)
    n_spoof = sum(1 for r in pair if int(r["label"]) == 1)
    assert len(pair) > 0
    assert len(pair) <= 16
    assert n_real == n_spoof
    for b in summary["per_bin"]:
        assert b["n_real"] == b["n_spoof"]
    before = summary["rolloff_mean_hz"]["before"]
    after = summary["rolloff_mean_hz"]["after"]
    assert abs(after["real"] - after["spoof"]) < abs(before["real"] - before["spoof"])
    assert summary["n_bins_used"] >= 1

    eval_results_dir = tmp_path / "eval_results"
    results = run_evaluation(
        manifest, detector=MockDetectorRuntime(), max_per_slice=12,
        out_json=tmp_path / "rm_eval.json", results_dir=eval_results_dir,
    )
    assert "rolloff_matched_clean" in results
    rm = results["rolloff_matched_clean"]
    assert "eer_pct" in rm["cloakbreak"] and "auroc" in rm["cloakbreak"]
    assert "rolloff_matching" in rm
    assert rm["rolloff_matching"]["n_bins_used"] >= 1
    assert "rolloff_matched_unseen_kokoro_clean" in results
    assert "rolloff_matching" in results["rolloff_matched_unseen_kokoro_clean"]
    assert list(eval_results_dir.glob("eval_shortcut_audit_*.json"))
    md = format_markdown_table(results)
    assert "rolloff_matched_clean" in md
    assert "rolloff_matched_unseen_kokoro_clean" in md


def test_format_markdown_table_floor_rows() -> None:
    results = {
        "test_clean": {
            "condition": "clean",
            "num_clips": 4,
            "num_real": 2,
            "num_spoof": 2,
            "cloakbreak": {"eer_pct": 12.5, "auroc": 0.9, "f1_at_05": 0.8, "eer_threshold": 0.4},
            "latency_p50_ms": 1.0,
            "latency_p95_ms": 2.0,
        },
        "shortcut_audit": {
            "num_test": 20,
            "model_window_shortcuts": {"eer_pct": 38.37, "auroc": 0.6459},
            "clean_real_vs_kokoro_shortcuts": {"eer_pct": 41.0, "auroc": 0.61, "num_scored": 8},
        },
        "clean_real_fpr": {
            "num_clean_real": 4,
            "cloakbreak_fpr_at_05_pct": 1.0,
            "cloakbreak_mean_prob": 0.1,
            "cloakbreak_p95_prob": 0.2,
        },
    }
    md = format_markdown_table(results)
    assert "Floor: trivial 4-feature logistic (dur/RMS/SNR/rolloff)" in md
    assert "Floor: trivial 4-feature logistic (clean real vs Kokoro)" in md
    assert "0.6459" in md and "38.37" in md
    assert "shortcut_audit" not in md
    assert md.count("| test_clean |") == 1


def test_eval_cli_augmented_shortcut_modes_and_results_dir(tmp_path: Path) -> None:
    """SNR-biased fixture: huge real-noise vs clean-spoof gap survives augmentation -> exit 2 strict."""
    bias_dir = tmp_path / "snr_biased"
    wav_dir = bias_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for si, split in enumerate(("train", "test")):
        for i in range(10):
            f0 = 180.0 + i * 4.0
            rw = _make_dummy_speech(2.5, freq_hz=f0, seed=si * 100 + i)
            r_wav = normalize_loudness(rw + rng.normal(0.0, 0.045, size=rw.shape[0]).astype(np.float32))
            s_wav = normalize_loudness(_make_dummy_speech(2.5, freq_hz=f0, seed=500 + si * 100 + i))
            rp = wav_dir / f"{split}_r{i}.wav"
            sp = wav_dir / f"{split}_s{i}.wav"
            save_wav_mono(rp, r_wav, TARGET_SR)
            save_wav_mono(sp, s_wav, TARGET_SR)
            rows.append({"path": str(rp.relative_to(bias_dir)), "label": 0, "generator": "bonafide_wild", "split": split})
            rows.append({"path": str(sp.relative_to(bias_dir)), "label": 1, "generator": "spoof", "split": split})
    manifest = bias_dir / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    results_dir = tmp_path / "cli_results_aug"

    env_strict = {k: v for k, v in os.environ.items() if k != "ALLOW_SHORTCUT_WARNING"}
    base_cmd = [
        sys.executable, "-m", "src.eval", "--shortcut-only",
        "--manifest", str(manifest), "--results-dir", str(results_dir),
    ]

    p_plain = subprocess.run(base_cmd, env=env_strict, capture_output=True, text=True)
    assert p_plain.returncode == 0
    assert list(results_dir.glob(f"shortcut_audit_{bias_dir.name}_*.json"))

    aug_cmd = base_cmd + ["--augment-train-windows", "--fail-on-shortcut"]
    p_fail = subprocess.run(aug_cmd, env=env_strict, capture_output=True, text=True)
    assert p_fail.returncode == 2
    assert "[augmented audit] trivial AUROC" in p_fail.stdout
    assert list(results_dir.glob(f"shortcut_audit_augmented_{bias_dir.name}_*.json"))

    p_ok = subprocess.run(
        aug_cmd, env={**env_strict, "ALLOW_SHORTCUT_WARNING": "1"}, capture_output=True, text=True
    )
    assert p_ok.returncode == 0


def test_hf_auth_check_validation() -> None:
    def whoami(role: str = "write", name: str = "Cipoklean", fine_grained: dict | None = None) -> dict:
        token = {"role": role}
        if fine_grained is not None:
            token["fineGrained"] = fine_grained
        return {"name": name, "auth": {"accessToken": token}}

    msg = hf_auth_check.validate_hf_identity(whoami(), "Cipoklean/cloakbreak-data-v1")
    assert "HF auth OK" in msg and "Cipoklean" in msg and "write" in msg

    with pytest.raises(hf_auth_check.HfAuthError, match="write token"):
        hf_auth_check.validate_hf_identity(whoami(role="read"), "Cipoklean/cloakbreak-data-v1")

    with pytest.raises(hf_auth_check.HfAuthError) as exc_info:
        hf_auth_check.validate_hf_identity(whoami(), "cipoklean/cloakbreak-data-v1")
    assert "cipoklean" in str(exc_info.value) and "Cipoklean" in str(exc_info.value)

    fg_ok = {"scoped": [{"entity": {"name": "Cipoklean"}, "permissions": ["repo.write", "repo.read"]}]}
    msg = hf_auth_check.validate_hf_identity(
        whoami(role="fineGrained", fine_grained=fg_ok), "Cipoklean/cloakbreak-data-v1"
    )
    assert "HF auth OK" in msg

    fg_read_only = {"scoped": [{"entity": {"name": "Cipoklean"}, "permissions": ["repo.read"]}]}
    with pytest.raises(hf_auth_check.HfAuthError):
        hf_auth_check.validate_hf_identity(
            whoami(role="fineGrained", fine_grained=fg_read_only), "Cipoklean/cloakbreak-data-v1"
        )

    fg_other_ns = {"scoped": [{"entity": {"name": "SomeoneElse"}, "permissions": ["repo.write"]}]}
    with pytest.raises(hf_auth_check.HfAuthError):
        hf_auth_check.validate_hf_identity(
            whoami(role="fineGrained", fine_grained=fg_other_ns), "Cipoklean/cloakbreak-data-v1"
        )

    for bad in ("noslash", "a/b/c", "/name", "ns/", "", "two words/x"):
        with pytest.raises(hf_auth_check.HfAuthError):
            hf_auth_check.validate_hf_identity(whoami(), bad)


def test_hf_auth_check_fetch_whoami_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error
    import urllib.request

    def fake_401(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_401)
    with pytest.raises(hf_auth_check.HfAuthError, match="invalid or expired"):
        hf_auth_check.fetch_whoami("tok")

    def fake_neterr(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_neterr)
    with pytest.raises(hf_auth_check.HfAuthError, match="request failed"):
        hf_auth_check.fetch_whoami("tok")


def test_hf_auth_check_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_DATASET_REPO", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        hf_auth_check.main()
    assert "HF AUTH PREFLIGHT FAIL" in str(exc_info.value)

    secret = "hf_secret_token_abc123"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("HF_DATASET_REPO", "Cipoklean/cloakbreak-data-v1")
    monkeypatch.setattr(
        hf_auth_check,
        "fetch_whoami",
        lambda token, endpoint="https://huggingface.co": {
            "name": "Cipoklean",
            "auth": {"accessToken": {"role": "write"}},
        },
    )
    hf_auth_check.main()
    out = capsys.readouterr().out
    assert "HF auth OK" in out
    assert secret not in out


def test_session1_script_lock_hash_and_hf_check() -> None:
    root = Path(__file__).resolve().parent.parent
    s1 = (root / "scripts" / "session1_vast.sh").read_text(encoding="utf-8")
    smoke = (root / "scripts" / "smoke_e2e.sh").read_text(encoding="utf-8")

    assert "sha256sum requirements-gpu.lock" in s1
    assert "cut -d' ' -f1" in s1
    assert 'echo "$LOCK_HASH" > logs/.s1_deps_done' in s1
    assert "rm -f logs/.s1_preflight_done" in s1
    assert "scripts.hf_auth_check" in s1
    assert s1.index("scripts.hf_auth_check") < s1.index("[step 1/6]")
    assert "Cipoklean/cloakbreak-data-v1" in s1
    assert "Cipoklean/cloakbreak-data-v1" in smoke
    assert "--augment-train-windows" in smoke

    for rel in ("scripts/session1_vast.sh", "scripts/smoke_e2e.sh"):
        proc = subprocess.run(["bash", "-n", str(root / rel)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
