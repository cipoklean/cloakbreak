"""Session-2 tests: resume/checkpoints, budget guard, INT8 parity, dataset restore,
HF hub helpers, Opus-cache progress, HF_MODEL_REPO auth, and the session2 script."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import hf_auth_check
from src import hub
from src.audio_utils import (
    TARGET_SR,
    load_wav_bytes_mono,
    load_wav_mono,
    normalize_loudness,
    pad_or_crop,
    save_wav_mono,
    wav_to_bytes,
)
from src.augment import precompute_opus_cache
from src.eval import check_int8_parity
from src.generate_spoofs import _inthewild_utt_id, build_from_local_dir
from src.restore_dataset import _restore_from_rows, restore_dataset
from src.train import BudgetExceeded, resolve_resume_checkpoint, train_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_dummy_speech(duration_s: float = 2.5, freq_hz: float = 220.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(TARGET_SR * duration_s)
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float32)
    env = 0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
    sig = env * (0.3 * np.sin(2.0 * np.pi * freq_hz * t) + 0.15 * np.sin(2.0 * np.pi * (2.5 * freq_hz) * t))
    sig += rng.normal(0.0, 0.01, size=n).astype(np.float32)
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


@pytest.fixture()
def s2_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Small local manifest: wild+clean reals, mock chatterbox/kokoro spoofs."""
    real_dir = tmp_path / "real_wavs"
    real_dir.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        wav = _make_dummy_speech(duration_s=2.5, freq_hz=150.0 + i * 15.0, seed=i)
        save_wav_mono(real_dir / f"spk{i:02d}_utt0.wav", wav, TARGET_SR)
    clean_dir = tmp_path / "clean_wavs"
    clean_dir.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        wav = _make_dummy_speech(duration_s=2.5, freq_hz=170.0 + i * 13.0, seed=100 + i)
        save_wav_mono(clean_dir / f"cln{i:02d}_utt0.wav", wav, TARGET_SR)
    data_dir = tmp_path / "dataset"
    manifest = build_from_local_dir(
        real_dir=real_dir,
        clean_real_dir=clean_dir,
        out_dir=data_dir,
        max_chatterbox=8,
        max_kokoro=4,
        mock_generator=True,
    )
    return data_dir, manifest


def _train_kwargs(manifest: Path, out_dir: Path, **overrides):
    kwargs = dict(
        manifest_path=manifest,
        out_dir=out_dir,
        epochs_stage1=1,
        epochs_stage2=0,
        batch_size=8,
        use_tiny_test_backbone=True,
        device_str="cpu",
        num_workers=0,
    )
    kwargs.update(overrides)
    return kwargs


def test_resume_last_ckpt_stage_skip_and_callback(s2_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    _, manifest = s2_dataset
    ckpt_dir = tmp_path / "ckpt"
    calls = []

    def cb(files, info):
        calls.append((list(files), dict(info)))

    train_model(**_train_kwargs(manifest, ckpt_dir, checkpoint_callback=cb))

    last = ckpt_dir / "last.pt"
    payload = torch.load(last, map_location="cpu", weights_only=True)
    assert payload["version"] == 1
    assert payload["stage"] == "stage1_head"
    assert payload["next_epoch"] == 2
    assert payload["partial_epoch"] is False
    for key, val in payload.items():
        if key in {"model", "optimizer", "scheduler", "scaler"}:
            continue
        assert isinstance(val, (int, float, str, bool, list, dict)) or val is None, key
    assert (ckpt_dir / "train_history.json").exists()
    assert calls, "checkpoint_callback never invoked"
    assert any(any(p.name == "last.pt" for p in files) for files, _ in calls)
    gs_after_stage1 = int(payload["global_steps"])

    # Resume with stage1 already finished -> stage1 skipped, stage2 runs once.
    summary = train_model(
        **_train_kwargs(
            manifest, ckpt_dir, epochs_stage2=1, resume_from=last, checkpoint_callback=cb
        )
    )
    stages = [h["stage"] for h in summary["history"]]
    assert stages.count("stage1_head") == 1
    assert stages.count("stage2_finetune") == 1
    payload2 = torch.load(last, map_location="cpu", weights_only=True)
    assert int(payload2["global_steps"]) > gs_after_stage1
    assert payload2["stage"] == "stage2_finetune"

    # Config mismatch must fail loudly.
    with pytest.raises(ValueError, match="unfreeze_layers"):
        train_model(
            **_train_kwargs(manifest, ckpt_dir, unfreeze_layers=3, resume_from=last)
        )


def test_budget_exceeded_saves_partial_and_resumes(s2_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    _, manifest = s2_dataset
    ckpt_dir = tmp_path / "ckpt_budget"
    cb_calls = []

    with pytest.raises(BudgetExceeded) as exc_info:
        train_model(
            **_train_kwargs(
                manifest,
                ckpt_dir,
                epochs_stage1=1,
                epochs_stage2=1,
                deadline_unix=time.time() - 1.0,
                checkpoint_callback=lambda f, i: cb_calls.append((f, i)),
            )
        )
    exc = exc_info.value
    assert exc.stage == "stage1_head"
    assert exc.epoch == 1
    assert cb_calls, "checkpoint_callback not called on budget stop"

    last = ckpt_dir / "last.pt"
    payload = torch.load(last, map_location="cpu", weights_only=True)
    assert payload["partial_epoch"] is True
    assert payload["next_epoch"] == exc.epoch  # rerun the interrupted epoch
    assert payload["stage"] == "stage1_head"

    # Resuming the partial checkpoint with no deadline completes the epoch.
    train_model(**_train_kwargs(manifest, ckpt_dir, epochs_stage2=0, resume_from=last))
    payload2 = torch.load(last, map_location="cpu", weights_only=True)
    assert payload2["partial_epoch"] is False
    assert payload2["next_epoch"] == 2


def test_checkpoint_callback_two_consecutive_failures_raise(s2_dataset: tuple[Path, Path], tmp_path: Path, capsys) -> None:
    _, manifest = s2_dataset
    ckpt_dir = tmp_path / "ckpt_cb_fail"

    def always_fail(files, info):
        raise OSError("simulated HF upload failure")

    with pytest.raises(RuntimeError, match="twice in a row"):
        train_model(
            **_train_kwargs(
                manifest, ckpt_dir, epochs_stage1=2, epochs_stage2=0,
                precompute_opus=False, checkpoint_callback=always_fail,
            )
        )
    out = capsys.readouterr().out
    assert "(1/2 consecutive)" in out  # first epoch only warns
    assert "(2/2 consecutive)" in out
    # last.pt was still written locally before the failure.
    assert (ckpt_dir / "last.pt").exists()


def test_checkpoint_callback_single_failure_recovers(s2_dataset: tuple[Path, Path], tmp_path: Path, capsys) -> None:
    _, manifest = s2_dataset
    ckpt_dir = tmp_path / "ckpt_cb_once"
    state = {"calls": 0}

    def flaky(files, info):
        state["calls"] += 1
        if state["calls"] == 1:
            raise OSError("transient upload failure")

    summary = train_model(
        **_train_kwargs(
            manifest, ckpt_dir, epochs_stage1=2, epochs_stage2=0,
            precompute_opus=False, checkpoint_callback=flaky,
        )
    )
    assert state["calls"] == 2
    out = capsys.readouterr().out
    assert "(1/2 consecutive)" in out
    assert "twice in a row" not in out
    assert summary["history"]  # training completed


def test_budget_exceeded_records_failed_upload(s2_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    _, manifest = s2_dataset
    ckpt_dir = tmp_path / "ckpt_budget_upload_fail"

    def always_fail(files, info):
        raise OSError("simulated HF upload failure")

    with pytest.raises(BudgetExceeded) as exc_info:
        train_model(
            **_train_kwargs(
                manifest, ckpt_dir, epochs_stage1=1, epochs_stage2=0,
                precompute_opus=False, deadline_unix=time.time() - 1.0,
                checkpoint_callback=always_fail,
            )
        )
    assert exc_info.value.uploaded is False
    assert (ckpt_dir / "last.pt").exists()


def test_train_cli_budget_exit_3(s2_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    _, manifest = s2_dataset
    ckpt_dir = tmp_path / "ckpt_cli"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [
            sys.executable, "-m", "src.train",
            "--manifest", str(manifest),
            "--out-dir", str(ckpt_dir),
            "--tiny-test",
            "--device", "cpu",
            "--epochs-stage1", "1",
            "--epochs-stage2", "0",
            "--batch-size", "8",
            "--deadline-unix", str(time.time() - 1.0),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 3, out[-3000:]
    assert "[budget] MAX_HOURS deadline reached" in out
    assert "checkpoint saved" in out
    assert (ckpt_dir / "last.pt").exists()


def test_int8_parity_stub_and_cli(s2_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    _, manifest = s2_dataset

    class _QueueRuntime:
        """Returns queued spoof_prob values in call order (row order is deterministic)."""

        def __init__(self, scores: list[float]) -> None:
            self._scores = list(scores)

        def predict(self, wav):
            class _R:
                pass

            r = _R()
            r.spoof_prob = float(self._scores.pop(0))
            return r

    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    val_rows = [r for r in rows if r.get("split") == "val" and "kokoro" not in r.get("generator", "")]
    assert val_rows, "fixture must produce val rows for parity test"
    labels = [int(r["label"]) for r in val_rows]
    assert 0 in labels and 1 in labels
    good = [0.9 if l else 0.1 for l in labels]      # perfect separation -> AUROC 1.0
    bad = [0.1 if l else 0.9 for l in labels]       # inverted -> AUROC 0.0

    same = check_int8_parity(manifest, _QueueRuntime(good), _QueueRuntime(list(good)))
    assert same["passed"] is True
    assert same["auroc_abs_diff"] == 0.0
    assert same["num_clips"] == len(val_rows)
    assert same["num_real"] == labels.count(0)
    assert same["num_spoof"] == labels.count(1)

    diff = check_int8_parity(manifest, _QueueRuntime(good), _QueueRuntime(bad), max_auroc_diff=0.01)
    assert diff["passed"] is False
    assert diff["auroc_abs_diff"] > 0.5

    # Real tiny ONNX pair via a short train run.
    ckpt_dir = tmp_path / "ckpt_parity"
    train_model(
        **_train_kwargs(manifest, ckpt_dir, epochs_stage2=0, max_steps=4)
    )
    fp32 = ckpt_dir / "cloakbreak_fp32.onnx"
    int8 = ckpt_dir / "cloakbreak_int8.onnx"
    assert fp32.exists() and int8.exists()

    results_dir = tmp_path / "parity_results"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    base_cmd = [
        sys.executable, "-m", "src.eval",
        "--int8-parity",
        "--manifest", str(manifest),
        "--fp32-model", str(fp32),
        "--model-path", str(int8),
        "--results-dir", str(results_dir),
    ]
    ok = subprocess.run(
        base_cmd + ["--max-auroc-diff", "1.0"], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert ok.returncode == 0, (ok.stdout + ok.stderr)[-3000:]
    assert "[int8 parity]" in ok.stdout
    parity_jsons = list(results_dir.glob("int8_parity_*.json"))
    assert parity_jsons, "parity JSON not saved"
    saved = json.loads(parity_jsons[0].read_text())
    assert saved["passed"] is True

    bad = subprocess.run(
        base_cmd + ["--max-auroc-diff=-1"], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert bad.returncode == 4, (bad.stdout + bad.stderr)[-3000:]
    assert "FAIL" in bad.stdout


def _restore_kwargs(data_dir: Path, **overrides):
    kwargs = dict(
        repo_id="test/repo",
        out_dir=data_dir,
        skip_download=True,
        skip_upstream=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_restore_dataset_verify_only_and_failures(s2_dataset: tuple[Path, Path], tmp_path: Path, capsys) -> None:
    data_dir, manifest = s2_dataset
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]

    rc = restore_dataset(**_restore_kwargs(data_dir, expected_rows=len(rows)))
    assert rc == 0
    out = capsys.readouterr().out
    assert f"manifest rows: {len(rows)} (verified {len(rows)})" in out

    # Wrong expected row count on an intact dataset -> failure.
    rc = restore_dataset(**_restore_kwargs(data_dir, expected_rows=len(rows) + 5))
    assert rc != 0
    assert "expected" in capsys.readouterr().err

    # Missing synthetic clip -> hard failure, never regenerate.
    cb_file = data_dir / "chatterbox" / "cb_00000.wav"
    assert cb_file.exists()
    cb_file.unlink()
    rc = restore_dataset(**_restore_kwargs(data_dir))
    assert rc != 0
    assert "never regenerates" in capsys.readouterr().err

    # Restore it, then delete a REAL clip -> verify failure under --skip-upstream.
    build_from_local_dir(
        real_dir=tmp_path / "real_wavs",
        clean_real_dir=tmp_path / "clean_wavs",
        out_dir=data_dir,
        max_chatterbox=8,
        max_kokoro=4,
        mock_generator=True,
    )
    real_file = data_dir / "real_norm" / "spk00_utt0.wav"
    assert real_file.exists()
    real_file.unlink()
    rc = restore_dataset(**_restore_kwargs(data_dir))
    assert rc != 0
    err = capsys.readouterr().err
    assert "missing or too small" in err or "verified" in err


def test_restore_dataset_duration_mismatch(s2_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    data_dir, _ = s2_dataset
    copy_dir = tmp_path / "dataset_copy"
    shutil.copytree(data_dir, copy_dir)
    manifest_path = copy_dir / "manifest.jsonl"
    lines = manifest_path.read_text().splitlines()
    row = json.loads(lines[0])
    row["duration_s"] = float(row.get("duration_s", 1.0)) + 5.0
    lines[0] = json.dumps(row)
    manifest_path.write_text("\n".join(lines) + "\n")
    rc = restore_dataset(repo_id="t/r", out_dir=copy_dir, skip_download=True, skip_upstream=True)
    assert rc != 0


def test_restore_from_rows_fake_iterable(tmp_path: Path) -> None:
    wav = _make_dummy_speech(2.5, freq_hz=300.0, seed=3)
    wav_bytes = wav_to_bytes(wav, TARGET_SR)
    decoded, _ = load_wav_bytes_mono(wav_bytes, target_sr=TARGET_SR)
    expected = normalize_loudness(decoded)

    needed = {
        "utt_keep": tmp_path / "keep.wav",
        "utt_skip_never_seen": tmp_path / "skip.wav",
    }
    rows_iter = [
        {"audio": {"bytes": wav_bytes}, "notes": json.dumps({"utterance_id": "utt_other"})},
        {"audio": {"bytes": wav_bytes}, "notes": json.dumps({"utterance_id": "utt_keep"})},
        {"audio": {"bytes": wav_bytes}},  # no notes -> itw_000002 fallback id
    ]
    restored = _restore_from_rows(rows_iter, needed, _inthewild_utt_id, 0.0, 100.0)
    assert set(restored) == {"utt_keep"}
    assert needed == {"utt_skip_never_seen": tmp_path / "skip.wav"}
    out_wav, _ = load_wav_mono(tmp_path / "keep.wav", TARGET_SR)
    assert out_wav.shape == expected.shape
    assert np.allclose(out_wav, expected, atol=1e-4)  # PCM_16 quantization
    assert not (tmp_path / "skip.wav").exists()


def test_hub_upload_artifacts_dry_run_and_download_none(tmp_path: Path, monkeypatch, capsys) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "cloakbreak_fp32.onnx").write_bytes(b"fp32")
    (ckpt / "cloakbreak_int8.onnx").write_bytes(b"int8")
    (ckpt / "cloakbreak_best.pt").write_bytes(b"pt")
    (ckpt / "train_summary.json").write_text("{}")
    (ckpt / "last.pt").write_bytes(b"last")  # must NOT be uploaded
    results = tmp_path / "results"
    results.mkdir()
    (results / "eval_results.json").write_text("{}")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "session2.log").write_text("log")
    (logs / ".s2smoke_gate_done").write_text("")  # marker must NOT be uploaded

    monkeypatch.setattr(sys, "argv", [
        "src.hub", "upload-artifacts",
        "--repo", "ns/model",
        "--ckpt-dir", str(ckpt),
        "--results-dir", str(results),
        "--log-dir", str(logs),
        "--prefix", "smoke_check/",
        "--dry-run",
    ])
    with pytest.raises(SystemExit) as exc_info:
        hub.main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "smoke_check/onnx/cloakbreak_fp32.onnx" in out
    assert "smoke_check/checkpoints/cloakbreak_best.pt" in out
    assert "smoke_check/results/" in out
    assert "smoke_check/logs/session2.log" in out
    assert "last.pt" not in out
    assert ".s2smoke_gate_done" not in out

    # download_checkpoint -> None on missing entries
    from huggingface_hub.errors import EntryNotFoundError

    def _raise(*a, **k):
        raise EntryNotFoundError("missing")

    monkeypatch.setattr(hub, "hf_hub_download", _raise)
    assert hub.download_checkpoint("ns/model", "x.pt", tmp_path) is None


def test_precompute_opus_cache_progress(capsys, tmp_path: Path) -> None:
    paths = []
    for i in range(5):
        p = tmp_path / f"c{i}.wav"
        save_wav_mono(p, _make_dummy_speech(1.5, freq_hz=200 + i * 30, seed=i), TARGET_SR)
        paths.append(str(p))
    precompute_opus_cache(paths, cache_dir=tmp_path / "cache", bitrate_kbps=16, num_workers=1, progress_every=2)
    out = capsys.readouterr().out
    progress_lines = [l for l in out.splitlines() if l.startswith("[cache]") and "Opus clips" in l]
    assert len(progress_lines) >= 3  # done=2, done=4, final
    assert "5/5" in progress_lines[-1]


def test_hf_auth_check_model_repo(monkeypatch, capsys) -> None:
    whoami = {
        "name": "Cipoklean",
        "auth": {"accessToken": {"role": "write", "fineGrained": {"scoped": []}}},
    }
    monkeypatch.setattr(hf_auth_check, "fetch_whoami", lambda token: whoami)
    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.setenv("HF_DATASET_REPO", "Cipoklean/cloakbreak-data-v1")
    monkeypatch.setenv("HF_MODEL_REPO", "Cipoklean/cloakbreak-wavlm")
    hf_auth_check.main()
    out = capsys.readouterr().out
    assert out.count("HF auth OK") == 2
    assert "cloakbreak-wavlm" in out

    monkeypatch.setenv("HF_MODEL_REPO", "cipoklean/cloakbreak-wavlm")
    with pytest.raises(SystemExit) as exc:
        hf_auth_check.main()
    assert "HF AUTH PREFLIGHT FAIL" in str(exc.value)
    assert "cipoklean" in str(exc.value)


def test_session2_script_static() -> None:
    import subprocess as sp

    script = REPO_ROOT / "scripts" / "session2_vast.sh"
    r = sp.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = script.read_text()
    for needle in (
        "sha256sum requirements-gpu.lock",
        "scripts.hf_auth_check",
        "--augment-train-windows",
        "--fail-on-shortcut",
        "Cipoklean/cloakbreak-wavlm",
        "Cipoklean/cloakbreak-data-v1",
        "--skip-tts",
        "resume-from",
        "deadline-unix",
        "int8-parity",
        "upload-artifacts",
    ):
        assert needle in text, needle


def _run_session2_dry_run(
    s2_dataset: tuple[Path, Path],
    tmp_path: Path,
    max_hours: str | None = None,
    extra_env: dict | None = None,
):
    data_dir, _ = s2_dataset
    log_dir = tmp_path / "s2logs"
    ckpt_dir = tmp_path / "s2ckpt"
    results_dir = tmp_path / "s2results"
    env = dict(os.environ)
    env.update({
        "SMOKE": "1",
        "DRY_RUN": "1",
        "ALLOW_NO_TMUX": "1",
        "PYTHON": sys.executable,
        "PYTHONPATH": str(REPO_ROOT),
        "DATA_DIR": str(data_dir),
        "CKPT_DIR": str(ckpt_dir),
        "RESULTS_DIR": str(results_dir),
        "LOG_DIR": str(log_dir),
        "ALLOW_SHORTCUT_WARNING": "1",
        "INT8_PARITY_MAX_DIFF": "1.0",
    })
    if max_hours is not None:
        env["MAX_HOURS"] = max_hours
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["bash", "scripts/session2_vast.sh", "--dry-run", "--max-steps", "4"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return proc, log_dir, ckpt_dir, results_dir, data_dir


def test_session2_dry_run_end_to_end(s2_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    stray_paths = (
        REPO_ROOT / "data" / "cloakbreak_v1",
        REPO_ROOT / "checkpoints" / "s2_smoke",
        REPO_ROOT / "results" / "smoke",
        REPO_ROOT / "logs" / "session2.log",
    )
    stray_before = {p: (p.stat().st_mtime if p.exists() else None) for p in stray_paths}
    proc, log_dir, ckpt_dir, results_dir, data_dir = _run_session2_dry_run(s2_dataset, tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-5000:]
    assert "SMOKE COMPLETE" in out
    assert (results_dir / "results_table.md").exists()
    assert (results_dir / "eval_results.json").exists()
    assert list(results_dir.glob("int8_parity_*.json")), "missing int8 parity JSON"
    assert list(results_dir.glob("shortcut_audit_augmented_*.json")), "missing gate audit JSON"
    table = (results_dir / "results_table.md").read_text()
    assert "Floor: trivial 4-feature logistic" in table
    assert "DRY RUN upload plan" in out or "upload plan" in out
    assert (log_dir / "session2.log").exists()

    # Grep-able [KEY] lines: stdout + persisted key_lines.txt.
    assert "[KEY] gate augmented_auroc=" in out
    assert "[KEY] int8_parity fp32_auroc=" in out
    assert "[KEY] headline unseen_kokoro_opus16k eer=" in out
    assert "[KEY] smoke used_min=" in out and "complete=true" in out
    assert "[smoke] used" in out
    key_lines = results_dir / "key_lines.txt"
    assert key_lines.exists()
    content = key_lines.read_text()
    for needle in (
        "[KEY] gate augmented_auroc=",
        "[KEY] int8_parity fp32_auroc=",
        "[KEY] headline unseen_kokoro_opus16k eer=",
        "[KEY] smoke used_min=",
    ):
        assert needle in content, needle
    import re

    assert re.search(
        r"^\[KEY\] gate augmented_auroc=\d+\.\d{4} eer=\d+\.\d{2} pass=(true|false)$",
        content, re.M,
    )
    assert re.search(
        r"^\[KEY\] int8_parity fp32_auroc=\d+\.\d{4} int8_auroc=\d+\.\d{4} delta=\d+\.\d{4}$",
        content, re.M,
    )
    assert re.search(
        r"^\[KEY\] headline unseen_kokoro_opus16k eer=\d+\.\d{2} auroc=\d+\.\d{4} baseline_eer=(\d+\.\d{2}|NA)$",
        content, re.M,
    )
    assert re.search(
        r"^\[KEY\] smoke used_min=\d+\.\d cap_min=\d+\.\d setup_min=\d+\.\d complete=true$",
        content, re.M,
    )
    # Nothing leaked into the repo working dirs: paths absent before the run
    # must still be absent; pre-existing paths (e.g. a real restored dataset
    # under the gitignored data/) must be unmodified.
    for stray in stray_paths:
        before = stray_before[stray]
        if before is None:
            assert not stray.exists(), stray
        else:
            assert stray.exists() and stray.stat().st_mtime == before, stray

    # Rerun with same LOG_DIR/CKPT_DIR/RESULTS_DIR -> markers skip finished steps.
    data_dir2 = data_dir
    env = dict(os.environ)
    env.update({
        "SMOKE": "1", "DRY_RUN": "1", "ALLOW_NO_TMUX": "1",
        "PYTHON": sys.executable, "PYTHONPATH": str(REPO_ROOT),
        "DATA_DIR": str(data_dir2), "CKPT_DIR": str(ckpt_dir),
        "RESULTS_DIR": str(results_dir), "LOG_DIR": str(log_dir),
        "ALLOW_SHORTCUT_WARNING": "1", "INT8_PARITY_MAX_DIFF": "1.0",
    })
    rerun = subprocess.run(
        ["bash", "scripts/session2_vast.sh", "--dry-run", "--max-steps", "4"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=1800,
    )
    rout = rerun.stdout + rerun.stderr
    assert rerun.returncode == 0, rout[-3000:]
    assert "SMOKE COMPLETE" in rout
    assert "skipping" in rout.lower()


def test_session2_dry_run_budget_stop(s2_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    fresh = tmp_path / "budget_run"
    fresh.mkdir()
    proc, *_ = _run_session2_dry_run(s2_dataset, fresh, max_hours="0")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-5000:]
    assert "[budget] STOPPED CLEANLY" in out
    assert "ALL STEPS COMPLETE" not in out
    assert "SMOKE COMPLETE" not in out


def test_session2_dry_run_smoke_max_min_zero(s2_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    fresh = tmp_path / "smoke_cap_run"
    fresh.mkdir()
    proc, *_ = _run_session2_dry_run(s2_dataset, fresh, extra_env={"SMOKE_MAX_MIN": "0"})
    out = proc.stdout + proc.stderr
    assert proc.returncode == 5, out[-5000:]
    assert "complete=false" in out
    assert "TIME CAP REACHED" in out
    assert "SMOKE COMPLETE" not in out


def test_train_eta_key_line(s2_dataset: tuple[Path, Path], tmp_path: Path, monkeypatch, capsys) -> None:
    import re

    _, manifest = s2_dataset
    key_file = tmp_path / "key_lines.txt"
    monkeypatch.setenv("CB_KEY_LINES_FILE", str(key_file))
    monkeypatch.delenv("VAST_RATE_PER_HR", raising=False)

    train_model(
        **_train_kwargs(
            manifest, tmp_path / "ckpt_eta", epochs_stage1=1, epochs_stage2=0,
            batch_size=8, precompute_opus=False, eta_after_steps=2,
        )
    )
    out = capsys.readouterr().out
    na_re = re.compile(
        r"^\[KEY\] eta stage=stage1_head steps_per_s=[0-9.]+ train_eta_min=[0-9.]+ projected_cost=NA$",
        re.M,
    )
    assert na_re.search(out), out[-3000:]
    assert key_file.exists()
    content = key_file.read_text()
    assert na_re.search(content)

    # With a rate set, projected_cost becomes a dollar amount.
    key_file2 = tmp_path / "key_lines2.txt"
    monkeypatch.setenv("CB_KEY_LINES_FILE", str(key_file2))
    monkeypatch.setenv("VAST_RATE_PER_HR", "0.20")
    capsys.readouterr()
    train_model(
        **_train_kwargs(
            manifest, tmp_path / "ckpt_eta2", epochs_stage1=1, epochs_stage2=0,
            batch_size=8, precompute_opus=False, eta_after_steps=2,
        )
    )
    out2 = capsys.readouterr().out
    cost_re = re.compile(r"projected_cost=\$[0-9]+\.[0-9]{2}")
    assert cost_re.search(out2), out2[-3000:]
    assert cost_re.search(key_file2.read_text())


def test_no_tmux_new_in_scripts() -> None:
    for p in (REPO_ROOT / "scripts").rglob("*"):
        if p.is_file():
            assert "tmux new" not in p.read_text(errors="ignore"), p


def test_preflight_transformers_backends_collecting() -> None:
    """check_transformers_backends must report zero failures on the pinned env.

    This is the check that crashed on transformers 5.2.0 (is_safetensors_available
    was removed); running it with a collecting check() reproduces that failure
    mode without needing the full hidden-backend suite.
    """
    from scripts import preflight_vast

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(f"{label}: {detail}")

    preflight_vast.check_transformers_backends(check)
    assert failures == []


def test_baseline_detector_label_mapping(tmp_path: Path) -> None:
    """BaselineDetectorRuntime.fake_idx must follow the checkpoint's id2label.

    Vansh180 ships {0: real, 1: fake} -> fake_idx 1; a flipped config must give
    fake_idx 0. predict_prob must equal softmax(model(extractor(x)))[fake_idx].
    """
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForSequenceClassification,
    )

    from src.model import BaselineDetectorRuntime

    torch.manual_seed(0)
    cfg = Wav2Vec2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=37,
        conv_dim=[16, 16],
        conv_kernel=[3, 3],
        conv_stride=[2, 2],
        feat_extract_norm="group",
        vocab_size=32,
        pad_token_id=0,
        num_labels=2,
    )
    x = _make_dummy_speech(duration_s=2.5, freq_hz=220.0, seed=3)

    for tag, id2label, expect_idx in (
        ("normal", {0: "real", 1: "fake"}, 1),
        ("flipped", {0: "fake", 1: "real"}, 0),
    ):
        model = Wav2Vec2ForSequenceClassification(cfg)
        model.config.id2label = {int(k): v for k, v in id2label.items()}
        model.config.label2id = {v: int(k) for k, v in id2label.items()}
        ckpt_dir = tmp_path / tag
        model.save_pretrained(ckpt_dir)
        Wav2Vec2FeatureExtractor().save_pretrained(ckpt_dir)

        rt = BaselineDetectorRuntime(str(ckpt_dir), device="cpu")
        assert rt.fake_idx == expect_idx, (tag, rt.fake_idx)

        fixed = normalize_loudness(pad_or_crop(x, rt.window_samples, random_crop=False))
        inputs = rt.extractor(fixed, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(rt.device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = rt.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        expected = round(float(probs[expect_idx].item()), 4)
        assert rt.predict_prob(x) == expected, (tag, rt.predict_prob(x), expected)


def test_sysinfo_effective_cpu_count(tmp_path: Path) -> None:
    from src.sysinfo import effective_cpu_count

    empty = tmp_path / "empty"
    empty.mkdir()
    base = effective_cpu_count(str(empty))
    assert base >= 1

    # cgroup v2 quota 400000/100000 -> ceil(4.0) = 4 (or lower if affinity is tighter)
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "cpu.max").write_text("400000 100000\n")
    assert effective_cpu_count(str(v2)) <= 4

    # v2 "max" quota -> ignored, same as empty
    v2max = tmp_path / "v2max"
    v2max.mkdir()
    (v2max / "cpu.max").write_text("max 100000\n")
    assert effective_cpu_count(str(v2max)) == base

    # cgroup v1 cpu/ quota 200000/100000 -> <= 2
    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("200000\n")
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert effective_cpu_count(str(v1)) <= 2

    # cgroup v1 cpu,cpuacct/ variant -> <= 3
    v1alt = tmp_path / "v1alt"
    (v1alt / "cpu,cpuacct").mkdir(parents=True)
    (v1alt / "cpu,cpuacct" / "cpu.cfs_quota_us").write_text("300000\n")
    (v1alt / "cpu,cpuacct" / "cpu.cfs_period_us").write_text("100000\n")
    assert effective_cpu_count(str(v1alt)) <= 3

    # v1 quota -1 -> unlimited -> ignored
    v1unlim = tmp_path / "v1unlim"
    (v1unlim / "cpu").mkdir(parents=True)
    (v1unlim / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    (v1unlim / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert effective_cpu_count(str(v1unlim)) == base


def test_restore_dataset_splits_filter(s2_dataset: tuple[Path, Path], capsys) -> None:
    """--splits restricts restore+verify to the listed manifest splits."""
    data_dir, manifest = s2_dataset
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    train_row = next(r for r in rows if r.get("split") == "train")
    victim = data_dir / str(train_row["path"])
    assert victim.exists()
    victim.unlink()

    # val-only restore ignores the deleted train clip
    rc = restore_dataset(
        repo_id="unused", out_dir=data_dir,
        skip_download=True, skip_upstream=True, splits={"val"},
    )
    assert rc == 0
    assert "--splits ['val']" in capsys.readouterr().out

    # train-only restore must fail verification on the missing clip
    rc2 = restore_dataset(
        repo_id="unused", out_dir=data_dir,
        skip_download=True, skip_upstream=True, splits={"train"},
    )
    assert rc2 == 2


def _calib_manifest(tmp_path: Path) -> tuple[Path, list[float]]:
    """Write wavs + manifest.jsonl for calibrate tests.

    Returns (manifest_path, val_probs_in_call_order). val rows are written first
    in manifest order: 20 real then 10 spoof, all split='val' non-Kokoro.
    Test rows follow: 2 bonafide_wild + 2 bonafide_clean reals, 2 chatterbox
    spoofs, 2 kokoro_unseen spoofs (all split='test').
    """
    data = tmp_path / "calib_data"
    wavs = data / "wavs"
    wavs.mkdir(parents=True)
    rows: list[dict] = []
    n = 0

    def add(split: str, label: int, gen: str) -> None:
        nonlocal n
        name = f"c{n:03d}.wav"
        save_wav_mono(wavs / name, _make_dummy_speech(0.5, seed=n), TARGET_SR)
        rows.append({
            "id": f"r{n:03d}",
            "path": f"wavs/{name}",
            "label": label,
            "generator": gen,
            "split": split,
            "speaker_id": f"spk{n % 7}",
            "duration_s": 0.5,
        })
        n += 1

    val_real_probs = [0.5, 0.4, 0.3, 0.2, 0.1] + [0.05] * 15
    val_spoof_probs = [0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.3, 0.2]
    assert len(val_real_probs) == 20 and len(val_spoof_probs) == 10
    for i in range(20):
        add("val", 0, "bonafide_wild" if i % 2 == 0 else "bonafide_clean")
    for _ in range(10):
        add("val", 1, "chatterbox_mock")
    for _ in range(2):
        add("test", 0, "bonafide_wild")
    for _ in range(2):
        add("test", 0, "bonafide_clean")
    for _ in range(2):
        add("test", 1, "chatterbox_mock")
    for _ in range(2):
        add("test", 1, "kokoro_unseen")

    manifest = data / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return manifest, val_real_probs + val_spoof_probs


class _QueueRuntime:
    """DetectorRuntime stub returning crafted probs in exact call order."""

    backend_name = "queue_stub"

    def __init__(self, probs: list[float]) -> None:
        self._probs = [float(p) for p in probs]
        self.calls = 0

    def predict(self, waveform):
        from src.model import DetectionResult

        assert self.calls < len(self._probs), "runtime called more times than planned"
        p = self._probs[self.calls]
        self.calls += 1
        return DetectionResult(
            spoof_prob=round(p, 4),
            attention_focus=[0.0] * 32,
            latency_ms=0.1,
            backend=self.backend_name,
        )


def test_select_operating_threshold_rule() -> None:
    from src.calibrate import select_operating_threshold

    desc = [0.9, 0.5, 0.4, 0.3, 0.2, 0.1]
    # k = floor(0.5 * 6) = 3 -> t = desc[3] + eps
    assert select_operating_threshold(desc, 0.5) == pytest.approx(0.30005)
    # k >= n_real -> min prob
    assert select_operating_threshold(desc, 1.0) == pytest.approx(0.1)
    assert select_operating_threshold(desc, 2.0) == pytest.approx(0.1)
    # achieved FPR never exceeds target under prob >= t
    probs = [0.9, 0.8, 0.7, 0.6, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    t = select_operating_threshold(sorted(probs, reverse=True), 0.3)
    fpr = sum(1 for p in probs if p >= t) / len(probs)
    assert fpr <= 0.3


def test_calibrate_threshold_and_report(tmp_path: Path, capsys) -> None:
    """End-to-end calibrate run with a queue stub: exact threshold rule, schema,
    per-generator counts, KEY lines, and the two clean_real_vs_kokoro slices."""
    import hashlib

    from src import calibrate

    manifest, val_probs = _calib_manifest(tmp_path)
    # 18 test predict calls: cache files are shared per (tag, condition, seed);
    # test_clean/unseen_kokoro_clean/crk_clean share tag__clean__s1000 and the
    # rw1/rw2/rc1/rc2/k1/k2 rows dedupe across slices (2+4+4+2+2+2+2 = 18).
    test_probs = [round(0.05 + 0.02 * i, 4) for i in range(18)]
    runtime = _QueueRuntime(val_probs + test_probs)
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"fake-onnx-bytes")

    out_json = tmp_path / "thr.json"
    key_file = tmp_path / "key_lines.txt"
    report = calibrate.run_calibration(
        manifest, runtime,
        model_path=str(model_file), backend="onnx", cache_tag="t1",
        target_fpr=0.05, max_per_slice=8,
        cache_dir=tmp_path / "cache", key_lines_path=key_file, out_path=out_json,
    )

    # exact rule: k = floor(0.05*20) = 1 -> t = 2nd highest real prob + eps
    assert report["operating_threshold"] == pytest.approx(0.40005)
    val_op = report["val"]["at_operating_threshold"]
    assert val_op["fpr"] == pytest.approx(0.05)  # == target, never above
    assert val_op["tpr"] == pytest.approx(0.8)
    assert val_op["fpr_by_generator"]["bonafide_wild"] == pytest.approx(0.1)
    assert val_op["fpr_by_generator"]["bonafide_clean"] == pytest.approx(0.0)

    for key in (
        "model_path", "model_sha256", "backend", "target_fpr", "selection",
        "operating_threshold", "val", "val_eer_threshold", "test",
        "torch_parity", "created_utc",
    ):
        assert key in report, key
    assert report["selection"] == "val split only; test never used for selection"
    assert report["model_sha256"] == hashlib.sha256(b"fake-onnx-bytes").hexdigest()
    assert report["torch_parity"] is None

    expected_slices = {
        "clean_real", "test_clean", "test_opus_16k",
        "unseen_kokoro_clean", "unseen_kokoro_opus_16k",
        "clean_real_vs_kokoro_clean", "clean_real_vs_kokoro_opus_16k",
    }
    assert set(report["test"]) == expected_slices
    for name, blk in report["test"].items():
        assert "rows_by_generator" in blk, name
        assert blk["n_real"] + blk["n_spoof"] == blk["n_clips"], name
        for sub in ("at_operating_threshold", "at_eer_threshold", "at_0.5"):
            assert sub in blk, (name, sub)
    assert report["test"]["test_clean"]["rows_by_generator"] == {
        "bonafide_wild": 2, "chatterbox_mock": 2,
    }
    assert report["test"]["unseen_kokoro_clean"]["rows_by_generator"] == {
        "bonafide_wild": 2, "kokoro_unseen": 2,
    }
    assert report["test"]["clean_real_vs_kokoro_clean"]["rows_by_generator"] == {
        "bonafide_clean": 2, "kokoro_unseen": 2,
    }
    assert report["test"]["clean_real"]["n_spoof"] == 0
    assert report["test"]["clean_real"]["at_operating_threshold"]["tpr"] is None

    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[KEY]")]
    assert any(ln.startswith("[KEY] threshold op=") for ln in lines)
    assert sum(1 for ln in lines if ln.startswith("[KEY] threshold_test")) == 7
    cr_line = next(ln for ln in lines if "slice=clean_real " in ln)
    assert "tpr=NA" in cr_line
    kf = key_file.read_text(encoding="utf-8")
    assert "[KEY] threshold op=" in kf
    assert kf.count("[KEY] threshold_test") == 7

    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["operating_threshold"] == report["operating_threshold"]
    assert saved["test"]["clean_real_vs_kokoro_opus_16k"]["rows_by_generator"] == {
        "bonafide_clean": 2, "kokoro_unseen": 2,
    }


def test_calibrate_test_scores_do_not_change_threshold(tmp_path: Path) -> None:
    """TEST-slice scores are report-only: different test probs -> same threshold."""
    from src import calibrate

    manifest, val_probs = _calib_manifest(tmp_path)
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"x")
    cache_dir = tmp_path / "cache"

    def _run(tag: str, test_probs: list[float]) -> dict:
        rt = _QueueRuntime(val_probs + test_probs)
        return calibrate.run_calibration(
            manifest, rt,
            model_path=str(model_file), backend="onnx", cache_tag=tag,
            target_fpr=0.05, max_per_slice=8,
            cache_dir=cache_dir, progress_every=1000,
        )

    probs_a = [round(0.05 + 0.02 * i, 4) for i in range(18)]
    probs_b = [round(0.9 - 0.04 * i, 4) for i in range(18)]
    r1 = _run("tagA", probs_a)
    r2 = _run("tagB", probs_b)
    assert r1["operating_threshold"] == pytest.approx(0.40005)
    assert r2["operating_threshold"] == pytest.approx(0.40005)
    assert r1["val"]["auroc"] == r2["val"]["auroc"]
    # test blocks genuinely differ (different crafted probs)
    assert r1["test"] != r2["test"]


def test_calibrate_cache_resume_skips_predict(tmp_path: Path) -> None:
    """A second run over the same cache_dir+tag must not call predict at all."""
    from src import calibrate

    manifest, val_probs = _calib_manifest(tmp_path)
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"x")
    cache_dir = tmp_path / "cache"

    rt1 = _QueueRuntime(val_probs + [0.5] * 18)
    r1 = calibrate.run_calibration(
        manifest, rt1, model_path=str(model_file), backend="onnx",
        cache_tag="resume", target_fpr=0.05, max_per_slice=8,
        cache_dir=cache_dir, progress_every=1000,
    )
    assert rt1.calls == 48  # 30 val + 18 test (shared-cache dedup across slices)

    class _BoomRuntime:
        backend_name = "boom"

        def __init__(self) -> None:
            self.calls = 0

        def predict(self, waveform):
            self.calls += 1
            raise AssertionError("predict called — score cache missed")

    boom = _BoomRuntime()
    r2 = calibrate.run_calibration(
        manifest, boom, model_path=str(model_file), backend="onnx",
        cache_tag="resume", target_fpr=0.05, max_per_slice=8,
        cache_dir=cache_dir, progress_every=1000,
    )
    assert boom.calls == 0
    assert r2["operating_threshold"] == r1["operating_threshold"]
