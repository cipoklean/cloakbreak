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


def _run_session2_dry_run(s2_dataset: tuple[Path, Path], tmp_path: Path, max_hours: str | None = None):
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
    # Nothing leaked into the repo working dirs.
    for stray in (
        REPO_ROOT / "data" / "cloakbreak_v1",
        REPO_ROOT / "checkpoints" / "s2_smoke",
        REPO_ROOT / "results" / "smoke",
        REPO_ROOT / "logs" / "session2.log",
    ):
        assert not stray.exists(), stray

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
