"""Live-mic path benchmark: real browser -> /ws/detect -> debug capture -> offline replay.

Drives headless Chromium (Playwright) with Chrome's fake audio capture fed by a
48 kHz WAV built from real manifest audio, clicks "Start Live Mic", records
every WS prediction the page renders, then stops and analyzes:

  * live window-score distribution and LIKELY CLONE fraction (live FPR on
    known-real audio),
  * offline re-score of the server's debug capture (replay_capture),
  * reference scores of the original 16 kHz source at the same window
    positions, aligned to the capture by cross-correlation.

NOT part of pytest. Run:

    uv run --with playwright==1.65.2 python -m scripts.bench_live_mic \
        --manifest data/cloakbreak_v1/manifest.jsonl \
        --model-path artifacts/session2/onnx/cloakbreak_fp32.onnx \
        --minutes 2.0 --out results/live_mic_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from src.audio_utils import save_wav_mono

REPO = Path(__file__).resolve().parents[1]
SERVER_PY = REPO / ".venv" / "bin" / "python"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _build_source16(manifest: Path, split: str, generator: str, minutes: float, seed: int) -> np.ndarray:
    from src.audio_utils import TARGET_SR, load_wav_mono
    from src.eval import _resolve_manifest_rows

    rows = [
        r
        for r in _resolve_manifest_rows(manifest)
        if r.get("split") == split and r.get("generator") == generator
    ]
    if not rows:
        raise SystemExit(f"no rows with split={split!r} generator={generator!r} in {manifest}")
    need = int(round(minutes * 60.0 * TARGET_SR))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    parts: list[np.ndarray] = []
    total = 0
    for idx in order:
        wav, _sr = load_wav_mono(rows[int(idx)]["path"], target_sr=TARGET_SR)
        parts.append(wav)
        total += wav.shape[0]
        if total >= need:
            break
    src = np.concatenate(parts)[:need]
    if src.shape[0] < need:
        raise SystemExit(f"manifest exhausted at {src.shape[0] / TARGET_SR:.1f}s < {minutes * 60:.1f}s")
    return src.astype(np.float32)


def _write_fake_mic_wav(src16: np.ndarray, path: Path, hf_noise_dbfs: float | None = None, seed: int = 42) -> None:
    """Upsample 16 kHz -> 48 kHz and write the WAV Chrome's fake mic consumes.

    With ``hf_noise_dbfs``, band-limited 8.5-20 kHz noise at that RMS level is
    added to stand in for out-of-band mic energy (it must not alias into 0-8 kHz).
    """
    import soundfile as sf
    from scipy.signal import butter, resample_poly, sosfilt

    up = resample_poly(src16.astype(np.float64), 3, 1)
    if hf_noise_dbfs is not None:
        sos = butter(8, [8500, 20000], btype="bandpass", fs=48000, output="sos")
        noise = sosfilt(sos, np.random.default_rng(seed).standard_normal(up.size))
        noise *= (10.0 ** (hf_noise_dbfs / 20.0)) / max(float(np.sqrt(np.mean(noise**2))), 1e-12)
        up = up + noise
    sf.write(str(path), np.clip(up, -1.0, 1.0).astype(np.float32), 48000, subtype="PCM_16")


def _wait_health(port: int, timeout_s: float = 180.0) -> dict:
    deadline = time.time() + timeout_s
    last_err = "no response"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as r:
                health = json.loads(r.read().decode())
            if health.get("status") == "ok":
                return health
            last_err = f"status={health.get('status')}"
        except Exception as exc:  # noqa: BLE001 - poll until server is up
            last_err = str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"/api/health not ok after {timeout_s}s: {last_err}")


def _browser_session(port: int, wav48: Path, minutes: float):
    """Open the page, patch renderResult -> window.__cbLog, run the mic, stop."""
    from playwright.sync_api import sync_playwright

    console_lines: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={wav48}",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        try:
            page = browser.new_page()
            page.on("console", lambda m: console_lines.append(m.text))
            page.goto(f"http://127.0.0.1:{port}/", wait_until="load")
            page.wait_for_function("() => typeof window.renderResult === 'function'", timeout=15000)
            page.evaluate(
                """() => {
                    window.__cbLog = [];
                    const orig = window.renderResult;
                    window.renderResult = function (d) {
                        window.__cbLog.push(d);
                        return orig.apply(this, arguments);
                    };
                }"""
            )
            page.click("#micBtn")
            page.wait_for_function(
                "() => { const el = document.getElementById('micFooter');"
                " return el && el.textContent.includes('Mic:'); }",
                timeout=30000,
            )
            time.sleep(minutes * 60.0)
            footer = page.evaluate("() => document.getElementById('micFooter').textContent")
            cb_log = page.evaluate("() => window.__cbLog")
            page.click("#micBtn")  # stop -> ws close -> server writes capture
            page.wait_for_function(
                "() => document.getElementById('micBtn').textContent.includes('Start')",
                timeout=15000,
            )
            time.sleep(1.0)  # let the ws close frame reach the server
        finally:
            browser.close()
    return footer, cb_log, console_lines


def _wait_capture(caps_dir: Path, timeout_s: float = 90.0) -> tuple[Path, Path]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        wavs = sorted(caps_dir.glob("*.wav"))
        jsons = sorted(caps_dir.glob("*.json"))
        pairs = [w for w in wavs if w.with_suffix(".json").exists()]
        if pairs:
            w = pairs[0]
            return w, w.with_suffix(".json")
        time.sleep(0.5)
    raise TimeoutError(f"no capture wav+json in {caps_dir} after {timeout_s}s")


def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--generator", default="bonafide_clean")
    ap.add_argument("--split", default="test")
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-noise-dbfs", type=float, default=None, help="add 8.5-20 kHz noise at this RMS dBFS to the fake mic")
    args = ap.parse_args()

    port = args.port if args.port else _free_port()
    tmpdir = Path(tempfile.mkdtemp(prefix="cb_live_mic_"))
    caps_dir = tmpdir / "captures"
    caps_dir.mkdir(parents=True, exist_ok=True)
    server_log = tmpdir / "uvicorn.log"

    print(f"[bench] tmpdir={tmpdir}", flush=True)
    print(f"[bench] building {args.minutes} min of {args.split}/{args.generator} audio ...", flush=True)
    src16 = _build_source16(args.manifest, args.split, args.generator, args.minutes, args.seed)
    fake_wav = tmpdir / "fake_mic_48k.wav"
    _write_fake_mic_wav(src16, fake_wav, hf_noise_dbfs=args.hf_noise_dbfs, seed=args.seed)
    src16_wav = tmpdir / "source_16k.wav"
    save_wav_mono(src16_wav, src16, 16000)
    print(f"[bench] fake mic wav: {fake_wav} ({fake_wav.stat().st_size / 1e6:.1f} MB)", flush=True)

    env = dict(os.environ)
    env.update(
        {
            "CLOAKBREAK_BACKEND": "onnx",
            "CLOAKBREAK_MODEL_PATH": str(args.model_path),
            "CLOAKBREAK_DEBUG_CAPTURE": "1",
            "CLOAKBREAK_CAPTURE_DIR": str(caps_dir),
            "ENABLE_CLONE_ENDPOINT": "false",
            "PYTHONPATH": str(REPO),
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_fh = open(server_log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            str(SERVER_PY), "-m", "uvicorn", "app.main:app",
            "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=REPO,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        health = _wait_health(port)
        print(f"[bench] server up: backend={health.get('backend')} threshold={health.get('threshold')}", flush=True)

        footer, cb_log, console_lines = _browser_session(port, fake_wav, args.minutes)
        print(f"[bench] browser session done; {len(cb_log)} renderResult payloads", flush=True)

        cap_wav, cap_json = _wait_capture(caps_dir)
        print(f"[bench] capture: {cap_wav.name} + {cap_json.name}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_fh.close()

    preds = [
        m for m in cb_log
        if m.get("type") == "prediction" and m.get("is_speech") and m.get("spoof_prob") is not None
    ]
    scores = [float(p["spoof_prob"]) for p in preds]
    threshold = float(health.get("threshold") or (preds[0].get("threshold") if preds else 0.5) or 0.5)
    n_clone = sum(1 for p in preds if p.get("verdict") == "spoof")
    m = re.search(r"resampler:\s*([^·]+)", footer or "")
    resampler_mode = m.group(1).strip() if m else "unknown"

    # Offline re-score of the capture + reference source scoring (pinned .venv).
    replay_out = tmpdir / "replay.json"
    rep_cmd = [
        str(SERVER_PY), "-m", "scripts.replay_capture",
        "--wav", str(cap_wav),
        "--model-path", str(args.model_path),
        "--json", str(cap_json),
        "--reference-wav", str(src16_wav),
        "--out", str(replay_out),
    ]
    print(f"[bench] offline replay: {' '.join(rep_cmd)}", flush=True)
    rep_res = subprocess.run(rep_cmd, cwd=REPO, capture_output=True, text=True)
    if rep_res.returncode != 0:
        print(rep_res.stdout)
        print(rep_res.stderr, file=sys.stderr)
        raise SystemExit("replay_capture failed")
    replay = json.loads(replay_out.read_text(encoding="utf-8"))
    ref = replay.get("reference") or {}

    report = {
        "manifest": str(args.manifest),
        "model_path": str(args.model_path),
        "generator": args.generator,
        "split": args.split,
        "minutes": args.minutes,
        "hf_noise_dbfs": args.hf_noise_dbfs,
        "seed": args.seed,
        "port": port,
        "tmpdir": str(tmpdir),
        "health": health,
        "footer": footer,
        "resampler_mode": resampler_mode,
        "n_render_result_payloads": len(cb_log),
        "n_speech_verdicts": len(preds),
        "score_p50": _pct(scores, 50),
        "score_p95": _pct(scores, 95),
        "score_max": _pct(scores, 100),
        "threshold": threshold,
        "frac_windows_ge_threshold": (sum(1 for s in scores if s >= threshold) / len(scores)) if scores else None,
        "frac_likely_clone": (n_clone / len(preds)) if preds else None,
        "offline_vs_live": {
            "n_windows": replay.get("n_windows"),
            "max_abs_diff": replay.get("max_abs_diff"),
            "mean_abs_diff": replay.get("mean_abs_diff"),
            "frac_offline_ge_threshold": replay.get("frac_offline_ge_threshold"),
        },
        "reference_source": {
            "offset_samples": ref.get("offset_samples"),
            "offset_sec": ref.get("offset_sec"),
            "align_peak": ref.get("align_peak"),
            "n_scored": ref.get("n_scored"),
            "frac_source_ge_threshold": ref.get("frac_source_ge_threshold"),
            "mean_abs_diff_capture_vs_source": ref.get("mean_abs_diff_capture_vs_source"),
            "max_abs_diff_capture_vs_source": ref.get("max_abs_diff_capture_vs_source"),
        },
        "capture_wav": str(cap_wav),
        "capture_json": str(cap_json),
        "replay_json": str(replay_out),
        "server_log": str(server_log),
        "console_tail": console_lines[-20:],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
