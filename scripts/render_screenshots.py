"""Render demo-state screenshots of the CloakBreak UI with Playwright.

Playwright is NOT a project dependency — run it through uv (the pinned
playwright 1.65.2 drives chromium-1255, already in ~/.cache/ms-playwright):

    uv run --with playwright==1.65.2 python scripts/render_screenshots.py

If the browser is missing:  uv run --with playwright==1.65.2 playwright install chromium

What it does: starts ``uvicorn app.main:app`` (CLOAKBREAK_BACKEND=mock,
ENABLE_CLONE_ENDPOINT=false) on a free port using ``.venv/bin/python``,
intercepts ``GET /api/health`` with a fixed payload, loads the page, then
drives ``window.renderResult(...)`` with deterministic seeded payloads for
three states (real / clone / warming) at desktop (1280x900) and mobile
(390x844, deviceScaleFactor 2) viewports, writing
``docs/screenshots/{desktop,mobile}-{real,clone,warming}.png``.

These are mocked UI states for documentation — not measurements.
"""

from __future__ import annotations

import json
import math
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "screenshots"
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
SERVER_PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

THRESHOLD = 0.95255
INFERENCE_MS = 633.0
LAG_MS = 681.0
LATENCY_MS = 630.0

HEALTH = {
    "status": "ok",
    "backend": "onnx:cloakbreak_fp32.onnx",
    "baseline_loaded": True,
    "baseline_id": "Vansh180/deepfake-audio-wav2vec2",
    "clone_enabled": False,
    "threshold": THRESHOLD,
    "threshold_source": "file",
}


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _wait_health(port: int, proc: subprocess.Popen, timeout_s: float = 60.0) -> None:
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"uvicorn exited early (rc={proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.4)
    raise SystemExit(f"/api/health did not come up within {timeout_s}s")


def _attention_focus(rng: random.Random) -> List[float]:
    """32 smooth attention bumps, normalized to max 1."""
    bumps = [(0.22, 0.07, 0.85), (0.55, 0.11, 1.0), (0.82, 0.06, 0.45)]
    vals = []
    for i in range(32):
        x = i / 31.0
        v = sum(a * math.exp(-((x - c) ** 2) / (2 * w * w)) for c, w, a in bumps)
        v += rng.uniform(0.0, 0.04)
        vals.append(v)
    m = max(vals) or 1.0
    return [round(v / m, 4) for v in vals]


def _seq_payloads(probs: List[float], base_probs: List[float], rng: random.Random) -> List[Dict[str, Any]]:
    """Build live prediction payloads with server-consistent smoothed verdicts."""
    hist: deque = deque(maxlen=3)
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(probs):
        hist.append(p)
        hits = sum(1 for x in hist if x >= THRESHOLD)
        verdict = "spoof" if hits >= 2 else "bonafide"
        window_verdict = "spoof" if p >= THRESHOLD else "bonafide"
        audio_end = round(2.5 + i * 2.1, 3)
        out.append(
            {
                "type": "prediction",
                "is_speech": True,
                "spoof_prob": p,
                "baseline_prob": base_probs[i],
                "window_verdict": window_verdict,
                "verdict": verdict,
                "smoothing": "2_of_3",
                "recent_scores": [round(float(x), 4) for x in hist],
                "threshold": THRESHOLD,
                "audio_start_sec": round(audio_end - 2.5, 3),
                "audio_end_sec": audio_end,
                "stream_sec_received": round(audio_end + LAG_MS / 1000.0, 3),
                "inference_ms": INFERENCE_MS,
                "lag_ms": LAG_MS,
                "dropped_windows": i,
                "attention_focus": _attention_focus(rng),
                "latency_ms": LATENCY_MS,
                "backend": "onnx",
                "condition": "clean",
                "window_sec": 2.5,
            }
        )
    return out


def _real_payloads(rng: random.Random) -> List[Dict[str, Any]]:
    probs = [round(rng.uniform(0.62, 0.62), 4) if i == 17 else round(rng.uniform(0.02, 0.30), 4) for i in range(30)]
    base = [round(rng.uniform(0.05, 0.25), 4) for _ in range(30)]
    return _seq_payloads(probs, base, rng)


def _clone_payloads(rng: random.Random) -> List[Dict[str, Any]]:
    real_probs = [round(rng.uniform(0.02, 0.30), 4) for _ in range(18)]
    clone_probs = [round(rng.uniform(0.97, 0.999), 4) for _ in range(12)]
    base = [round(rng.uniform(0.06, 0.20), 4) for _ in range(30)]
    return _seq_payloads(real_probs + clone_probs, base, rng)


def _warming_payload() -> List[Dict[str, Any]]:
    return [
        {
            "type": "prediction",
            "warming_up": True,
            "is_speech": True,
            "spoof_prob": None,
            "baseline_prob": None,
            "window_verdict": None,
            "verdict": None,
            "smoothing": "2_of_3",
            "recent_scores": [],
            "threshold": THRESHOLD,
            "audio_start_sec": 0.0,
            "audio_end_sec": 1.2,
            "stream_sec_received": 1.2,
            "inference_ms": 0.0,
            "lag_ms": 0.0,
            "dropped_windows": 0,
            "attention_focus": [0.0] * 32,
            "latency_ms": 0.0,
            "backend": "onnx",
            "condition": "clean",
            "window_sec": 1.2,
        }
    ]


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    env = dict(os.environ)
    env.update(
        {
            "CLOAKBREAK_BACKEND": "mock",
            "ENABLE_CLONE_ENDPOINT": "false",
            "PYTHONPATH": str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    proc = subprocess.Popen(
        [SERVER_PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_health(port, proc)
        print(f"[shots] uvicorn up on :{port}")

        rng = random.Random(42)
        states = {
            "real": _real_payloads(rng),
            "clone": _clone_payloads(rng),
            "warming": _warming_payload(),
        }
        base_url = f"http://127.0.0.1:{port}/"

        def health_route(route) -> None:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(HEALTH))

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                for vp_name, viewport, dsf in (
                    ("desktop", {"width": 1280, "height": 900}, 1),
                    ("mobile", {"width": 390, "height": 844}, 2),
                ):
                    ctx = browser.new_context(viewport=viewport, device_scale_factor=dsf)
                    ctx.route("**/api/health", health_route)
                    for state_name, payloads in states.items():
                        page = ctx.new_page()
                        page.goto(base_url, wait_until="load")
                        page.wait_for_function(
                            "() => !document.getElementById('backendBadge')"
                            ".textContent.includes('connecting')",
                            timeout=15000,
                        )
                        page.evaluate(
                            "ps => { for (const p of ps) window.renderResult(p); }",
                            payloads,
                        )
                        page.wait_for_timeout(200)
                        out = OUT_DIR / f"{vp_name}-{state_name}.png"
                        page.screenshot(path=str(out), full_page=True)
                        print(f"[shots] {out.relative_to(REPO_ROOT)}")
                        page.close()
                    ctx.close()
            finally:
                browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    print(f"[shots] done -> {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
