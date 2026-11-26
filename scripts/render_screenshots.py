"""Render CloakBreak UI screenshots with Playwright.

Playwright is NOT a project dependency — run it through uv (the pinned
playwright 1.65.2 drives chromium-1255, already in ~/.cache/ms-playwright):

    uv run --with playwright==1.65.2 python scripts/render_screenshots.py          # REAL mode
    uv run --with playwright==1.65.2 python scripts/render_screenshots.py --mock    # MOCK mode

If the browser is missing:  uv run --with playwright==1.65.2 playwright install chromium

REAL mode (default) starts ``uvicorn app.main:app`` on a free port with the
real ONNX detector + Hugging Face baseline
(CLOAKBREAK_BACKEND=onnx, CLOAKBREAK_MODEL_PATH=artifacts/session2/onnx/cloakbreak_fp32.onnx,
CLOAKBREAK_BASELINE_ID=Vansh180/deepfake-audio-wav2vec2, ENABLE_CLONE_ENDPOINT=false),
waits for /api/health, opens the Try samples tab (the default), keeps the
channel on Clean, clicks each of the six Score buttons in order waiting for
every result, then captures desktop (1280x900) and mobile (390x844,
deviceScaleFactor 2) full-page shots into
``docs/screenshots/real-try-samples-{desktop,mobile}.png``. The six rows'
displayed score/verdict text plus the health threshold is dumped to
``results/try_samples_real.json`` and printed. Model loading on a small VM
takes a while — the health wait is generous.

MOCK mode (--mock) uses CLOAKBREAK_BACKEND=mock with an intercepted
/api/health payload and drives ``window.renderResult(...)`` with deterministic
seeded live-mic payloads, saving ``docs/screenshots/mock-live-clone-desktop.png``
and ``mock-live-warming-mobile.png``. A visible ribbon reading
"MOCK — illustrative data, not a measurement" is injected via page script in
this mode only — the mocked shots must never be mistaken for measurements.
"""

from __future__ import annotations

import argparse
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
RESULTS_DIR = REPO_ROOT / "results"
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
SERVER_PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

THRESHOLD = 0.95255
INFERENCE_MS = 633.0
LAG_MS = 681.0
LATENCY_MS = 630.0

SAMPLE_IDS = [
    "kokoro_00483",
    "kokoro_00069",
    "kokoro_00454",
    "ls_8463-294828-0031",
    "ls_8455-210777-0040",
    "ls_672-122797-0057",
]

MOCK_RIBBON = (
    "MOCK — illustrative data, not a measurement"
)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _wait_health(port: int, proc: subprocess.Popen, timeout_s: float = 60.0) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.monotonic() + timeout_s
    last_err: Any = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"uvicorn exited early (rc={proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - keep polling until deadline
            last_err = exc
        time.sleep(0.5)
    raise SystemExit(f"/api/health did not come up within {timeout_s}s (last error: {last_err})")


def _start_server(env_extra: Dict[str, str]) -> tuple[subprocess.Popen, int]:
    port = _free_port()
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [SERVER_PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc, port


def _stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


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
                "input_dbfs": round(-41.0 - 0.1 * i, 1),
                "noise_floor_dbfs": round(-54.0 + 0.05 * i, 1),
                "speech_gate_dbfs": round(-48.0 + 0.05 * i, 1),
            }
        )
    return out


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
            "input_dbfs": -46.5,
            "noise_floor_dbfs": -54.0,
            "speech_gate_dbfs": -48.0,
        }
    ]


MOCK_HEALTH = {
    "status": "ok",
    "backend": "onnx:cloakbreak_fp32.onnx",
    "baseline_loaded": True,
    "baseline_id": "Vansh180/deepfake-audio-wav2vec2",
    "clone_enabled": False,
    "threshold": THRESHOLD,
    "threshold_source": "file",
    "debug_capture": False,
    "samples_available": 6,
}


def _inject_mock_ribbon(page) -> None:
    page.evaluate(
        """(text) => {
            const d = document.createElement('div');
            d.id = 'mockRibbon';
            d.textContent = text;
            d.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:9999;'
                + 'background:#7f1d1d;color:#fee2e2;text-align:center;'
                + 'font:600 13px system-ui,sans-serif;padding:8px 12px;'
                + 'letter-spacing:0.02em;';
            document.body.appendChild(d);
        }""",
        MOCK_RIBBON,
    )


# Every descendant of .sample-row / .card must stay inside its container box
# (1 px tolerance for fractional layout). Zero-size rects are skipped so
# display:none subtrees (hidden panels, clone panel) don't false-positive.
OVERFLOW_JS = """() => {
  const bad = [];
  const fmt = (el) => {
    const cls = typeof el.className === 'string' && el.className.trim()
      ? '.' + el.className.trim().split(/\\s+/).join('.')
      : '';
    return el.tagName.toLowerCase() + cls;
  };
  document.querySelectorAll('.sample-row, .card').forEach((box) => {
    const r = box.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    const boxSel = box.id ? '#' + box.id : fmt(box);
    box.querySelectorAll('*').forEach((el) => {
      const er = el.getBoundingClientRect();
      if (er.width === 0 && er.height === 0) return;
      if (er.right > r.right + 1 || er.left < r.left - 1) {
        bad.push(boxSel + ' > ' + fmt(el)
          + ' right=' + er.right.toFixed(1) + '/' + r.right.toFixed(1)
          + ' left=' + er.left.toFixed(1) + '/' + r.left.toFixed(1));
      }
    });
  });
  return bad;
}"""


def _assert_no_overflow(page, label: str) -> None:
    bad = page.evaluate(OVERFLOW_JS)
    if bad:
        raise SystemExit(
            f"[shots] HORIZONTAL OVERFLOW in {label} — {len(bad)} descendant(s) escape their box:\n  "
            + "\n  ".join(bad[:25])
        )
    print(f"[shots] overflow check OK — {label}", flush=True)


def real_mode() -> None:
    """Score all six allowlisted samples with the real models; screenshot."""
    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    proc, port = _start_server(
        {
            "CLOAKBREAK_BACKEND": "onnx",
            "CLOAKBREAK_MODEL_PATH": "artifacts/session2/onnx/cloakbreak_fp32.onnx",
            "CLOAKBREAK_BASELINE_ID": "Vansh180/deepfake-audio-wav2vec2",
            "ENABLE_CLONE_ENDPOINT": "false",
        }
    )
    try:
        health = _wait_health(port, proc, timeout_s=600.0)
        print(f"[shots:real] uvicorn up on :{port}; health={json.dumps(health)[:200]}", flush=True)
        base_url = f"http://127.0.0.1:{port}/"

        def score_all(page) -> List[Dict[str, str]]:
            """Click every Score button in order; return the displayed row text."""
            rows = page.evaluate(
                """() => [...document.querySelectorAll('.sample-row')].map(r => ({
                    id: r.dataset.id,
                    name: r.querySelector('.sample-name').textContent,
                    truth: (r.querySelector('.sample-sub .truth') || {}).textContent || '',
                    cb: (r.querySelector('.res-val.cb') || {}).textContent || '',
                    legacy: (r.querySelector('.res-val.bl') || {}).textContent || '',
                }))"""
            )
            assert len(rows) == len(SAMPLE_IDS), f"expected {len(SAMPLE_IDS)} rows, got {len(rows)}"
            for i, sid in enumerate(SAMPLE_IDS):
                assert rows[i]["id"] == sid, f"row {i} id {rows[i]['id']} != {sid}"
                page.click(f".sample-row[data-id='{sid}'] .score-btn")
                page.wait_for_function(
                    """(sid) => {
                        const r = document.querySelector(`.sample-row[data-id='${sid}']`);
                        const el = r && r.querySelector('.res-val.cb');
                        return el && /\\d+\\.\\d{3}/.test(el.textContent);
                    }""",
                    arg=sid,
                    timeout=180000,
                )
                page.wait_for_timeout(150)
            return page.evaluate(
                """() => [...document.querySelectorAll('.sample-row')].map(r => ({
                    id: r.dataset.id,
                    name: r.querySelector('.sample-name').textContent,
                    truth: (r.querySelector('.sample-sub .truth') || {}).textContent || '',
                    cb: (r.querySelector('.res-val.cb') || {}).textContent || '',
                    legacy: (r.querySelector('.res-val.bl') || {}).textContent || '',
                }))"""
            )

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                # Desktop pass: score all six, capture, dump row text.
                ctx = browser.new_context(viewport={"width": 1280, "height": 900}, device_scale_factor=1)
                page = ctx.new_page()
                page.goto(base_url, wait_until="load")
                page.wait_for_function(
                    "() => document.querySelectorAll('.sample-row').length === 6",
                    timeout=30000,
                )
                page.select_option("#conditionSel", "clean")
                rows_desktop = score_all(page)
                page.wait_for_timeout(300)
                _assert_no_overflow(page, "real-try-samples-desktop")
                page.screenshot(path=str(OUT_DIR / "real-try-samples-desktop.png"), full_page=True)
                print("[shots:real] docs/screenshots/real-try-samples-desktop.png", flush=True)
                ctx.close()

                # Mobile pass (fresh page, rescore — same deterministic results).
                ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2)
                page = ctx.new_page()
                page.goto(base_url, wait_until="load")
                page.wait_for_function(
                    "() => document.querySelectorAll('.sample-row').length === 6",
                    timeout=30000,
                )
                page.select_option("#conditionSel", "clean")
                rows_mobile = score_all(page)
                page.wait_for_timeout(300)
                _assert_no_overflow(page, "real-try-samples-mobile")
                page.screenshot(path=str(OUT_DIR / "real-try-samples-mobile.png"), full_page=True)
                print("[shots:real] docs/screenshots/real-try-samples-mobile.png", flush=True)
                ctx.close()
            finally:
                browser.close()

        report = {
            "threshold": health.get("threshold"),
            "backend": health.get("backend"),
            "baseline_id": health.get("baseline_id"),
            "rows_desktop": rows_desktop,
            "rows_mobile": rows_mobile,
        }
        out_json = RESULTS_DIR / "try_samples_real.json"
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[shots:real] wrote {out_json.relative_to(REPO_ROOT)}")
        print(json.dumps(report, indent=2))
    finally:
        _stop_server(proc)


def mock_mode() -> None:
    """Mock backend + intercepted health; deterministic live-mic states."""
    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    states = {
        "clone": _clone_payloads(rng),
        "warming": _warming_payload(),
    }

    proc, port = _start_server({"CLOAKBREAK_BACKEND": "mock", "ENABLE_CLONE_ENDPOINT": "false"})
    try:
        _wait_health(port, proc, timeout_s=60.0)
        print(f"[shots:mock] uvicorn up on :{port}")
        base_url = f"http://127.0.0.1:{port}/"

        def health_route(route) -> None:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(MOCK_HEALTH))

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                shots = (
                    ("clone", "desktop", {"width": 1280, "height": 900}, 1, "mock-live-clone-desktop.png"),
                    ("warming", "mobile", {"width": 390, "height": 844}, 2, "mock-live-warming-mobile.png"),
                )
                for state_name, vp_name, viewport, dsf, fname in shots:
                    ctx = browser.new_context(viewport=viewport, device_scale_factor=dsf)
                    ctx.route("**/api/health", health_route)
                    page = ctx.new_page()
                    page.goto(base_url, wait_until="load")
                    page.wait_for_function(
                        "() => !document.getElementById('backendBadge')"
                        ".textContent.includes('connecting')",
                        timeout=15000,
                    )
                    page.click("#tab-live")  # live-mic state -> show the live panel
                    page.evaluate(
                        "ps => { for (const p of ps) window.renderResult(p); }",
                        states[state_name],
                    )
                    page.wait_for_timeout(300)
                    _inject_mock_ribbon(page)
                    page.wait_for_timeout(100)
                    _assert_no_overflow(page, f"mock-live-{state_name}-{vp_name}")
                    out = OUT_DIR / fname
                    page.screenshot(path=str(out), full_page=True)
                    print(f"[shots:mock] {out.relative_to(REPO_ROOT)}")
                    ctx.close()
            finally:
                browser.close()
    finally:
        _stop_server(proc)


def _clear_old_pngs() -> None:
    """Remove screenshots from the previous (dark, mocked) naming scheme."""
    for old in OUT_DIR.glob("*.png"):
        if old.name.startswith(("desktop-", "mobile-")):
            old.unlink()
            print(f"[shots] removed stale {old.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="mock backend + illustrative payloads")
    args = parser.parse_args()
    _clear_old_pngs()
    if args.mock:
        mock_mode()
    else:
        real_mode()


if __name__ == "__main__":
    main()
