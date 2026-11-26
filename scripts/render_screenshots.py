"""Render real CloakBreak React-UI screenshots with Playwright.

Playwright is NOT a project dependency — run it through uv:

    uv run --with playwright==1.65.2 python scripts/render_screenshots.py

The script starts ``uvicorn app.main:app`` on a random free port with the real
fp32 ONNX detector and the Hugging Face legacy baseline, drives the production
React build in headless Chromium, and writes eight real screenshots into
``docs/screenshots/``:

    real-{samples-kokoro,samples-librispeech,upload,busy}-{desktop,mobile}.png

Desktop is 1440x900; mobile is 390x844 with deviceScaleFactor 2. It also
rebuilds ``results/try_samples_real.json`` from the values actually displayed
by the UI for all six Try-samples clips. There is no mock mode.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "screenshots"
RESULTS_DIR = REPO_ROOT / "results"
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
SERVER_PY = str(VENV_PY) if VENV_PY.exists() else sys.executable
MODEL_PATH = REPO_ROOT / "artifacts" / "session2" / "onnx" / "cloakbreak_fp32.onnx"
BASELINE_ID = "Vansh180/deepfake-audio-wav2vec2"
FORBIDDEN_PORTS = {8765, 8000, 8088}

SAMPLE_IDS = [
    "kokoro_00483",
    "kokoro_00069",
    "kokoro_00454",
    "ls_8463-294828-0031",
    "ls_8455-210777-0040",
    "ls_672-122797-0057",
]
KOKORO_SHOT = "kokoro_00483"
LIBRISPEECH_SHOT = "ls_8463-294828-0031"

DESKTOP = {"width": 1440, "height": 900}
MOBILE = {"width": 390, "height": 844}

OLD_PNGS = [
    "mock-live-clone-desktop.png",
    "mock-live-warming-mobile.png",
    "real-try-samples-desktop.png",
    "real-try-samples-mobile.png",
]


def _free_port() -> int:
    while True:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
        sock.close()
        if port not in FORBIDDEN_PORTS:
            return port


def _wait_health(port: int, proc: subprocess.Popen, timeout_s: float = 600.0) -> dict[str, Any]:
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
        except Exception as exc:  # keep polling until deadline
            last_err = exc
        time.sleep(0.5)
    raise SystemExit(f"/api/health did not come up within {timeout_s}s (last error: {last_err})")


def _start_server(workdir: Path) -> tuple[subprocess.Popen, int, Path]:
    port = _free_port()
    log_path = workdir / "uvicorn-screenshots.log"
    env = dict(os.environ)
    env.pop("CLOAKBREAK_WEB_DIST", None)
    env.update(
        {
            "CLOAKBREAK_BACKEND": "onnx",
            "CLOAKBREAK_MODEL_PATH": str(MODEL_PATH),
            "CLOAKBREAK_BASELINE_ID": BASELINE_ID,
            "ENABLE_CLONE_ENDPOINT": "false",
            "PYTHONPATH": str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [SERVER_PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    proc._cloakbreak_log_fh = log_fh  # type: ignore[attr-defined] - kept alive with the process
    return proc, port, log_path


def _stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    log_fh = getattr(proc, "_cloakbreak_log_fh", None)
    if log_fh is not None:
        log_fh.close()


# Every descendant of a UI container must stay inside that container's box
# (1 px tolerance for fractional layout). Zero-size rects are skipped so
# hidden tab panels do not false-positive.
OVERFLOW_JS = """() => {
  const bad = [];
  const fmt = (el) => {
    const cls = typeof el.className === 'string' && el.className.trim()
      ? '.' + el.className.trim().split(/\\s+/).join('.')
      : '';
    const tid = el.getAttribute && el.getAttribute('data-testid');
    return el.tagName.toLowerCase() + (tid ? '[data-testid=' + tid + ']' : cls);
  };
  const boxes = document.querySelectorAll(
    '[data-testid=input-card], [data-testid=verdict-card], [data-testid^=sample-card-], .card'
  );
  boxes.forEach((box) => {
    const r = box.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    box.querySelectorAll('*').forEach((el) => {
      const er = el.getBoundingClientRect();
      if (er.width === 0 && er.height === 0) return;
      if (er.right > r.right + 1 || er.left < r.left - 1) {
        bad.push(fmt(box) + ' > ' + fmt(el)
          + ' right=' + er.right.toFixed(1) + '/' + r.right.toFixed(1)
          + ' left=' + er.left.toFixed(1) + '/' + r.left.toFixed(1));
      }
    });
  });
  if (document.documentElement.scrollWidth > window.innerWidth + 1) {
    bad.push('document scrollWidth=' + document.documentElement.scrollWidth + ' innerWidth=' + window.innerWidth);
  }
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


def _wait_contains(page, selector: str, text: str, timeout: float) -> None:
    page.wait_for_function(
        """({selector, text}) => {
            const el = document.querySelector(selector);
            return el && el.textContent && el.textContent.includes(text);
        }""",
        arg={"selector": selector, "text": text},
        timeout=timeout,
    )


def _capture(page, path: Path, label: str) -> None:
    page.wait_for_timeout(350)  # let the score-bar width transitions settle
    _assert_no_overflow(page, label)
    page.screenshot(path=str(path), full_page=True)
    print(f"[shots] {path.relative_to(REPO_ROOT)}", flush=True)


def _new_page(browser, viewport: dict[str, int], scale: float, base_url: str):
    ctx = browser.new_context(viewport=viewport, device_scale_factor=scale)
    page = ctx.new_page()
    page.goto(base_url, wait_until="load")
    return ctx, page


def _wait_samples(page) -> None:
    page.wait_for_function(
        "() => document.querySelectorAll('[data-testid^=sample-card-]').length === 6",
        timeout=30000,
    )


def _analyze_sample(page, sample_id: str) -> dict[str, Any]:
    button = page.locator(f'[data-testid="analyze-sample-{sample_id}"]')
    button.wait_for(state="visible", timeout=15000)
    page.wait_for_function(
        "(sid) => !(document.querySelector(`[data-testid=analyze-sample-${sid}]`) || {}).disabled",
        arg=sample_id,
        timeout=180000,
    )
    button.click()
    page.locator(f'[data-testid="sample-result-{sample_id}"]').wait_for(state="visible", timeout=180000)
    page.wait_for_timeout(150)
    rows = page.evaluate(
        """() => [...document.querySelectorAll('[data-testid^=window-row-]')]
            .map((row) => {
              const idx = Number((row.getAttribute('data-testid') || 'window-row--1').replace('window-row-', ''));
              return {
                index: idx,
                time: (row.children[0] || {}).textContent?.trim() || '',
                cloakbreak: (row.querySelector('[data-testid=window-cb-' + idx + ']') || {}).textContent?.trim() || '',
                legacy: (row.querySelector('[data-testid=window-legacy-' + idx + ']') || {}).textContent?.trim() || '',
                state: (row.children[3] || {}).textContent?.trim() || '',
              };
            })
            .sort((a, b) => a.index - b.index)"""
    )
    over_text = page.locator(f'[data-testid="sample-over-{sample_id}"]').inner_text().strip()
    n_over, n_speech = [int(part) for part in over_text.split("/", 1)]
    return {
        "id": sample_id,
        "overall_verdict": page.locator(f'[data-testid="sample-verdict-{sample_id}"]').inner_text().strip(),
        "max_window_score": page.locator(f'[data-testid="sample-max-{sample_id}"]').inner_text().strip(),
        "windows_over_threshold": over_text,
        "n_over_threshold": n_over,
        "n_speech": n_speech,
        "truth_label": page.locator(f'[data-testid="truth-pill-{sample_id}"]').inner_text().strip(),
        "windows": rows,
    }


def _show_sample_result(page, sample_id: str, label_fragment: str) -> None:
    page.locator(f'[data-testid="show-result-{sample_id}"]').click()
    _wait_contains(page, '[data-testid="timeline-meta"]', label_fragment, 15000)


def _select_upload_file(page, path: Path) -> None:
    page.locator('[data-testid="upload-file-input"]').set_input_files(str(path))
    _wait_contains(page, '[data-testid="selected-file-name"]', path.name, 10000)


def _open_upload_tab(page) -> None:
    page.locator('[data-testid="tab-upload"]').click()
    page.locator('[data-testid="upload-file-input"]').wait_for(state="visible", timeout=10000)


def _make_media(workdir: Path) -> tuple[Path, Path]:
    """Return (upload_mp3, slow_50s_wav), both through real ffmpeg encode paths."""
    src = REPO_ROOT / "data" / "cloakbreak_v1" / "kokoro_unseen" / "kokoro_00069.wav"
    if not src.is_file():
        raise SystemExit(f"missing media source {src}")
    mp3 = workdir / "kokoro_00069_upload.mp3"
    conv = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-codec:a", "libmp3lame", "-q:a", "4", str(mp3)],
        capture_output=True,
        text=True,
    )
    if conv.returncode != 0:
        raise SystemExit(f"ffmpeg mp3 encode failed: {conv.stderr}")

    concat_list = workdir / "concat.txt"
    concat_list.write_text("".join(f"file '{src}'\n" for _ in range(9)), encoding="utf-8")  # ~53 s
    slow = workdir / "slow_50s.wav"
    conv = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(slow),
        ],
        capture_output=True,
        text=True,
    )
    if conv.returncode != 0:
        raise SystemExit(f"ffmpeg slow-wav concat failed: {conv.stderr}")
    return mp3, slow


def _samples_desktop(page, browser, base_url: str) -> list[dict[str, Any]]:
    _wait_samples(page)
    entries = []
    for sample_id in SAMPLE_IDS:
        print(f"[shots] analyzing sample {sample_id} (desktop results capture)", flush=True)
        entries.append(_analyze_sample(page, sample_id))

    _show_sample_result(page, KOKORO_SHOT, "Kokoro clip 00483")
    _capture(page, OUT_DIR / "real-samples-kokoro-desktop.png", "real-samples-kokoro-desktop")
    _show_sample_result(page, LIBRISPEECH_SHOT, "LibriSpeech 8463-294828-0031")
    _capture(page, OUT_DIR / "real-samples-librispeech-desktop.png", "real-samples-librispeech-desktop")
    return entries


def _samples_mobile(browser, base_url: str) -> None:
    ctx, page = _new_page(browser, MOBILE, 2, base_url)
    try:
        _wait_samples(page)
        print("[shots] analyzing kokoro sample (mobile)", flush=True)
        _analyze_sample(page, KOKORO_SHOT)
        _capture(page, OUT_DIR / "real-samples-kokoro-mobile.png", "real-samples-kokoro-mobile")
        print("[shots] analyzing LibriSpeech sample (mobile)", flush=True)
        _analyze_sample(page, LIBRISPEECH_SHOT)
        _capture(page, OUT_DIR / "real-samples-librispeech-mobile.png", "real-samples-librispeech-mobile")
    finally:
        ctx.close()


def _upload_scenario(browser, base_url: str, upload_mp3: Path, viewport: dict[str, int], scale: float, suffix: str) -> None:
    ctx, page = _new_page(browser, viewport, scale, base_url)
    try:
        _open_upload_tab(page)
        _select_upload_file(page, upload_mp3)
        page.locator('[data-testid="analyze-upload-button"]').click()
        page.locator('[data-testid="timeline-meta"]').wait_for(state="visible", timeout=180000)
        _wait_contains(page, '[data-testid="timeline-meta"]', upload_mp3.name, 15000)
        _capture(page, OUT_DIR / f"real-upload-{suffix}.png", f"real-upload-{suffix}")
    finally:
        ctx.close()


def _busy_scenario(browser, base_url: str, slow_wav: Path, upload_mp3: Path, viewport: dict[str, int], scale: float, suffix: str) -> None:
    ctx = browser.new_context(viewport=viewport, device_scale_factor=scale)
    try:
        first = ctx.new_page()
        first.goto(base_url, wait_until="load")
        _open_upload_tab(first)
        _select_upload_file(first, slow_wav)
        first.locator('[data-testid="analyze-upload-button"]').click()
        _wait_contains(first, '[data-testid="job-status"]', "Analysis is running.", 30000)

        second = ctx.new_page()
        second.goto(base_url, wait_until="load")
        _open_upload_tab(second)
        _select_upload_file(second, upload_mp3)
        second.locator('[data-testid="analyze-upload-button"]').click()
        _wait_contains(
            second,
            '[data-testid="job-status"]',
            "Someone else is being analyzed. You're #1 in line.",
            30000,
        )
        _capture(second, OUT_DIR / f"real-busy-{suffix}.png", f"real-busy-{suffix}")

        # Leave the queue empty for the next viewport/scenario: the first job
        # analyzes twenty windows, then the second job runs behind it.
        second.locator('[data-testid="timeline-meta"]').wait_for(state="visible", timeout=240000)
    finally:
        ctx.close()


def real_mode() -> None:
    from playwright.sync_api import sync_playwright

    if not MODEL_PATH.is_file():
        raise SystemExit(f"missing ONNX model: {MODEL_PATH}")
    if not (REPO_ROOT / "web" / "dist" / "index.html").is_file():
        raise SystemExit("web/dist/index.html is missing — run: cd web && npm ci && npm run build")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in OLD_PNGS:
        old = OUT_DIR / name
        if old.exists():
            old.unlink()
            print(f"[shots] removed stale {name}", flush=True)

    workdir = Path(tempfile.mkdtemp(prefix="cb_screenshots_"))
    upload_mp3, slow_wav = _make_media(workdir)
    print(f"[shots] media: {upload_mp3.name} ({upload_mp3.stat().st_size} bytes), {slow_wav.name} ({slow_wav.stat().st_size} bytes)", flush=True)

    proc, port, log_path = _start_server(workdir)
    try:
        health = _wait_health(port, proc, timeout_s=600.0)
        print(f"[shots] uvicorn up on :{port}; backend={health.get('backend')} threshold={health.get('threshold')}", flush=True)
        if health.get("baseline_loaded") is not True:
            raise SystemExit(f"legacy baseline did not load; see {log_path}")
        base_url = f"http://127.0.0.1:{port}/"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                # Desktop samples pass: all six analyses are also the data source
                # for results/try_samples_real.json.
                ctx, page = _new_page(browser, DESKTOP, 1, base_url)
                try:
                    entries = _samples_desktop(page, browser, base_url)
                finally:
                    ctx.close()

                _samples_mobile(browser, base_url)
                _upload_scenario(browser, base_url, upload_mp3, DESKTOP, 1, "desktop")
                _upload_scenario(browser, base_url, upload_mp3, MOBILE, 2, "mobile")
                _busy_scenario(browser, base_url, slow_wav, upload_mp3, DESKTOP, 1, "desktop")
                _busy_scenario(browser, base_url, slow_wav, upload_mp3, MOBILE, 2, "mobile")
            finally:
                browser.close()

        report = {
            "threshold": health.get("threshold"),
            "backend": health.get("backend"),
            "baseline_id": health.get("baseline_id"),
            "condition": "clean",
            "source": "Values captured from the rendered React UI by scripts/render_screenshots.py",
            "samples": entries,
        }
        out_json = RESULTS_DIR / "try_samples_real.json"
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[shots] wrote {out_json.relative_to(REPO_ROOT)}", flush=True)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        _stop_server(proc)


def main() -> None:
    real_mode()


if __name__ == "__main__":
    main()
