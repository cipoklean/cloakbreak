"""Node-run checks for the JS ``StreamResampler16k`` fallback resampler.

The block between ``// BEGIN StreamResampler16k`` / ``// END StreamResampler16k``
in frontend/index.html is plain ES2015; this test extracts it, feeds float32
audio through it in Node in 4096-sample chunks (same as the page), and
compares against ``src.audio_utils.resample_audio``.

Skipped when ``node`` is not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, correlate, resample_poly, sosfiltfilt, welch

from src.audio_utils import TARGET_SR, resample_audio

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_HTML = REPO_ROOT / "frontend" / "index.html"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node executable not on PATH")

BEGIN = "// BEGIN StreamResampler16k"
END = "// END StreamResampler16k"
DUR_S = 6.0
CHUNK = 4096


def _resampler_block() -> str:
    html = FRONTEND_HTML.read_text(encoding="utf-8")
    i = html.index(BEGIN)
    j = html.index(END)
    return html[i:j]


def _expected_ntaps(in_rate: int) -> int:
    n = int(round(129 * in_rate / 48000))
    if n % 2 == 0:
        n += 1
    return max(n, 33)


def _run_node(in_rate: int, samples: np.ndarray, tmp_path: Path, chunk: int = CHUNK) -> np.ndarray:
    """Feed float32 samples through the JS resampler in Node; return float32 out."""
    inp = tmp_path / f"in_{in_rate}_{chunk}.f32"
    outp = tmp_path / f"out_{in_rate}_{chunk}.f32"
    drv = tmp_path / f"drv_{in_rate}_{chunk}.js"
    inp.write_bytes(samples.astype("<f4").tobytes())
    drv.write_text(
        _resampler_block()
        + f"""
const fs = require('fs');
const buf = fs.readFileSync({str(inp)!r});
const f32 = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.length / 4));
const r = new StreamResampler16k({in_rate});
const parts = [];
for (let k = 0; k < f32.length; k += {chunk}) {{
  const o = r.process(f32.subarray(k, Math.min(k + {chunk}, f32.length)));
  parts.push(Buffer.from(o.buffer, o.byteOffset, o.byteLength));
}}
fs.writeFileSync({str(outp)!r}, Buffer.concat(parts));
""",
        encoding="utf-8",
    )
    subprocess.run([NODE, str(drv)], check=True, capture_output=True, text=True)
    return np.fromfile(outp, dtype="<f4")


def _log_sweep(in_rate: int, dur_s: float = DUR_S) -> np.ndarray:
    """6 s log sweep 50 Hz -> 20 kHz (phase integral of f(t)=50*exp(k t/T))."""
    n = int(in_rate * dur_s)
    t = np.arange(n, dtype=np.float64) / in_rate
    k = np.log(20000.0 / 50.0)
    phase = 2.0 * np.pi * 50.0 * (dur_s / k) * (np.exp(k * t / dur_s) - 1.0)
    return (0.5 * np.sin(phase)).astype(np.float32)


def _speech_like(in_rate: int, dur_s: float = DUR_S) -> np.ndarray:
    """Harmonic speech-like clip at in_rate, plus a 12 kHz tone at -20 dBFS."""
    n16 = int(TARGET_SR * dur_s)
    t16 = np.arange(n16, dtype=np.float32) / TARGET_SR
    rng = np.random.default_rng(0)
    env = 0.55 + 0.45 * np.sin(2.0 * np.pi * 2.7 * t16)
    sig = np.zeros(n16, dtype=np.float32)
    for h in range(1, 40):
        sig += (0.5 / h) * np.sin(2.0 * np.pi * 140.0 * h * t16 + rng.uniform(0.0, 2.0 * np.pi))
    sig *= env
    sig += rng.normal(0.0, 0.008, size=n16).astype(np.float32)
    sig = np.clip(sig / np.max(np.abs(sig)) * 0.5, -1.0, 1.0).astype(np.float32)
    if in_rate == TARGET_SR:
        base = sig
    else:
        # 16000 -> in_rate (44100 = 441/160, 48000 = 3/1)
        g = int(np.gcd(in_rate, TARGET_SR))
        base = resample_poly(sig.astype(np.float64), in_rate // g, TARGET_SR // g).astype(np.float32)
    t = np.arange(base.shape[0], dtype=np.float64) / in_rate
    tone = (0.1 * np.sin(2.0 * np.pi * 12000.0 * t)).astype(np.float32)
    return np.clip(base + tone, -1.0, 1.0).astype(np.float32)


def _naive_decimate(x: np.ndarray, in_rate: int) -> np.ndarray:
    """The old every-Nth-sample decimation (no anti-alias filter)."""
    if in_rate % TARGET_SR == 0:
        return x[:: in_rate // TARGET_SR].astype(np.float32)
    ratio = in_rate / TARGET_SR
    out_len = int(len(x) / ratio)
    return x[np.floor(np.arange(out_len) * ratio).astype(np.int64)].astype(np.float32)


def _rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


def _sweep_alias_db(out: np.ndarray, in_rate: int, dur_s: float = DUR_S) -> float:
    """dB ratio of output energy while the sweep is >8.5 kHz vs 0.2-6 kHz."""
    k = np.log(20000.0 / 50.0)
    t_hi = dur_s * np.log(8500.0 / 50.0) / k
    t_lo0 = dur_s * np.log(200.0 / 50.0) / k
    t_lo1 = dur_s * np.log(6000.0 / 50.0) / k
    tt = np.arange(len(out), dtype=np.float64) / TARGET_SR
    m_hi = (tt >= t_hi + 0.10) & (tt <= dur_s - 0.05)
    m_mid = (tt >= t_lo0 + 0.10) & (tt <= t_lo1 - 0.10)
    return _rms_db(out[m_hi]) - _rms_db(out[m_mid])


def _alias_tone_db(out: np.ndarray) -> float:
    """Coherent 4 kHz tone level: bandpass 3.9-4.1 kHz, Goertzel at 4 kHz.

    The band also carries genuine speech, so raw band RMS cannot isolate the
    aliased 12 kHz tone; the Goertzel bin at exactly 4 kHz measures the
    tone-coherent component (speech contributes ~1/sqrt(N) leakage).
    """
    sos = butter(8, [3900.0 / (TARGET_SR / 2.0), 4100.0 / (TARGET_SR / 2.0)], btype="band", output="sos")
    y = sosfiltfilt(sos, out.astype(np.float64))
    n = np.arange(y.shape[0], dtype=np.float64)
    amp = 2.0 * abs(np.sum(y * np.exp(-2j * np.pi * 4000.0 * n / TARGET_SR))) / y.shape[0]
    return float(20.0 * np.log10(amp + 1e-12))


def _align_and_maxdiff(js: np.ndarray, ref: np.ndarray, in_rate: int) -> float:
    """Max |js - ref| after aligning the FIR group delay.

    The FIR group delay is (ntaps-1)/2 input samples; we locate the best lag
    by cross-correlation restricted to +/-8 output samples around the
    theoretical delay (avoids picking a spurious peak on periodic speech),
    then compare interiors (first/last 0.1 s trimmed: FIR startup and the
    emit-tail edge transient are not part of steady-state fidelity).
    """
    n_out = min(len(js), len(ref))
    js = js[:n_out]
    ref = ref[:n_out]
    step = in_rate / TARGET_SR
    theory = (_expected_ntaps(in_rate) - 1) / 2.0 / step
    lo = max(0, int(round(theory)) - 8)
    hi = min(n_out - 1, int(round(theory)) + 8)
    c = np.abs(correlate(ref, js, mode="full"))
    center = n_out - 1
    # js lagging ref by s samples peaks at index center - s.
    idx_lo, idx_hi = center - hi, center - lo
    best_idx = idx_lo + int(np.argmax(c[idx_lo : idx_hi + 1]))
    lag = center - best_idx  # js[t + lag] ~= ref[t]
    if lag >= 0:
        aligned = np.concatenate([js[lag:], np.zeros(lag, np.float32)])[:n_out]
    else:
        aligned = np.concatenate([np.zeros(-lag, np.float32), js])[:n_out]
    trim = int(0.1 * TARGET_SR)
    return float(np.max(np.abs(aligned[trim:-trim].astype(np.float64) - ref[trim:-trim].astype(np.float64))))


def _check_length(out: np.ndarray, dur_s: float) -> None:
    expected = dur_s * TARGET_SR
    assert abs(len(out) - expected) <= 2.0 * dur_s, (
        f"output length {len(out)} vs expected {expected} (tolerance 2 samples/s)"
    )


def test_resampler_block_markers() -> None:
    block = _resampler_block()
    assert "class StreamResampler16k" in block
    assert "passthrough" in block


@pytest.mark.parametrize("in_rate", [48000, 44100])
def test_resampler_sweep_alias_and_length(in_rate: int, tmp_path: Path) -> None:
    sweep = _log_sweep(in_rate)
    js = _run_node(in_rate, sweep, tmp_path)
    naive = _naive_decimate(sweep, in_rate)
    n = min(len(js), len(naive))
    js_db = _sweep_alias_db(js[:n], in_rate)
    naive_db = _sweep_alias_db(naive[:n], in_rate)
    # Spec: alias energy >8.5 kHz must be >=40 dB below the 0.2-6 kHz band.
    # Measured ~ -96 dB @48k / -98 dB @44.1k; naive decimation lands ~0 dB.
    assert js_db < -40.0, f"JS alias ratio {js_db:.1f} dB at {in_rate} Hz"
    assert naive_db > -40.0, "naive decimation should FAIL the same check (test sensitivity)"
    _check_length(js, DUR_S)


@pytest.mark.parametrize("in_rate", [48000, 44100])
def test_resampler_speech_alias_spectral_and_diff(in_rate: int, tmp_path: Path) -> None:
    speech = _speech_like(in_rate)
    js = _run_node(in_rate, speech, tmp_path)
    naive = _naive_decimate(speech, in_rate)
    ref = resample_audio(speech, in_rate, TARGET_SR)
    _check_length(js, DUR_S)

    # 12 kHz tone must not alias into 4 kHz: coherent-tone level at least
    # 40 dB below naive decimation. Measured ~81 dB @48k / ~79 dB @44.1k.
    js_tone = _alias_tone_db(js)
    naive_tone = _alias_tone_db(naive)
    assert naive_tone - js_tone >= 40.0, (
        f"4 kHz alias suppression {naive_tone - js_tone:.1f} dB at {in_rate} Hz "
        f"(js {js_tone:.1f} dB, naive {naive_tone:.1f} dB)"
    )

    # Spectral match vs resample_audio: Welch log-mag MAD 100 Hz-7 kHz < 1 dB.
    # Measured 0.11 dB @48k / 0.33 dB @44.1k.
    f, p_js = welch(js, TARGET_SR, nperseg=1024)
    f, p_ref = welch(ref, TARGET_SR, nperseg=1024)
    mask = (f >= 100.0) & (f <= 7000.0)
    mad = float(np.mean(np.abs(10.0 * np.log10(p_js[mask]) - 10.0 * np.log10(p_ref[mask]))))
    assert mad < 1.0, f"spectral MAD {mad:.3f} dB at {in_rate} Hz"

    # Steady-state fidelity: interior max abs diff after group-delay alignment.
    # Measured 0.060 @48k / 0.072 @44.1k on this signal; bound 0.1 covers
    # passband ripple differences + linear-interp error while still catching
    # chunk-boundary clicks (a discontinuity would spike far above this).
    md = _align_and_maxdiff(js, ref, in_rate)
    assert md < 0.1, f"max abs diff {md:.4f} at {in_rate} Hz"


def test_resampler_chunk_boundaries_identical(tmp_path: Path) -> None:
    """4096-sample chunked feeding must equal one-shot processing bit-for-bit."""
    speech = _speech_like(48000)
    chunked = _run_node(48000, speech, tmp_path, chunk=CHUNK)
    whole = _run_node(48000, speech, tmp_path, chunk=len(speech))
    n = min(len(chunked), len(whole))
    np.testing.assert_array_equal(chunked[:n], whole[:n])
    _check_length(chunked, DUR_S)
