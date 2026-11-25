from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
from scipy.signal import butter, filtfilt

from src.audio_utils import TARGET_SR, load_wav_mono, normalize_loudness, resample_audio, save_wav_mono

EvalCondition = Literal["clean", "opus_16k", "phone_8k", "noise_10db", "noise_5db", "short_0.5s", "short_0.8s"]


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def apply_mulaw_compand(waveform: np.ndarray, mu: float = 255.0) -> np.ndarray:
    x = np.clip(waveform.astype(np.float32), -1.0, 1.0)
    encoded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.round((encoded + 1.0) * 127.5) / 127.5 - 1.0
    decoded = np.sign(quantized) * (1.0 / mu) * ((1.0 + mu) ** np.abs(quantized) - 1.0)
    return decoded.astype(np.float32)


def apply_phone_8k(waveform: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    if waveform.size < 32:
        return waveform.astype(np.float32, copy=False)
    nyq = 0.5 * sr
    b, a = butter(4, [300.0 / nyq, min(3400.0 / nyq, 0.95)], btype="band")
    filtered = filtfilt(b, a, waveform.astype(np.float64)).astype(np.float32)
    down = resample_audio(filtered, sr, 8000)
    companded = apply_mulaw_compand(down)
    up = resample_audio(companded, 8000, sr)
    if up.shape[0] > waveform.shape[0]:
        up = up[: waveform.shape[0]]
    elif up.shape[0] < waveform.shape[0]:
        up = np.pad(up, (0, waveform.shape[0] - up.shape[0]))
    return np.clip(up, -1.0, 1.0).astype(np.float32)


def apply_opus_codec(waveform: np.ndarray, sr: int = TARGET_SR, bitrate_kbps: int = 16) -> np.ndarray:
    if not has_ffmpeg():
        return apply_phone_8k(waveform, sr)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_wav = tmp / "in.wav"
        opus_file = tmp / "enc.opus"
        out_wav = tmp / "out.wav"
        save_wav_mono(in_wav, waveform, sr)
        enc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(in_wav),
                "-c:a", "libopus",
                "-b:a", f"{bitrate_kbps}k",
                "-vbr", "on",
                str(opus_file),
            ],
            capture_output=True,
        )
        if enc.returncode != 0:
            return apply_phone_8k(waveform, sr)
        dec = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(opus_file),
                "-ar", str(sr),
                "-ac", "1",
                str(out_wav),
            ],
            capture_output=True,
        )
        if dec.returncode != 0 or not out_wav.exists():
            return apply_phone_8k(waveform, sr)
        decoded, _ = load_wav_mono(out_wav, target_sr=sr)
        if decoded.shape[0] > waveform.shape[0]:
            decoded = decoded[: waveform.shape[0]]
        elif decoded.shape[0] < waveform.shape[0]:
            decoded = np.pad(decoded, (0, waveform.shape[0] - decoded.shape[0]))
        return np.clip(decoded, -1.0, 1.0).astype(np.float32)


def _precompute_one_opus(args: Tuple[str, str, int]) -> Tuple[str, str]:
    src_str, dst_str, bitrate_kbps = args
    dst = Path(dst_str)
    if not (dst.exists() and dst.stat().st_size > 100):
        wav, sr = load_wav_mono(src_str, target_sr=TARGET_SR)
        degraded = apply_opus_codec(wav, sr=sr, bitrate_kbps=bitrate_kbps)
        degraded = normalize_loudness(degraded)
        save_wav_mono(dst, degraded, TARGET_SR)
    return src_str, dst_str


def precompute_opus_cache(
    wav_paths: List[str],
    cache_dir: str | Path,
    bitrate_kbps: int = 16,
    num_workers: int | None = None,
) -> Dict[str, str]:
    """Pre-compute Opus-degraded WAV variants once in parallel across vCPUs.

    Eliminates per-step ffmpeg subprocess spawning inside training DataLoader
    workers so the GPU never sits idle waiting on CPU subprocesses.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    unique_paths = sorted(set(wav_paths))
    tasks: List[Tuple[str, str, int]] = []
    for p in unique_paths:
        key = hashlib.sha1(f"{p}:{bitrate_kbps}".encode("utf-8")).hexdigest()[:16]
        dst = cache_path / f"{key}.wav"
        tasks.append((p, str(dst), bitrate_kbps))

    workers = num_workers if num_workers is not None else min(os.cpu_count() or 2, 12)
    mapping: Dict[str, str] = {}
    if workers <= 1 or len(tasks) <= 4:
        for item in tasks:
            src_s, dst_s = _precompute_one_opus(item)
            mapping[src_s] = dst_s
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for src_s, dst_s in ex.map(_precompute_one_opus, tasks, chunksize=8):
                mapping[src_s] = dst_s
    return mapping


def apply_additive_noise(
    waveform: np.ndarray,
    snr_db: float = 15.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    sig_power = float(np.mean(waveform.astype(np.float64) ** 2)) + 1e-10
    noise = rng.standard_normal(waveform.shape[0]).astype(np.float64)
    noise_power = float(np.mean(noise ** 2)) + 1e-10
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    return np.clip(waveform + scaled_noise, -1.0, 1.0).astype(np.float32)


def apply_soft_clipping(waveform: np.ndarray, drive: float = 1.8) -> np.ndarray:
    return (np.tanh(waveform * drive) / np.tanh(drive)).astype(np.float32)


SPECTRAL_AUG_RANGES: Dict[str, List[float]] = {
    "lowpass_fc_hz": [3400.0, 8000.0],  # high end clamped to 0.95 * nyquist
    "low_shelf_gain_db": [-6.0, 6.0],
    "low_shelf_corner_hz": [100.0, 500.0],
    "high_shelf_gain_db": [-6.0, 6.0],
    "high_shelf_corner_hz": [1000.0, 4000.0],
    "tilt_db_per_octave": [-1.5, 1.5],
}


def _sample_spectral_params(
    rng: np.random.Generator,
    sr: int,
    p_lowpass: float = 0.4,
    p_low_shelf: float = 0.6,
    p_high_shelf: float = 0.6,
    p_tilt: float = 0.6,
) -> Dict[str, Dict[str, float]]:
    """Draw every spectral-augmentation decision/parameter from ``rng``.

    The draw order is fixed and never inspects the waveform, so real and fake
    clips receive identical transform distributions for a given seed.
    """
    fc_hi = max(3400.0, min(8000.0, 0.95 * 0.5 * sr))
    return {
        "lowpass": {
            "apply": bool(rng.random() < p_lowpass),
            "fc_hz": float(rng.uniform(3400.0, fc_hi)),
        },
        "low_shelf": {
            "apply": bool(rng.random() < p_low_shelf),
            "gain_db": float(rng.uniform(*SPECTRAL_AUG_RANGES["low_shelf_gain_db"])),
            "corner_hz": float(rng.uniform(*SPECTRAL_AUG_RANGES["low_shelf_corner_hz"])),
        },
        "high_shelf": {
            "apply": bool(rng.random() < p_high_shelf),
            "gain_db": float(rng.uniform(*SPECTRAL_AUG_RANGES["high_shelf_gain_db"])),
            "corner_hz": float(rng.uniform(*SPECTRAL_AUG_RANGES["high_shelf_corner_hz"])),
        },
        "tilt": {
            "apply": bool(rng.random() < p_tilt),
            "tilt_db_per_octave": float(rng.uniform(*SPECTRAL_AUG_RANGES["tilt_db_per_octave"])),
        },
    }


def _shelf_weight(freqs: np.ndarray, corner_hz: float, width_octaves: float = 0.3) -> np.ndarray:
    """Smooth 0 -> 1 logistic step in log2-frequency, ~1 octave wide, at ``corner_hz``."""
    u = np.log2(np.maximum(freqs, 1.0) / corner_hz)
    return 1.0 / (1.0 + np.exp(-u / width_octaves))


def _spectral_gain_linear(freqs: np.ndarray, params: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Combined linear amplitude gain curve (length = len(freqs)) for one rfft."""
    f = freqs.astype(np.float64)
    gain_db = np.zeros(f.shape[0], dtype=np.float64)

    tilt = params.get("tilt") or {}
    if tilt.get("apply"):
        gain_db += float(tilt["tilt_db_per_octave"]) * np.log2(np.maximum(f, 50.0) / 1000.0)

    low = params.get("low_shelf") or {}
    if low.get("apply"):
        gain_db += float(low["gain_db"]) * (1.0 - _shelf_weight(f, float(low["corner_hz"])))

    high = params.get("high_shelf") or {}
    if high.get("apply"):
        gain_db += float(high["gain_db"]) * _shelf_weight(f, float(high["corner_hz"]))

    lowpass = params.get("lowpass") or {}
    if lowpass.get("apply"):
        fc = float(lowpass["fc_hz"])
        gain_db += -10.0 * np.log10(1.0 + (f / fc) ** 8)

    return np.power(10.0, gain_db / 20.0)


def apply_spectral_augment(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
    rng: np.random.Generator | None = None,
    params: Dict[str, Dict[str, float]] | None = None,
) -> np.ndarray:
    """Class-agnostic spectral augmentation: one rfft gain curve, then irfft.

    Length-preserving, float32, finite. When ``params`` is omitted they are drawn
    via :func:`_sample_spectral_params` so callers never branch on content.
    """
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if x.size < 8:
        return x.astype(np.float32, copy=False)
    if params is None:
        params = _sample_spectral_params(rng or np.random.default_rng(), sr)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sr))
    y = np.fft.irfft(spec * _spectral_gain_linear(freqs, params), n=x.size)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y.astype(np.float32)


def degrade_for_eval(
    waveform: np.ndarray,
    condition: EvalCondition,
    sr: int = TARGET_SR,
    seed: int = 1234,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if condition == "clean":
        return normalize_loudness(waveform)
    if condition == "opus_16k":
        return normalize_loudness(apply_opus_codec(waveform, sr=sr, bitrate_kbps=16))
    if condition == "phone_8k":
        return normalize_loudness(apply_phone_8k(waveform, sr=sr))
    if condition == "noise_10db":
        return normalize_loudness(apply_additive_noise(waveform, snr_db=10.0, rng=rng))
    if condition == "noise_5db":
        return normalize_loudness(apply_additive_noise(waveform, snr_db=5.0, rng=rng))
    if condition == "short_0.5s":
        return normalize_loudness(waveform[: int(0.5 * sr)])
    if condition == "short_0.8s":
        return normalize_loudness(waveform[: int(0.8 * sr)])
    raise ValueError(f"Unknown eval condition: {condition}")


class TrainCodecAugmenter:
    """Fast CPU-worker augmentation without per-step ffmpeg blocking.

    When opus_cache is provided, Opus codec augmentation loads the pre-cached
    Opus variant directly from disk (microseconds) instead of spawning ffmpeg.
    """

    def __init__(
        self,
        p_codec: float = 0.6,
        p_noise: float = 0.5,
        p_clip: float = 0.15,
        p_lowpass: float = 0.4,
        p_low_shelf: float = 0.6,
        p_high_shelf: float = 0.6,
        p_tilt: float = 0.6,
        allow_inline_opus: bool = False,
        seed: int | None = None,
    ) -> None:
        self.p_codec = p_codec
        self.p_noise = p_noise
        self.p_clip = p_clip
        self.p_lowpass = p_lowpass
        self.p_low_shelf = p_low_shelf
        self.p_high_shelf = p_high_shelf
        self.p_tilt = p_tilt
        self.allow_inline_opus = allow_inline_opus and has_ffmpeg()
        self.rng = np.random.default_rng(seed)
        self.last_spectral_params: Dict[str, Dict[str, float]] | None = None

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        waveform: np.ndarray,
        sr: int = TARGET_SR,
        cached_opus_waveform: np.ndarray | None = None,
    ) -> np.ndarray:
        x = waveform.astype(np.float32, copy=True)
        if self.rng.random() < self.p_codec:
            if cached_opus_waveform is not None and self.rng.random() < 0.55:
                x = cached_opus_waveform.astype(np.float32, copy=True)
            elif self.allow_inline_opus and self.rng.random() < 0.5:
                bitrate = int(self.rng.choice([12, 16, 24]))
                x = apply_opus_codec(x, sr=sr, bitrate_kbps=bitrate)
            else:
                x = apply_phone_8k(x, sr=sr)
        spectral_params = _sample_spectral_params(
            self.rng, sr, self.p_lowpass, self.p_low_shelf, self.p_high_shelf, self.p_tilt
        )
        self.last_spectral_params = spectral_params
        x = apply_spectral_augment(x, sr=sr, params=spectral_params)
        if self.rng.random() < self.p_noise:
            snr = float(self.rng.uniform(8.0, 25.0))
            x = apply_additive_noise(x, snr_db=snr, rng=self.rng)
        if self.rng.random() < self.p_clip:
            drive = float(self.rng.uniform(1.3, 2.5))
            x = apply_soft_clipping(x, drive=drive)
        return normalize_loudness(x)
