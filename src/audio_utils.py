from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
TARGET_RMS = 0.0708  # approx -23 dBFS
PEAK_CEILING = 0.95


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    gcd = int(np.gcd(orig_sr, target_sr))
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(waveform, up, down).astype(np.float32)


def load_wav_mono(path: str | Path, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    mono = resample_audio(mono, sr, target_sr)
    return mono, target_sr


def load_wav_bytes_mono(data: bytes, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    mono = resample_audio(mono, sr, target_sr)
    return mono, target_sr


def save_wav_mono(path: str | Path, waveform: np.ndarray, sr: int = TARGET_SR) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(waveform.astype(np.float32), -1.0, 1.0)
    sf.write(str(path), clipped, sr, subtype="PCM_16")


def wav_to_bytes(waveform: np.ndarray, sr: int = TARGET_SR) -> bytes:
    buf = io.BytesIO()
    clipped = np.clip(waveform.astype(np.float32), -1.0, 1.0)
    sf.write(buf, clipped, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(pcm_bytes[:usable], dtype=np.int16)
    return (samples.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def float32_to_pcm16_bytes(waveform: np.ndarray) -> bytes:
    clipped = np.clip(waveform, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def audio_rms(waveform: np.ndarray) -> float:
    if waveform.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))


def normalize_peak(waveform: np.ndarray, target_peak: float = PEAK_CEILING) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak < 1e-6:
        return waveform.astype(np.float32, copy=False)
    return (waveform * (target_peak / peak)).astype(np.float32)


def normalize_loudness(
    waveform: np.ndarray,
    target_rms: float = TARGET_RMS,
    peak_ceiling: float = PEAK_CEILING,
) -> np.ndarray:
    """Normalize clip RMS loudness and apply peak ceiling identically across real & spoof."""
    x = waveform.astype(np.float32, copy=False)
    rms = audio_rms(x)
    if rms < 1e-6:
        return x
    scaled = x * (target_rms / rms)
    peak = float(np.max(np.abs(scaled)))
    if peak > peak_ceiling:
        scaled = scaled * (peak_ceiling / peak)
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def pad_or_crop(
    waveform: np.ndarray,
    num_samples: int,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    n = waveform.shape[0]
    if n == num_samples:
        return waveform.astype(np.float32, copy=False)
    if n > num_samples:
        if random_crop:
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, n - num_samples + 1))
        else:
            start = (n - num_samples) // 2
        return waveform[start : start + num_samples].astype(np.float32, copy=False)
    out = np.zeros(num_samples, dtype=np.float32)
    out[:n] = waveform
    return out


def rms_vad(waveform: np.ndarray, threshold: float = 0.012) -> bool:
    return audio_rms(waveform) >= threshold


class RollingAudioBuffer:
    def __init__(self, window_sec: float = 2.5, stride_sec: float = 0.25, sr: int = TARGET_SR) -> None:
        self.sr = sr
        self.window_samples = int(round(window_sec * sr))
        self.stride_samples = int(round(stride_sec * sr))
        self._buf = np.zeros(self.window_samples, dtype=np.float32)
        self._since_last_emit = 0
        self._total_received = 0

    def push(self, chunk: np.ndarray) -> bool:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        n = chunk.shape[0]
        if n == 0:
            return False
        if n >= self.window_samples:
            self._buf[:] = chunk[-self.window_samples :]
        else:
            self._buf[:-n] = self._buf[n:]
            self._buf[-n:] = chunk
        self._since_last_emit += n
        self._total_received += n
        if self._since_last_emit >= self.stride_samples:
            self._since_last_emit = 0
            return True
        return False

    def get_window(self) -> np.ndarray:
        return self._buf.copy()

    @property
    def filled_sec(self) -> float:
        return min(self._total_received, self.window_samples) / float(self.sr)
