from __future__ import annotations

import io
from collections import deque
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


def estimate_snr_db(waveform: np.ndarray, sr: int = TARGET_SR, frame_ms: float = 25.0) -> float:
    """Estimate signal-to-noise-floor ratio (dB) from 90th vs 10th percentile frame RMS."""
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    frame_len = max(16, int(round(sr * frame_ms / 1000.0)))
    n_frames = x.size // frame_len
    if n_frames < 4:
        return 0.0
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    sig_level = float(np.percentile(frame_rms, 90))
    noise_floor = max(float(np.percentile(frame_rms, 10)), 1e-6)
    if sig_level <= 1e-6:
        return 0.0
    snr = 20.0 * np.log10(sig_level / noise_floor)
    return float(np.clip(snr, 0.0, 80.0))


def spectral_rolloff_hz(waveform: np.ndarray, sr: int = TARGET_SR, roll_percent: float = 0.85) -> float:
    """Frequency (Hz) below which roll_percent of total spectral energy resides."""
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if x.size < 32:
        return 0.0
    mag_sq = np.abs(np.fft.rfft(x)) ** 2
    total = float(np.sum(mag_sq))
    if total <= 1e-12:
        return 0.0
    cum = np.cumsum(mag_sq)
    idx = int(np.searchsorted(cum, roll_percent * total))
    idx = min(idx, mag_sq.size - 1)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sr))
    return float(freqs[idx])


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


def rms_dbfs(waveform: np.ndarray) -> float:
    """Window RMS in dBFS, floored at -120 (digital silence)."""
    rms = audio_rms(waveform)
    if rms <= 0.0:
        return -120.0
    return float(max(-120.0, 20.0 * np.log10(rms)))


# Blocks louder than this are treated as signal, not noise floor: real mic
# noise floors sit below -40 dBFS, and without the ceiling continuous loud
# speech would raise its own 10th-percentile "floor" above the gate and reject
# itself.
_FLOOR_BLOCK_MAX_DBFS = -30.0
_FLOOR_BLOCK_MIN_DBFS = -90.0  # digital silence never enters the floor


class SpeechGate:
    """Noise-floor-relative speech gate for live microphone streams.

    Quiet headset mics deliver speech near -40 dBFS, so a fixed RMS threshold
    either rejects real speech or accepts the noise floor. This gate tracks the
    rolling 10th-percentile block level (the noise floor) and opens the gate
    ``margin_db`` above it, clamped to ``[min_gate_dbfs, ...]``; with too little
    history it falls back to ``fallback_gate_dbfs``.
    """

    def __init__(
        self,
        sr: int = TARGET_SR,
        block_sec: float = 0.25,
        history_sec: float = 10.0,
        percentile: float = 10.0,
        margin_db: float = 6.0,
        min_gate_dbfs: float = -60.0,
        fallback_gate_dbfs: float = -50.0,
        min_history_blocks: int = 4,
    ) -> None:
        self.sr = int(sr)
        self.block_samples = max(1, int(round(block_sec * self.sr)))
        self.percentile = float(percentile)
        self.margin_db = float(margin_db)
        self.min_gate_dbfs = float(min_gate_dbfs)
        self.fallback_gate_dbfs = float(fallback_gate_dbfs)
        self.min_history_blocks = int(min_history_blocks)
        self._history = deque(maxlen=max(1, int(round(history_sec / block_sec))))
        self._carry = np.zeros(0, dtype=np.float32)

    def push(self, chunk: np.ndarray) -> None:
        """Feed one incoming chunk; accumulates into 0.25 s blocks across pushes."""
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        data = np.concatenate([self._carry, x]) if self._carry.size else x
        n_full = data.size // self.block_samples
        for i in range(n_full):
            blk = data[i * self.block_samples : (i + 1) * self.block_samples]
            db = rms_dbfs(blk)
            if _FLOOR_BLOCK_MIN_DBFS < db < _FLOOR_BLOCK_MAX_DBFS:
                self._history.append(db)
        self._carry = data[n_full * self.block_samples :]

    def noise_floor_dbfs(self) -> float | None:
        if len(self._history) < self.min_history_blocks:
            return None
        return float(np.percentile(np.fromiter(self._history, dtype=np.float64), self.percentile))

    def gate_dbfs(self) -> float:
        floor = self.noise_floor_dbfs()
        if floor is None:
            return self.fallback_gate_dbfs
        return max(self.min_gate_dbfs, floor + self.margin_db)

    def is_speech(self, window: np.ndarray) -> bool:
        return rms_dbfs(window) >= self.gate_dbfs()


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
