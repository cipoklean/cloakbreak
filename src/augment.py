from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import butter, filtfilt

from src.audio_utils import TARGET_SR, load_wav_mono, resample_audio, save_wav_mono

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


def degrade_for_eval(
    waveform: np.ndarray,
    condition: EvalCondition,
    sr: int = TARGET_SR,
    seed: int = 1234,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if condition == "clean":
        return waveform.astype(np.float32, copy=False)
    if condition == "opus_16k":
        return apply_opus_codec(waveform, sr=sr, bitrate_kbps=16)
    if condition == "phone_8k":
        return apply_phone_8k(waveform, sr=sr)
    if condition == "noise_10db":
        return apply_additive_noise(waveform, snr_db=10.0, rng=rng)
    if condition == "noise_5db":
        return apply_additive_noise(waveform, snr_db=5.0, rng=rng)
    if condition == "short_0.5s":
        return waveform[: int(0.5 * sr)].astype(np.float32, copy=False)
    if condition == "short_0.8s":
        return waveform[: int(0.8 * sr)].astype(np.float32, copy=False)
    raise ValueError(f"Unknown eval condition: {condition}")


class TrainCodecAugmenter:
    def __init__(
        self,
        p_codec: float = 0.6,
        p_noise: float = 0.5,
        p_clip: float = 0.15,
        allow_opus: bool = True,
        seed: int | None = None,
    ) -> None:
        self.p_codec = p_codec
        self.p_noise = p_noise
        self.p_clip = p_clip
        self.allow_opus = allow_opus and has_ffmpeg()
        self.rng = np.random.default_rng(seed)

    def __call__(self, waveform: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        x = waveform.astype(np.float32, copy=True)
        if self.rng.random() < self.p_codec:
            if self.allow_opus and self.rng.random() < 0.5:
                bitrate = int(self.rng.choice([12, 16, 24]))
                x = apply_opus_codec(x, sr=sr, bitrate_kbps=bitrate)
            else:
                x = apply_phone_8k(x, sr=sr)
        if self.rng.random() < self.p_noise:
            snr = float(self.rng.uniform(8.0, 25.0))
            x = apply_additive_noise(x, snr_db=snr, rng=self.rng)
        if self.rng.random() < self.p_clip:
            drive = float(self.rng.uniform(1.3, 2.5))
            x = apply_soft_clipping(x, drive=drive)
        return np.clip(x, -1.0, 1.0).astype(np.float32)
