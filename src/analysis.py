"""Shared all-window analysis pipeline behind the file/upload/url analysis API.

Every input type (Try-samples clip, multipart upload, yt-dlp download) lands
here as an already-decoded float32 mono 16 kHz waveform. The clip is truncated
to the first 60 s, sliced into NON-overlapping 2.5 s windows, speech-gated per
window, and each speech window is scored through the exact runtime path the
old single-shot endpoints used (``degrade_for_eval`` -> ``detector.predict`` ->
``baseline.predict_prob``). The clip-level verdict is a k-of-n style rule:
LIKELY CLONE iff 2 windows at or above the threshold fall inside some span of
3 consecutive speech windows.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from src.audio_utils import TARGET_SR, SpeechGate, rms_dbfs
from src.augment import EvalCondition, degrade_for_eval
from src.model import BaselineDetectorRuntime, DetectorRuntime

WINDOW_SEC = 2.5
MAX_ANALYZE_SEC = 60.0
VERDICT_RULE = "2 of any 3 consecutive speech windows at or above the threshold"


class ClipTooShortError(ValueError):
    """Raised when the clip is shorter than the minimum analysis window."""


def window_start_samples(n_samples: int, sr: int = TARGET_SR, window_sec: float = WINDOW_SEC) -> List[int]:
    """Non-overlapping window starts at 0, 2.5, 5.0, ... while start+2.5 s fits.

    Falls back to a single window at 0 when no full window fits (the runtime
    pads short windows internally); callers reject clips under the minimum
    window length before reaching this.
    """
    win = int(round(window_sec * sr))
    starts: List[int] = []
    pos = 0
    while pos + win <= n_samples:
        starts.append(pos)
        pos += win
    if not starts:
        starts = [0]
    return starts


def analyze_clip(
    waveform: np.ndarray,
    condition: EvalCondition,
    *,
    detector: DetectorRuntime,
    baseline: Optional[BaselineDetectorRuntime],
    threshold: float,
    min_window_sec: float,
    input_type: str,
    sample_id: Optional[str] = None,
    truth: Optional[str] = None,
    url: Optional[str] = None,
) -> dict:
    """Run the shared analysis pipeline on a decoded 16 kHz mono waveform.

    ``input_type`` is one of "sample" | "upload" | "url". Raises
    ``ClipTooShortError`` for clips shorter than ``min_window_sec`` seconds.
    """
    t0 = time.perf_counter()
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    max_samples = int(round(MAX_ANALYZE_SEC * TARGET_SR))
    if x.size > max_samples:
        x = x[:max_samples]
    duration_s = x.size / float(TARGET_SR)
    if duration_s < float(min_window_sec):
        raise ClipTooShortError(
            f"need at least {float(min_window_sec):.1f} s of audio "
            "(clips under 1 s are at chance)"
        )

    win_samples = int(round(WINDOW_SEC * TARGET_SR))
    starts = window_start_samples(x.size, sr=TARGET_SR, window_sec=WINDOW_SEC)

    # One gate per clip: the whole clip is pushed through feed-forward first
    # (order is irrelevant for files), then each window is classified.
    gate = SpeechGate(sr=TARGET_SR)
    gate.push(x)

    windows: List[dict] = []
    speech_probs: List[float] = []
    for pos in starts:
        w = x[pos : pos + win_samples]
        start_s = pos / float(TARGET_SR)
        w_dur_s = w.size / float(TARGET_SR)
        is_speech = bool(gate.is_speech(w))
        entry = {
            "start": round(start_s, 3),
            "end": round(start_s + w_dur_s, 3),
            "dbfs": round(rms_dbfs(w), 1),
            "is_speech": is_speech,
            "spoof_prob": None,
            "baseline_prob": None,
            "window_verdict": None,
            "attention_focus": None,
        }
        if is_speech:
            degraded = degrade_for_eval(w, condition=condition, sr=TARGET_SR)
            res = detector.predict(degraded)
            prob = float(res.spoof_prob)
            entry["spoof_prob"] = prob
            entry["baseline_prob"] = (
                float(baseline.predict_prob(degraded)) if baseline is not None else None
            )
            entry["window_verdict"] = "spoof" if prob >= threshold else "bonafide"
            entry["attention_focus"] = res.attention_focus
            speech_probs.append(prob)
        windows.append(entry)

    over = [p >= threshold for p in speech_probs]
    n_over = sum(1 for o in over if o)
    n_speech = len(speech_probs)
    spoof = False
    if n_speech == 1:
        # A single speech window decides alone.
        spoof = bool(over[0])
    elif n_speech >= 2:
        # 2 hits inside some span of 3 consecutive speech windows <=> a pair of
        # hits at speech-sequence distance <= 2 (adjacent-hit scan suffices:
        # non-adjacent hit pairs are never closer).
        hit_idx = [i for i, o in enumerate(over) if o]
        for a, b in zip(hit_idx, hit_idx[1:]):
            if b - a <= 2:
                spoof = True
                break

    latency_ms = (time.perf_counter() - t0) * 1000.0
    out = {
        "type": input_type,
        "condition": condition,
        "duration_s": round(duration_s, 2),
        "threshold": float(threshold),
        "verdict": "spoof" if spoof else "bonafide",
        "verdict_rule": VERDICT_RULE,
        "n_windows": len(windows),
        "n_speech": n_speech,
        "n_over_threshold": n_over,
        "frac_over_threshold": round(n_over / n_speech, 4) if n_speech else 0.0,
        "max_spoof_prob": round(max(speech_probs), 4) if speech_probs else None,
        "latency_ms": round(latency_ms, 2),
        "backend": detector.backend_name,
        "windows": windows,
    }
    if sample_id is not None:
        out["sample_id"] = sample_id
    if truth is not None:
        out["truth"] = truth
    if url is not None:
        out["url"] = url
    return out
