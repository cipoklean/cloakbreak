"""Live-streaming scorer for the /ws/detect demo.

The websocket used to run the detector synchronously on the event loop for
every stride; on a 2-vCPU ARM VM a 2.5 s window takes ~0.7-1.2 s, so work
queued without bound. ``LiveScorer`` decouples ingest (``push`` is O(chunk),
never blocks, never infers) from a single-worker async loop that always scores
the NEWEST buffered window — superseded strides are dropped and counted, never
queued — so emitted verdicts stay bounded-lag no matter how slow the model is.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Dict, Optional

import numpy as np

from src.audio_utils import RollingAudioBuffer, rms_vad

EmitFn = Callable[[Dict[str, Any]], Awaitable[None]]
ScoreFn = Callable[[np.ndarray], Dict[str, Any]]


class LiveScorer:
    """Newest-window live scorer with k-of-n verdict smoothing.

    ``score_fn(window)`` runs in a worker thread via ``asyncio.to_thread`` and
    must return a dict with at least ``spoof_prob``, ``attention_focus``,
    ``latency_ms`` and ``backend`` (``baseline_prob`` optional).
    """

    def __init__(
        self,
        score_fn: ScoreFn,
        threshold: float,
        sr: int = 16000,
        window_sec: float = 2.5,
        min_window_sec: float = 2.0,
        stride_sec: float = 0.25,
        smooth_k: int = 2,
        smooth_n: int = 3,
        clock=time.monotonic,
        max_consecutive_failures: int = 3,
    ) -> None:
        self.score_fn = score_fn
        self.threshold = float(threshold)
        self.sr = int(sr)
        self.window_sec = float(window_sec)
        self.min_window_sec = float(min_window_sec)
        self.stride_sec = float(stride_sec)
        self.smooth_k = int(smooth_k)
        self.smooth_n = int(smooth_n)
        self._clock = clock
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.consecutive_failures = 0

        self._ring = RollingAudioBuffer(window_sec=window_sec, stride_sec=stride_sec, sr=sr)
        self._window_samples = self._ring.window_samples
        self._stride_samples = self._ring.stride_samples

        self._event = asyncio.Event()
        self._stop = False

        # ingest state — written by push() from the receiver coroutine
        self.total_samples = 0
        self.n_pushed_chunks = 0
        self.last_arrival: Optional[float] = None

        # worker state
        self._last_consumed_samples = 0
        self._ever_consumed = False
        self._last_warm_stride_idx = -1
        self._history: Deque[float] = deque(maxlen=max(1, self.smooth_n))
        self._pending_windows = 0
        self._inflight = 0

        # observability counters (read by tests and the {"type":"stats"} reply)
        self.inflight_max = 0
        self.n_scored = 0
        self.max_pending_windows = 0
        self.dropped_windows = 0

    # -- ingest ---------------------------------------------------------------

    def push(self, samples: np.ndarray) -> None:
        """Ingest one PCM chunk. O(chunk); never triggers inference, never blocks."""
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        self._ring.push(chunk)
        self.total_samples += int(chunk.size)
        self.n_pushed_chunks += 1
        self.last_arrival = float(self._clock())
        self._event.set()

    def stop(self) -> None:
        """Ask the worker loop to exit; an in-flight score may finish and is discarded."""
        self._stop = True
        self._event.set()

    @property
    def total_sec(self) -> float:
        return self.total_samples / float(self.sr)

    def stats(self) -> Dict[str, Any]:
        return {
            "inflight_max": self.inflight_max,
            "n_scored": self.n_scored,
            "n_pushed_chunks": self.n_pushed_chunks,
            "max_pending_windows": self.max_pending_windows,
            "dropped_windows": self.dropped_windows,
            "total_sec": round(self.total_sec, 3),
        }

    # -- worker loop ----------------------------------------------------------

    async def run(self, emit: EmitFn) -> None:
        """Single worker loop: waits for pushes, scores at most one window at a time."""
        while not self._stop:
            await self._event.wait()
            self._event.clear()
            if self._stop:
                break
            await self._dispatch(emit)

    def _filled_sec(self, total: int) -> float:
        return min(total, self._window_samples) / float(self.sr)

    async def _dispatch(self, emit: EmitFn) -> None:
        total = self.total_samples
        filled_sec = self._filled_sec(total)

        if filled_sec < self.min_window_sec:
            # warming_up at most once per stride of incoming audio
            stride_idx = total // self._stride_samples
            if stride_idx != self._last_warm_stride_idx:
                self._last_warm_stride_idx = stride_idx
                await emit(
                    {
                        "type": "prediction",
                        "warming_up": True,
                        "is_speech": rms_vad(self._ring.get_window()),
                        "spoof_prob": None,
                        "baseline_prob": None,
                        "window_verdict": None,
                        "verdict": None,
                        "smoothing": f"{self.smooth_k}_of_{self.smooth_n}",
                        "recent_scores": [],
                        "threshold": self.threshold,
                        "audio_start_sec": 0.0,
                        "audio_end_sec": round(total / self.sr, 3),
                        "stream_sec_received": round(self.total_sec, 3),
                        "inference_ms": 0.0,
                        "lag_ms": 0.0,
                        "dropped_windows": self.dropped_windows,
                        "attention_focus": [0.0] * 32,
                        "latency_ms": 0.0,
                        "backend": None,
                        "window_sec": round(filled_sec, 2),
                    }
                )
            return

        new_samples = total - self._last_consumed_samples
        if self._ever_consumed and new_samples < self._stride_samples:
            return  # keep waiting for the next stride of audio

        # Consume the NEWEST window; intermediate strides are dropped, not queued.
        if self._ever_consumed:
            self.dropped_windows += max(0, new_samples // self._stride_samples - 1)
        self._ever_consumed = True
        self._last_consumed_samples = total

        window = self._ring.get_window()
        audio_end_sec = total / float(self.sr)
        audio_start_sec = (total - self._window_samples) / float(self.sr)
        arrival = self.last_arrival if self.last_arrival is not None else self._clock()
        window_sec = round(filled_sec, 2)

        self._pending_windows += 1
        self.max_pending_windows = max(self.max_pending_windows, self._pending_windows)
        try:
            base = {
                "type": "prediction",
                "threshold": self.threshold,
                "smoothing": f"{self.smooth_k}_of_{self.smooth_n}",
                "recent_scores": [round(float(p), 4) for p in list(self._history)[-3:]],
                "audio_start_sec": round(audio_start_sec, 3),
                "audio_end_sec": round(audio_end_sec, 3),
                "stream_sec_received": round(self.total_sec, 3),
                "dropped_windows": self.dropped_windows,
                "window_sec": window_sec,
            }
            if not rms_vad(window):
                await emit(
                    {
                        **base,
                        "is_speech": False,
                        "spoof_prob": None,
                        "baseline_prob": None,
                        "window_verdict": None,
                        "verdict": None,
                        "inference_ms": 0.0,
                        "lag_ms": round((self._clock() - arrival) * 1000.0, 1),
                        "attention_focus": [0.0] * 32,
                        "latency_ms": 0.0,
                        "backend": None,
                    }
                )
                return

            self._inflight += 1
            self.inflight_max = max(self.inflight_max, self._inflight)
            t0 = time.perf_counter()
            try:
                res = await asyncio.to_thread(self.score_fn, window)
            except Exception as exc:  # drop this window; stop after repeated failures
                self.consecutive_failures += 1
                print(
                    f"[live] score_fn failed ({exc}); skipping window "
                    f"({self.consecutive_failures}/{self.max_consecutive_failures} consecutive)",
                    flush=True,
                )
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self._stop = True
                    await emit({"type": "error", "detail": f"detector failed {self.consecutive_failures} times in a row: {exc}"})
                return
            finally:
                self._inflight -= 1
            inference_ms = (time.perf_counter() - t0) * 1000.0
            self.consecutive_failures = 0
            self.n_scored += 1
            if self._stop:
                return  # in-flight result discarded after stop

            prob = float(res.get("spoof_prob", 0.5))
            self._history.append(prob)
            window_verdict = "spoof" if prob >= self.threshold else "bonafide"
            hits = sum(1 for p in self._history if p >= self.threshold)
            verdict = "spoof" if hits >= self.smooth_k else "bonafide"
            lag_ms = (self._clock() - arrival) * 1000.0
            await emit(
                {
                    **base,
                    "is_speech": True,
                    "spoof_prob": prob,
                    "baseline_prob": res.get("baseline_prob"),
                    "window_verdict": window_verdict,
                    "verdict": verdict,
                    "recent_scores": [round(float(p), 4) for p in list(self._history)[-3:]],
                    "inference_ms": round(inference_ms, 1),
                    "lag_ms": round(lag_ms, 1),
                    "attention_focus": res.get("attention_focus") or [0.0] * 32,
                    "latency_ms": res.get("latency_ms", 0.0),
                    "backend": res.get("backend"),
                }
            )
        finally:
            self._pending_windows -= 1
