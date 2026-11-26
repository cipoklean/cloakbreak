"""FIFO single-worker job queue + media decode/download helpers for the analysis API.

One worker thread processes analysis jobs in submission order so the CPU-heavy
scoring path never runs concurrently with itself. Finished jobs are kept for
10 minutes so clients can poll ``GET /api/jobs/{id}``; unfinished (queued +
running) jobs are capped so a flood of submissions gets a fast 429 instead of
an unbounded backlog.
"""

from __future__ import annotations

import ipaddress
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, Optional
from urllib.parse import urlparse

import numpy as np

from src.audio_utils import TARGET_SR, load_wav_mono

MAX_JOB_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_UPLOAD_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg"}
DECODE_ERROR_MESSAGE = "Could not decode this file. Try a WAV or MP3."
BLOCKED_ERROR_MESSAGE = "This site blocked our server. Download the video and upload the file instead."
TIMEOUT_ERROR_MESSAGE = "The download timed out. Download the video and upload the file instead."
SSRF_ERROR_MESSAGE = "That address is not allowed."
FFMPEG_DECODE_TIMEOUT_SEC = 60.0
YTDLP_TIMEOUT_SEC = 60.0


class QueueFullError(Exception):
    """Raised when the number of unfinished jobs is at capacity."""


class ClipInputError(Exception):
    """User-facing input failure (undecodable media, blocked download, ...)."""


def decode_to_mono16k(inp_path: Path, out_wav: Path, timeout: float = FFMPEG_DECODE_TIMEOUT_SEC) -> np.ndarray:
    """Decode any ffmpeg-supported media file to 16 kHz mono float32.

    Raises ``ClipInputError(DECODE_ERROR_MESSAGE)`` on any failure.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(inp_path),
        "-ac", "1", "-ar", str(TARGET_SR),
        "-f", "wav", str(out_wav),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise ClipInputError(DECODE_ERROR_MESSAGE) from exc
    if proc.returncode != 0 or not out_wav.exists():
        raise ClipInputError(DECODE_ERROR_MESSAGE)
    try:
        wav, _sr = load_wav_mono(out_wav, target_sr=TARGET_SR)
    except Exception as exc:
        raise ClipInputError(DECODE_ERROR_MESSAGE) from exc
    return wav


def download_video_audio(url: str, tmpdir: Path, timeout: float = YTDLP_TIMEOUT_SEC) -> Path:
    """Download best-audio media from ``url`` into ``tmpdir`` via yt-dlp.

    Returns the downloaded file path. Nonzero exit -> the site blocked us;
    timeout -> its own message. Both raise ``ClipInputError``.
    """
    out_template = str(Path(tmpdir) / "out.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--no-warnings",
        "-f", "ba/b",
        "--max-filesize", "50M",
        "--socket-timeout", "15",
        "--retries", "1",
        "-o", out_template,
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ClipInputError(TIMEOUT_ERROR_MESSAGE) from exc
    except OSError as exc:
        raise ClipInputError(BLOCKED_ERROR_MESSAGE) from exc
    if proc.returncode != 0:
        raise ClipInputError(BLOCKED_ERROR_MESSAGE)
    outs = sorted(p for p in Path(tmpdir).glob("out.*") if p.is_file())
    if not outs:
        raise ClipInputError(BLOCKED_ERROR_MESSAGE)
    return outs[0]


def url_validation_error(url: str, resolver: Optional[Callable] = None) -> Optional[str]:
    """SSRF guard for analyze-url. Returns the error detail or None if allowed.

    Only http/https, a real hostname, and no userinfo. Every resolved IP must
    be a public unicast address — private/loopback/link-local/multicast/
    reserved/unspecified are all refused, as is any DNS failure.
    """
    if resolver is None:
        resolver = socket.getaddrinfo
    if not isinstance(url, str) or len(url) > 2048:
        return SSRF_ERROR_MESSAGE
    try:
        parsed = urlparse(url)
    except Exception:
        return SSRF_ERROR_MESSAGE
    if parsed.scheme not in ("http", "https"):
        return SSRF_ERROR_MESSAGE
    if not parsed.hostname:
        return SSRF_ERROR_MESSAGE
    if parsed.username is not None or parsed.password is not None:
        return SSRF_ERROR_MESSAGE
    try:
        port = parsed.port
    except ValueError:
        return SSRF_ERROR_MESSAGE
    try:
        infos = resolver(
            parsed.hostname,
            port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except (socket.gaierror, OSError, UnicodeError):
        return SSRF_ERROR_MESSAGE
    if not infos:
        return SSRF_ERROR_MESSAGE
    for info in infos:
        host = str(info[4][0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return SSRF_ERROR_MESSAGE
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return SSRF_ERROR_MESSAGE
    return None


class _Job:
    __slots__ = (
        "job_id", "kind", "fn", "status", "result", "error", "error_type",
        "created_at", "started_at", "finished_at", "done_event",
    )

    def __init__(self, job_id: str, kind: str, fn: Callable[[], dict], now: float) -> None:
        self.job_id = job_id
        self.kind = kind
        self.fn = fn
        self.status = "queued"
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.error_type: Optional[str] = None
        self.created_at = now
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.done_event = threading.Event()


class JobQueue:
    """Thread-safe single-worker FIFO queue.

    ``clock`` is injectable and ``expire_sec`` configurable so the expiry logic
    is unit-testable without real waiting.
    """

    def __init__(
        self,
        max_unfinished: int = 10,
        expire_sec: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_unfinished = int(max_unfinished)
        self.expire_sec = float(expire_sec)
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: Dict[str, _Job] = {}
        self._fifo: Deque[str] = deque()
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._worker_main, name="cloakbreak-job-worker", daemon=True
        )
        self._thread.start()

    # -- producer side --------------------------------------------------------

    def submit(self, fn: Callable[[], dict], kind: str = "job") -> str:
        with self._lock:
            self._purge_expired_locked()
            n_unfinished = sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))
            if n_unfinished >= self.max_unfinished:
                raise QueueFullError("too many jobs in flight")
            job_id = uuid.uuid4().hex[:16]
            self._jobs[job_id] = _Job(job_id, kind, fn, self._clock())
            self._fifo.append(job_id)
        self._wake.set()
        return job_id

    # -- consumer/worker side -------------------------------------------------

    def _worker_main(self) -> None:
        while True:
            with self._lock:
                job: Optional[_Job] = None
                while self._fifo:
                    candidate = self._jobs.get(self._fifo[0])
                    if candidate is None:
                        self._fifo.popleft()  # expired before it ever ran
                        continue
                    self._fifo.popleft()
                    candidate.status = "running"
                    candidate.started_at = self._clock()
                    job = candidate
                    break
                if job is None:
                    self._purge_expired_locked()
            if job is None:
                self._wake.wait(timeout=5.0)
                self._wake.clear()
                continue
            try:
                result = job.fn()
            except Exception as exc:  # job failures are data, not worker crashes
                with self._lock:
                    job.status = "error"
                    job.error = str(exc) or exc.__class__.__name__
                    job.error_type = exc.__class__.__name__
                    job.finished_at = self._clock()
            else:
                with self._lock:
                    job.status = "done"
                    job.result = result
                    job.finished_at = self._clock()
            job.done_event.set()

    # -- status/expiry ---------------------------------------------------------

    def _purge_expired_locked(self) -> None:
        now = self._clock()
        expired = [
            jid
            for jid, j in self._jobs.items()
            if j.status in ("done", "error")
            and j.finished_at is not None
            and now - j.finished_at >= self.expire_sec
        ]
        for jid in expired:
            del self._jobs[jid]

    def get_job(self, job_id: str) -> Optional[_Job]:
        with self._lock:
            self._purge_expired_locked()
            return self._jobs.get(job_id)

    def position_locked(self, job_id: str) -> Optional[int]:
        """1-based FIFO position among queued jobs (1 = next in line)."""
        for idx, jid in enumerate(self._fifo):
            if jid == job_id:
                return idx + 1
        return None

    def status_view(self, job_id: str) -> Optional[dict]:
        with self._lock:
            self._purge_expired_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return None
            view = {"job_id": job.job_id, "status": job.status, "kind": job.kind}
            if job.status == "queued":
                pos = self.position_locked(job_id)
                if pos is not None:
                    view["position"] = pos
            if job.status == "done":
                view["result"] = job.result
            if job.status == "error":
                view["error"] = job.error
                view["error_type"] = job.error_type
            return view

    def wait(self, job_id: str, timeout: Optional[float] = None) -> bool:
        job = self.get_job(job_id)
        if job is None:
            return False
        return job.done_event.wait(timeout=timeout)
