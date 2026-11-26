from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.analysis import analyze_clip
from src.audio_utils import (
    TARGET_SR,
    load_wav_bytes_mono,
    load_wav_mono,
    pcm16_bytes_to_float32,
    resample_audio,
    save_wav_mono,
    wav_to_bytes,
)
from src.augment import EvalCondition, degrade_for_eval
from src.jobs import (
    ALLOWED_UPLOAD_EXTS,
    DECODE_ERROR_MESSAGE,
    MAX_JOB_UPLOAD_BYTES,
    JobQueue,
    QueueFullError,
    decode_to_mono16k,
    download_video_audio,
    url_validation_error,
)
from src.live import LiveScorer
from src.model import (
    BaselineDetectorRuntime,
    DetectorRuntime,
    MockDetectorRuntime,
    OnnxDetectorRuntime,
    TorchDetectorRuntime,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_HTML = ROOT_DIR / "frontend" / "index.html"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB (~15s stereo WAV max)
CAPTURE_MAX_SECONDS = 30 * 60
CAPTURE_MAX_SAMPLES = CAPTURE_MAX_SECONDS * TARGET_SR

# Allowlisted demo clips for the "Try samples" tab. IDs only — clients never
# send paths; each entry maps to a fixed path under the samples dir.
SAMPLE_CLIPS = {
    "kokoro_00483": {"path": "kokoro_unseen/kokoro_00483.wav", "truth": "clone", "source": "Kokoro-82M (unseen TTS)"},
    "kokoro_00069": {"path": "kokoro_unseen/kokoro_00069.wav", "truth": "clone", "source": "Kokoro-82M (unseen TTS)"},
    "kokoro_00454": {"path": "kokoro_unseen/kokoro_00454.wav", "truth": "clone", "source": "Kokoro-82M (unseen TTS)"},
    "ls_8463-294828-0031": {"path": "librispeech_clean/test/8463-294828-0031.wav", "truth": "real", "source": "LibriSpeech test-clean (human)"},
    "ls_8455-210777-0040": {"path": "librispeech_clean/test/8455-210777-0040.wav", "truth": "real", "source": "LibriSpeech test-clean (human)"},
    "ls_672-122797-0057": {"path": "librispeech_clean/test/672-122797-0057.wav", "truth": "real", "source": "LibriSpeech test-clean (human)"},
}


def _samples_dir() -> Path:
    raw = os.getenv("CLOAKBREAK_SAMPLES_DIR", "").strip()
    return Path(raw) if raw else ROOT_DIR / "data" / "cloakbreak_v1"


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class AppState:
    def __init__(self) -> None:
        self.backend_mode = os.getenv("CLOAKBREAK_BACKEND", "mock").strip().lower()
        self.model_path = os.getenv("CLOAKBREAK_MODEL_PATH", "").strip()
        self.baseline_id = os.getenv("CLOAKBREAK_BASELINE_ID", "").strip()
        self.device = os.getenv("CLOAKBREAK_DEVICE", "cpu").strip()
        self.clone_enabled = _env_flag("ENABLE_CLONE_ENDPOINT", default=False)
        self.clone_token = os.getenv("CLOAKBREAK_CLONE_TOKEN", "").strip()
        # Sub-2s windows are at chance; never run the detector on them.
        self.min_window_sec = max(2.0, _env_float("CLOAKBREAK_MIN_WINDOW_SEC", 2.0))
        self.threshold, self.threshold_source = self._resolve_threshold()
        # Opt-in debug capture of raw WS audio (contains the user's voice —
        # results/live_capture/ is gitignored and must never be committed).
        self.debug_capture = _env_flag("CLOAKBREAK_DEBUG_CAPTURE", default=False)
        self.capture_dir = Path(
            os.getenv("CLOAKBREAK_CAPTURE_DIR", "").strip()
            or (ROOT_DIR / "results" / "live_capture")
        )

        self.detector: DetectorRuntime = self._init_detector()
        self.baseline: Optional[BaselineDetectorRuntime] = self._init_baseline()
        self._chatterbox_model = None
        # One live mic session at a time: nonzero while a /ws/detect handler
        # is active; guarded by the lock since connect/disconnect interleave.
        self.live_ws_sessions = 0
        self.live_ws_lock = threading.Lock()

    def _resolve_threshold(self) -> tuple[float, str]:
        env_raw = os.getenv("CLOAKBREAK_THRESHOLD", "").strip()
        if env_raw:
            try:
                return float(env_raw), "env"
            except ValueError:
                print(f"[app] CLOAKBREAK_THRESHOLD={env_raw!r} is not a float — ignoring", flush=True)
        tfile_raw = os.getenv("CLOAKBREAK_THRESHOLD_FILE", "").strip()
        tfile = Path(tfile_raw) if tfile_raw else ROOT_DIR / "results" / "threshold_calibration.json"
        if tfile.exists():
            try:
                data = json.loads(tfile.read_text(encoding="utf-8"))
                t = float(data.get("operating_threshold"))
            except Exception as exc:
                print(f"[app] could not read threshold file {tfile}: {exc}", flush=True)
            else:
                file_sha = data.get("model_sha256")
                if file_sha:
                    actual = _sha256_file(Path(self.model_path)) if self.model_path else None
                    if actual != file_sha:
                        print(
                            f"[app] threshold file {tfile} model_sha256 does not match "
                            f"CLOAKBREAK_MODEL_PATH — falling back to 0.5",
                            flush=True,
                        )
                        return 0.5, "default_0.5_model_mismatch"
                return t, "file"
        return 0.5, "default_0.5"

    def _init_detector(self) -> DetectorRuntime:
        if self.backend_mode == "onnx" and self.model_path:
            return OnnxDetectorRuntime(self.model_path)
        if self.backend_mode == "torch" and self.model_path:
            return TorchDetectorRuntime(self.model_path, device=self.device)
        return MockDetectorRuntime()

    def _init_baseline(self) -> Optional[BaselineDetectorRuntime]:
        if not self.baseline_id:
            return None
        return BaselineDetectorRuntime(self.baseline_id, device=self.device)

    def get_cloner(self):
        if self._chatterbox_model is None:
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self._chatterbox_model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        return self._chatterbox_model


def _write_live_capture(capture_dir: Path, cap: dict) -> Path:
    """Persist one captured WS session: 16 kHz mono PCM_16 WAV + JSON sidecar."""
    capture_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}_{cap['id']}"
    wav_path = capture_dir / f"{stem}.wav"
    chunks = cap["chunks"]
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    save_wav_mono(wav_path, audio, TARGET_SR)
    meta = {
        "file": wav_path.name,
        "sample_rate": TARGET_SR,
        "channels": 1,
        "duration_s": round(cap["total"] / float(TARGET_SR), 3),
        "truncated": bool(cap["truncated"]),
        "n_chunks": len(chunks),
        "windows": cap["verdicts"],
    }
    (capture_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return wav_path


ANALYSIS_CONDITIONS = {
    "clean", "opus_16k", "phone_8k", "noise_10db", "noise_5db", "short_0.5s", "short_0.8s",
}
QUEUE_FULL_DETAIL = "Too many analyses in flight. Try again in a moment."


def _validate_condition(value: object) -> EvalCondition:
    cond = value if isinstance(value, str) and value.strip() else "clean"
    cond = cond.strip()
    if cond not in ANALYSIS_CONDITIONS:
        raise HTTPException(status_code=422, detail=f"unknown condition {cond!r}")
    return cond  # type: ignore[return-value]


async def _read_payload(request: Request) -> dict:
    """Leniently read a small JSON-or-form POST payload as a plain dict."""
    ctype = (request.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        try:
            body = await request.json()
        except Exception:
            return {}
        return body if isinstance(body, dict) else {}
    try:
        form = await request.form()
    except Exception:
        return {}
    return {k: v for k, v in form.items()}


async def _stage_upload(file: UploadFile, enforce_ext: bool = True) -> tuple[tempfile.TemporaryDirectory, Path]:
    """Stream an upload into a fresh TemporaryDirectory (caller transfers ownership
    to the job, or cleans up itself on failure). Enforces extension + size caps.

    The sync /api/detect-file compat endpoint passes enforce_ext=False: its
    legacy contract never inspected the filename, only the bytes.
    """
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if enforce_ext and ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type {ext or '(no extension)'!r}. "
                "Use one of: .wav, .mp3, .m4a, .mp4, .webm, .ogg"
            ),
        )
    tmpdir = tempfile.TemporaryDirectory()
    inp = Path(tmpdir.name) / f"input{ext or '.bin'}"
    size = 0
    try:
        with open(inp, "wb") as fh:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_JOB_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
                fh.write(chunk)
    except Exception:
        tmpdir.cleanup()
        raise
    if size == 0:
        tmpdir.cleanup()
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    return tmpdir, inp


def _analyze_with_ctx(ctx: AppState, wav: np.ndarray, condition: EvalCondition, **meta) -> dict:
    return analyze_clip(
        wav,
        condition,
        detector=ctx.detector,
        baseline=ctx.baseline,
        threshold=ctx.threshold,
        min_window_sec=ctx.min_window_sec,
        **meta,
    )


def _make_upload_job(ctx: AppState, tmpdir: tempfile.TemporaryDirectory, inp: Path, condition: EvalCondition):
    """Decode-then-analyze closure for multipart uploads; owns the temp dir."""

    def run() -> dict:
        try:
            wav = decode_to_mono16k(inp, Path(tmpdir.name) / "decoded.wav")
            return _analyze_with_ctx(ctx, wav, condition, input_type="upload")
        finally:
            tmpdir.cleanup()

    return run


def _make_sample_job(ctx: AppState, sample_id: str, meta: dict, condition: EvalCondition):
    path = _samples_dir() / meta["path"]

    def run() -> dict:
        wav, _sr = load_wav_mono(path, target_sr=TARGET_SR)
        return _analyze_with_ctx(
            ctx, wav, condition, input_type="sample", sample_id=sample_id, truth=meta["truth"]
        )

    return run


def _make_url_job(ctx: AppState, url: str, condition: EvalCondition):
    """yt-dlp download-then-analyze closure; the temp dir lives and dies in the worker."""

    def run() -> dict:
        tmpdir = tempfile.TemporaryDirectory()
        try:
            media = download_video_audio(url, Path(tmpdir.name))
            wav = decode_to_mono16k(media, Path(tmpdir.name) / "decoded.wav")
            return _analyze_with_ctx(ctx, wav, condition, input_type="url", url=url)
        finally:
            tmpdir.cleanup()

    return run


def _submit_job(queue: JobQueue, fn, kind: str) -> str:
    try:
        return queue.submit(fn, kind=kind)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=QUEUE_FULL_DETAIL) from exc


async def _wait_for_job(queue: JobQueue, job_id: str) -> None:
    await asyncio.to_thread(queue.wait, job_id)


def _job_result_or_raise(queue: JobQueue, job_id: str) -> dict:
    """Translate a finished job into a sync-endpoint response or HTTP error."""
    view = queue.status_view(job_id)
    if view is None:
        raise HTTPException(status_code=500, detail="analysis job vanished")
    if view["status"] == "error":
        err = view.get("error") or "analysis failed"
        if view.get("error_type") in {"ClipTooShortError", "ClipInputError"}:
            raise HTTPException(status_code=400, detail=err)
        raise HTTPException(status_code=500, detail=err)
    return view["result"]


def _sample_or_404(sample_id: str) -> dict:
    meta = SAMPLE_CLIPS.get(sample_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="unknown sample")
    path = _samples_dir() / meta["path"]
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="sample not available; restore data/cloakbreak_v1",
        )
    return meta


def _load_facts() -> dict:
    """Headline numbers pulled from the results files at startup.

    Every key is omitted individually when its file or json path is missing —
    the endpoint never hardcodes numbers.
    """
    out: dict = {}
    eval_path = ROOT_DIR / "results" / "session2" / "eval_results.json"
    if eval_path.is_file():
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, dict):
            def eer(slice_name: str, model: str = "cloakbreak"):
                slice_ = data.get(slice_name)
                if not isinstance(slice_, dict):
                    return None
                block = slice_.get(model)
                if not isinstance(block, dict):
                    return None
                val = block.get("eer_pct")
                return float(val) if isinstance(val, (int, float)) else None

            headline_cb = eer("clean_real_vs_kokoro_opus_16k", "cloakbreak")
            if headline_cb is not None:
                out["headline_cb_eer"] = headline_cb
            headline_bl = eer("clean_real_vs_kokoro_opus_16k", "baseline")
            if headline_bl is not None:
                out["headline_baseline_eer"] = headline_bl
            noise_10db = eer("test_noise_10db")
            if noise_10db is not None:
                out["noise_10db_eer"] = noise_10db
            short05 = eer("test_short_0.5s")
            if short05 is not None:
                out["short_eer_0_5"] = short05
            short08 = eer("test_short_0.8s")
            if short08 is not None:
                out["short_eer_0_8"] = short08
    grid_path = ROOT_DIR / "results" / "live_mic_headset_ab17x_grid.json"
    if grid_path.is_file():
        try:
            grid = json.loads(grid_path.read_text(encoding="utf-8"))
        except Exception:
            grid = None
        summary = grid.get("summary") if isinstance(grid, dict) else None
        if isinstance(summary, dict):
            frac_all = summary.get("frac_ge_threshold")
            if isinstance(frac_all, (int, float)):
                out["livemic_frac_all"] = float(frac_all)
            frac_gated = summary.get("gate_frac_ge_threshold")
            if isinstance(frac_gated, (int, float)):
                out["livemic_frac_gated"] = float(frac_gated)
    return out


def _honest_limits_markdown() -> Optional[str]:
    """Raw README.md '## Honest Limits' section up to the next '## ' heading."""
    readme = ROOT_DIR / "README.md"
    if not readme.is_file():
        return None
    try:
        lines = readme.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## Honest Limits"):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    section = "\n".join(lines[start:end]).strip()
    return section or None


def _web_dist_dir() -> Path:
    raw = os.getenv("CLOAKBREAK_WEB_DIST", "").strip()
    return Path(raw) if raw else ROOT_DIR / "web" / "dist"


def create_app(state: AppState | None = None) -> FastAPI:
    app = FastAPI(title="CloakBreak", version="0.1.0")
    app.state.ctx = state or AppState()
    app.state.job_queue = JobQueue()
    app.state.facts = _load_facts()

    cors_origin = os.getenv("CLOAKBREAK_CORS_ORIGIN", "").strip()
    if cors_origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[cors_origin],
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    web_dist = _web_dist_dir()
    serve_static_spa = web_dist.is_dir()

    if not serve_static_spa:
        # Legacy single-file frontend, exactly as before the React rebuild:
        # keeps serving until web/dist exists (phase B ships the real UI).
        @app.get("/")
        def index() -> FileResponse:
            if not FRONTEND_HTML.exists():
                raise HTTPException(status_code=404, detail="Frontend not found")
            return FileResponse(FRONTEND_HTML)

    @app.get("/api/health")
    def health() -> JSONResponse:
        ctx: AppState = app.state.ctx
        return JSONResponse({
            "status": "ok",
            "backend": ctx.detector.backend_name,
            "baseline_loaded": ctx.baseline is not None,
            "baseline_id": ctx.baseline.model_id if ctx.baseline else None,
            "clone_enabled": ctx.clone_enabled,
            "threshold": ctx.threshold,
            "threshold_source": ctx.threshold_source,
            "debug_capture": ctx.debug_capture,
            "samples_available": sum(
                1 for meta in SAMPLE_CLIPS.values() if (_samples_dir() / meta["path"]).is_file()
            ),
        })

    @app.post("/api/detect-file")
    async def detect_file(
        file: UploadFile = File(...),
        condition: EvalCondition = Form("clean"),
    ) -> JSONResponse:
        """Synchronous compatibility endpoint: same FIFO worker, same all-window
        result shape as the async upload job — the response just waits for it."""
        ctx: AppState = app.state.ctx
        queue: JobQueue = app.state.job_queue
        cond = _validate_condition(condition)
        tmpdir, inp = await _stage_upload(file, enforce_ext=False)
        try:
            job_id = _submit_job(queue, _make_upload_job(ctx, tmpdir, inp, cond), kind="upload")
        except HTTPException:
            tmpdir.cleanup()
            raise
        await _wait_for_job(queue, job_id)
        return JSONResponse(_job_result_or_raise(queue, job_id))

    @app.post("/api/analyze/upload", status_code=202)
    async def analyze_upload(
        file: UploadFile = File(...),
        condition: str = Form("clean"),
    ) -> JSONResponse:
        ctx: AppState = app.state.ctx
        queue: JobQueue = app.state.job_queue
        cond = _validate_condition(condition)
        tmpdir, inp = await _stage_upload(file)
        try:
            job_id = _submit_job(queue, _make_upload_job(ctx, tmpdir, inp, cond), kind="upload")
        except HTTPException:
            tmpdir.cleanup()
            raise
        return JSONResponse({"job_id": job_id}, status_code=202)

    @app.post("/api/analyze/sample/{sample_id}", status_code=202)
    async def analyze_sample(sample_id: str, request: Request) -> JSONResponse:
        ctx: AppState = app.state.ctx
        queue: JobQueue = app.state.job_queue
        meta = _sample_or_404(sample_id)
        payload = await _read_payload(request)
        cond = _validate_condition(payload.get("condition"))
        job_id = _submit_job(queue, _make_sample_job(ctx, sample_id, meta, cond), kind="sample")
        return JSONResponse({"job_id": job_id}, status_code=202)

    @app.post("/api/analyze-url", status_code=202)
    async def analyze_url(request: Request) -> JSONResponse:
        ctx: AppState = app.state.ctx
        queue: JobQueue = app.state.job_queue
        payload = await _read_payload(request)
        url = payload.get("url")
        if not isinstance(url, str) or not url.strip():
            raise HTTPException(status_code=422, detail="url is required")
        url = url.strip()
        cond = _validate_condition(payload.get("condition"))
        ssrf_err = await asyncio.to_thread(url_validation_error, url)
        if ssrf_err is not None:
            raise HTTPException(status_code=422, detail=ssrf_err)
        job_id = _submit_job(queue, _make_url_job(ctx, url, cond), kind="url")
        return JSONResponse({"job_id": job_id}, status_code=202)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JSONResponse:
        queue: JobQueue = app.state.job_queue
        view = queue.status_view(job_id)
        if view is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return JSONResponse(view)

    @app.get("/api/facts")
    def facts() -> JSONResponse:
        return JSONResponse(app.state.facts)

    @app.get("/api/honest-limits")
    def honest_limits() -> JSONResponse:
        section = _honest_limits_markdown()
        if section is None:
            raise HTTPException(status_code=404, detail="could not parse the Honest Limits section")
        return JSONResponse({"markdown": section})

    @app.get("/api/samples")
    def list_samples() -> JSONResponse:
        out = []
        for sample_id, meta in SAMPLE_CLIPS.items():
            path = _samples_dir() / meta["path"]
            available = path.is_file()
            duration_s = None
            if available:
                try:
                    info = sf.info(str(path))
                    duration_s = round(info.frames / float(info.samplerate), 2)
                except Exception:
                    duration_s = None
            out.append({
                "id": sample_id,
                "truth": meta["truth"],
                "source": meta["source"],
                "available": available,
                "duration_s": duration_s,
            })
        return JSONResponse(out)

    @app.post("/api/samples/{sample_id}/detect")
    async def sample_detect(
        sample_id: str,
        condition: EvalCondition = Form("clean"),
    ) -> JSONResponse:
        """Synchronous compatibility endpoint: submits the sample analysis to the
        same FIFO worker as /api/analyze/sample and waits for the result."""
        ctx: AppState = app.state.ctx
        queue: JobQueue = app.state.job_queue
        meta = _sample_or_404(sample_id)
        cond = _validate_condition(condition)
        job_id = _submit_job(queue, _make_sample_job(ctx, sample_id, meta, cond), kind="sample")
        await _wait_for_job(queue, job_id)
        return JSONResponse(_job_result_or_raise(queue, job_id))

    @app.get("/api/samples/{sample_id}/audio")
    def sample_audio(sample_id: str) -> FileResponse:
        meta = SAMPLE_CLIPS.get(sample_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="unknown sample")
        path = _samples_dir() / meta["path"]
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail="sample not available; restore data/cloakbreak_v1",
            )
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/clone")
    async def private_self_clone(
        text: str = Form(...),
        reference_wav: UploadFile = File(...),
        condition: EvalCondition = Form("clean"),
        authorization: Optional[str] = Header(default=None),
    ) -> Response:
        """Private self-clone endpoint for the builder's own voice demo only.

        Disabled by default (ENABLE_CLONE_ENDPOINT=false). Requires Bearer token
        when enabled on a private Vast GPU session. Never exposed as a public API.
        """
        ctx: AppState = app.state.ctx
        if not ctx.clone_enabled:
            raise HTTPException(status_code=403, detail="Private clone endpoint is disabled")
        if not ctx.clone_token:
            raise HTTPException(status_code=503, detail="Clone token is not configured")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        provided = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(provided, ctx.clone_token):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

        clean_text = text.strip()
        if not clean_text or len(clean_text) > 280:
            raise HTTPException(status_code=400, detail="Text must be 1..280 characters")

        raw = await reference_wav.read()
        if len(raw) == 0 or len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="Reference WAV must be non-empty and <= 5 MB")
        ref_wav, sr = load_wav_bytes_mono(raw, target_sr=TARGET_SR)
        dur = ref_wav.shape[0] / float(sr)
        if dur < 5.5 or dur > 15.0:
            raise HTTPException(status_code=400, detail="Reference clip must be between 5.5s and 15s (Chatterbox requires > 5s)")

        if ctx.detector.backend_name == "mock":
            from src.generate_spoofs import synthesize_mock_clone

            synth = synthesize_mock_clone(ref_wav, seed=len(clean_text))
        else:
            cloner = ctx.get_cloner()
            with tempfile.TemporaryDirectory() as tmpdir:
                ref_path = Path(tmpdir) / "ref.wav"
                save_wav_mono(ref_path, ref_wav, TARGET_SR)
                wav_tensor = cloner.generate(clean_text, audio_prompt_path=str(ref_path))
                synth = resample_audio(
                    wav_tensor.squeeze().detach().cpu().numpy(),
                    int(cloner.sr),
                    TARGET_SR,
                )

        degraded = degrade_for_eval(synth, condition=condition, sr=TARGET_SR)
        return Response(content=wav_to_bytes(degraded, TARGET_SR), media_type="audio/wav")

    @app.websocket("/ws/detect")
    async def ws_detect(websocket: WebSocket) -> None:
        await websocket.accept()
        ctx: AppState = app.state.ctx

        # One live mic session at a time — a second concurrent session would
        # contend with the first for the same CPU-bound detector.
        with ctx.live_ws_lock:
            if ctx.live_ws_sessions >= 1:
                busy = True
            else:
                ctx.live_ws_sessions += 1
                busy = False
        if busy:
            await websocket.send_json({
                "type": "busy",
                "detail": "Someone else is using the mic. Try again in a moment.",
            })
            await websocket.close()
            return

        cond_box = {"condition": "clean"}

        def score_fn(window: np.ndarray) -> dict:
            # Runs in a worker thread via LiveScorer — never on the event loop.
            degraded = degrade_for_eval(window, condition=cond_box["condition"], sr=TARGET_SR)
            res = ctx.detector.predict(degraded)
            out = {
                "spoof_prob": res.spoof_prob,
                "attention_focus": res.attention_focus,
                "latency_ms": res.latency_ms,
                "backend": res.backend,
            }
            if ctx.baseline is not None:
                out["baseline_prob"] = ctx.baseline.predict_prob(degraded)
            return out

        scorer = LiveScorer(
            score_fn,
            ctx.threshold,
            sr=TARGET_SR,
            window_sec=2.5,
            min_window_sec=ctx.min_window_sec,
        )

        cap = None
        if ctx.debug_capture:
            cap = {
                "id": uuid.uuid4().hex[:8],
                "chunks": [],
                "total": 0,
                "truncated": False,
                "verdicts": [],
            }

        async def emit(msg: dict) -> None:
            msg = dict(msg)
            msg["condition"] = cond_box["condition"]
            if msg.get("backend") is None:
                msg["backend"] = ctx.detector.backend_name
            if cap is not None and msg.get("type") == "prediction" and msg.get("is_speech") is True:
                if msg.get("spoof_prob") is not None:
                    cap["verdicts"].append({
                        "audio_start_sec": msg.get("audio_start_sec"),
                        "audio_end_sec": msg.get("audio_end_sec"),
                        "spoof_prob": msg.get("spoof_prob"),
                        "input_dbfs": msg.get("input_dbfs"),
                        "baseline_prob": msg.get("baseline_prob"),
                        "verdict": msg.get("verdict"),
                        "window_verdict": msg.get("window_verdict"),
                        "condition": msg.get("condition"),
                        "threshold": msg.get("threshold"),
                    })
            await websocket.send_json(msg)

        scorer_task = asyncio.create_task(scorer.run(emit))
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                pcm_bytes = message.get("bytes")
                if pcm_bytes is None and message.get("text"):
                    payload = json.loads(message["text"])
                    if payload.get("type") == "stats":
                        await websocket.send_json({"type": "stats", **scorer.stats()})
                        continue
                    if "condition" in payload and payload["condition"] in {
                        "clean", "opus_16k", "phone_8k", "noise_10db", "noise_5db"
                    }:
                        cond_box["condition"] = payload["condition"]
                    if "pcm16_b64" in payload:
                        pcm_bytes = base64.b64decode(payload["pcm16_b64"])
                    else:
                        await websocket.send_json({"type": "ack", "condition": cond_box["condition"]})
                        continue

                if not pcm_bytes:
                    continue

                pcm = pcm16_bytes_to_float32(pcm_bytes)
                if cap is not None:
                    # Capture exactly what the scorer sees — same decoded array,
                    # no resampling or normalization. Cap at 30 min of audio.
                    if cap["total"] >= CAPTURE_MAX_SAMPLES:
                        cap["truncated"] = True
                    else:
                        cap["chunks"].append(pcm)
                        cap["total"] += int(pcm.size)
                scorer.push(pcm)
        except WebSocketDisconnect:
            pass
        finally:
            with ctx.live_ws_lock:
                ctx.live_ws_sessions = max(0, ctx.live_ws_sessions - 1)
            try:
                scorer.stop()
                try:
                    await asyncio.wait_for(asyncio.shield(scorer_task), timeout=5.0)
                except asyncio.TimeoutError:
                    scorer_task.cancel()
                    try:
                        await scorer_task
                    except asyncio.CancelledError:
                        pass
                except asyncio.CancelledError:
                    scorer_task.cancel()
                    raise
                except Exception as exc:
                    print(f"[ws] live scorer task ended with error: {exc}", flush=True)
            finally:
                # Runs even if the handler task is cancelled (e.g. TestClient
                # teardown): the capture must not be lost with the session.
                # Written synchronously — an await here could be cancelled
                # before the thread picks up the work.
                if cap is not None and cap["chunks"]:
                    try:
                        _write_live_capture(ctx.capture_dir, cap)
                    except Exception as exc:
                        print(f"[ws] debug capture write failed: {exc}", flush=True)

    # Catch-all for unknown API paths, registered after every real API route:
    # without it an unmatched POST would fall into the static mount below and
    # surface as a 405 instead of the JSON 404 API clients expect.
    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    def api_not_found(path: str) -> None:
        raise HTTPException(status_code=404, detail="unknown api endpoint")

    if serve_static_spa:
        # Mounted LAST so every explicit /api and /ws route wins over it.
        # html=True serves the React index.html at / (and any directory path).
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")

    return app


app = create_app()
