from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from src.audio_utils import (
    TARGET_SR,
    load_wav_bytes_mono,
    pcm16_bytes_to_float32,
    resample_audio,
    rms_vad,
    save_wav_mono,
    wav_to_bytes,
)
from src.augment import EvalCondition, degrade_for_eval
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


def create_app(state: AppState | None = None) -> FastAPI:
    app = FastAPI(title="CloakBreak", version="0.1.0")
    app.state.ctx = state or AppState()

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
        })

    @app.post("/api/detect-file")
    async def detect_file(
        file: UploadFile = File(...),
        condition: EvalCondition = Form("clean"),
    ) -> JSONResponse:
        ctx: AppState = app.state.ctx
        raw = await file.read()
        if len(raw) == 0 or len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="Audio file must be non-empty and <= 5 MB")
        try:
            wav, sr = load_wav_bytes_mono(raw, target_sr=TARGET_SR)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid WAV audio: {exc}") from exc

        if wav.shape[0] / float(TARGET_SR) < ctx.min_window_sec:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"need at least {ctx.min_window_sec:.1f} s of audio "
                    "(clips under 1 s are at chance)"
                ),
            )
        if wav.shape[0] > TARGET_SR * 20:
            wav = wav[: TARGET_SR * 20]
        degraded = degrade_for_eval(wav, condition=condition, sr=sr)
        res = ctx.detector.predict(degraded)
        baseline_prob = ctx.baseline.predict_prob(degraded) if ctx.baseline is not None else None
        return JSONResponse({
            "spoof_prob": res.spoof_prob,
            "baseline_prob": baseline_prob,
            "attention_focus": res.attention_focus,
            "latency_ms": res.latency_ms,
            "backend": res.backend,
            "condition": condition,
            "duration_s": round(degraded.shape[0] / float(TARGET_SR), 2),
            "is_speech": rms_vad(degraded),
            "threshold": ctx.threshold,
            "verdict": "spoof" if res.spoof_prob >= ctx.threshold else "bonafide",
        })

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

    return app


app = create_app()
