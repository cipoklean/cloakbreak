from __future__ import annotations

import base64
import hmac
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from src.audio_utils import (
    TARGET_SR,
    RollingAudioBuffer,
    load_wav_bytes_mono,
    pcm16_bytes_to_float32,
    resample_audio,
    rms_vad,
    save_wav_mono,
    wav_to_bytes,
)
from src.augment import EvalCondition, degrade_for_eval
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


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class AppState:
    def __init__(self) -> None:
        self.backend_mode = os.getenv("CLOAKBREAK_BACKEND", "mock").strip().lower()
        self.model_path = os.getenv("CLOAKBREAK_MODEL_PATH", "").strip()
        self.baseline_id = os.getenv("CLOAKBREAK_BASELINE_ID", "").strip()
        self.device = os.getenv("CLOAKBREAK_DEVICE", "cpu").strip()
        self.clone_enabled = _env_flag("ENABLE_CLONE_ENDPOINT", default=False)
        self.clone_token = os.getenv("CLOAKBREAK_CLONE_TOKEN", "").strip()

        self.detector: DetectorRuntime = self._init_detector()
        self.baseline: Optional[BaselineDetectorRuntime] = self._init_baseline()
        self._chatterbox_model = None

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
        ring = RollingAudioBuffer(window_sec=2.5, stride_sec=0.25, sr=TARGET_SR)
        condition: EvalCondition = "clean"

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                pcm_bytes = message.get("bytes")
                if pcm_bytes is None and message.get("text"):
                    payload = json.loads(message["text"])
                    if "condition" in payload and payload["condition"] in {
                        "clean", "opus_16k", "phone_8k", "noise_10db", "noise_5db"
                    }:
                        condition = payload["condition"]
                    if "pcm16_b64" in payload:
                        pcm_bytes = base64.b64decode(payload["pcm16_b64"])
                    else:
                        await websocket.send_json({"type": "ack", "condition": condition})
                        continue

                if not pcm_bytes:
                    continue

                samples = pcm16_bytes_to_float32(pcm_bytes)
                if not ring.push(samples):
                    continue

                window = ring.get_window()
                speech = rms_vad(window)
                if not speech:
                    await websocket.send_json({
                        "type": "prediction",
                        "is_speech": False,
                        "spoof_prob": None,
                        "baseline_prob": None,
                        "attention_focus": [0.0] * 32,
                        "latency_ms": 0.0,
                        "backend": ctx.detector.backend_name,
                        "condition": condition,
                        "window_sec": round(ring.filled_sec, 2),
                    })
                    continue

                eval_window = degrade_for_eval(window, condition=condition, sr=TARGET_SR)
                res = ctx.detector.predict(eval_window)
                baseline_prob = ctx.baseline.predict_prob(eval_window) if ctx.baseline is not None else None
                await websocket.send_json({
                    "type": "prediction",
                    "is_speech": True,
                    "spoof_prob": res.spoof_prob,
                    "baseline_prob": baseline_prob,
                    "attention_focus": res.attention_focus,
                    "latency_ms": res.latency_ms,
                    "backend": res.backend,
                    "condition": condition,
                    "window_sec": round(ring.filled_sec, 2),
                })
        except WebSocketDisconnect:
            return

    return app


app = create_app()
