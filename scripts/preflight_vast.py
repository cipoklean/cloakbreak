from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from src.audio_utils import TARGET_SR, save_wav_mono


def _step(name: str) -> None:
    print(f"\n[preflight] {name}...", flush=True)


def main() -> None:
    t_start = time.perf_counter()

    _step("1/6 System binaries (nvidia-smi, ffmpeg libopus, espeak-ng)")
    for binary in ("nvidia-smi", "ffmpeg", "espeak-ng"):
        if shutil.which(binary) is None:
            sys.exit(f"PREFLIGHT FAIL: required binary '{binary}' not found on PATH.")
    nvsmi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if nvsmi.returncode != 0:
        sys.exit(f"PREFLIGHT FAIL: nvidia-smi failed: {nvsmi.stderr.strip()}")
    print("  GPU:", nvsmi.stdout.strip())

    codecs = subprocess.run(["ffmpeg", "-codecs"], capture_output=True, text=True)
    if "libopus" not in codecs.stdout:
        sys.exit("PREFLIGHT FAIL: ffmpeg does not have libopus enabled.")
    print("  ffmpeg libopus: OK | espeak-ng: OK")

    _step("2/6 PyTorch & CUDA availability")
    import torch

    print(f"  torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_version={torch.version.cuda}")
    if not torch.cuda.is_available():
        sys.exit("PREFLIGHT FAIL: torch.cuda.is_available() is False.")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"  VRAM: {free_b / 1e9:.1f} GB free / {total_b / 1e9:.1f} GB total")

    _step("3/6 Synthetic reference clip")
    tmpdir = Path(tempfile.mkdtemp(prefix="cloakbreak_preflight_"))
    ref_wav = tmpdir / "ref.wav"
    t = np.linspace(0.0, 2.0, TARGET_SR * 2, endpoint=False, dtype=np.float32)
    wave = (0.25 * np.sin(2.0 * np.pi * 210.0 * t)).astype(np.float32)
    save_wav_mono(ref_wav, wave, TARGET_SR)

    _step("4/6 Load Chatterbox-Turbo on CUDA & synthesize 1 test clip")
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        cb = ChatterboxTurboTTS.from_pretrained(device="cuda")
        out_cb = cb.generate("Preflight voice check on Vast.", audio_prompt_path=str(ref_wav))
        arr_cb = out_cb.squeeze().detach().cpu().numpy()
        if arr_cb.size < 1000 or not np.isfinite(arr_cb).all():
            sys.exit("PREFLIGHT FAIL: Chatterbox produced empty or non-finite audio.")
        print(f"  Chatterbox-Turbo OK: {arr_cb.shape} @ {cb.sr} Hz")
        del cb
        torch.cuda.empty_cache()
    except Exception as exc:
        sys.exit(f"PREFLIGHT FAIL (Chatterbox): {exc}")

    _step("5/6 Load Kokoro-82M & synthesize 1 test clip")
    try:
        from kokoro import KPipeline

        pipe = KPipeline(lang_code="a")
        chunks = [np.asarray(a, dtype=np.float32) for _, _, a in pipe("Preflight Kokoro test.", voice="af_heart")]
        arr_k = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if arr_k.size < 1000 or not np.isfinite(arr_k).all():
            sys.exit("PREFLIGHT FAIL: Kokoro-82M produced empty or non-finite audio.")
        print(f"  Kokoro-82M OK: {arr_k.shape} @ 24000 Hz")
        del pipe
        torch.cuda.empty_cache()
    except Exception as exc:
        sys.exit(f"PREFLIGHT FAIL (Kokoro): {exc}")

    _step("6/6 Load microsoft/wavlm-base-plus CloakBreakDetector & run 1 CUDA forward pass")
    try:
        from src.model import DEFAULT_BACKBONE_ID, DEFAULT_WINDOW_SAMPLES, CloakBreakDetector

        det = CloakBreakDetector(backbone_id=DEFAULT_BACKBONE_ID).to("cuda").eval()
        with torch.inference_mode():
            dummy = torch.zeros(1, DEFAULT_WINDOW_SAMPLES, dtype=torch.float32, device="cuda")
            logit, attn = det(dummy)
        if logit.shape != (1,) or attn.ndim != 2:
            sys.exit(f"PREFLIGHT FAIL: unexpected WavLM output shapes logit={logit.shape} attn={attn.shape}")
        print(f"  WavLM-Base-Plus OK: logit={float(logit.item()):.4f}, attn_frames={attn.shape[1]}")
        del det
        torch.cuda.empty_cache()
    except Exception as exc:
        sys.exit(f"PREFLIGHT FAIL (WavLM): {exc}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    elapsed = time.perf_counter() - t_start
    print(f"\nPREFLIGHT PASSED in {elapsed:.1f}s — ready for GPU Session 1.", flush=True)


if __name__ == "__main__":
    main()
