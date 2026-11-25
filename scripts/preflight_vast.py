from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from src.audio_utils import TARGET_SR, save_wav_mono

REQUIRED_IMPORTS = [
    "pkg_resources",
    "numpy",
    "scipy",
    "soundfile",
    "librosa",
    "pyarrow",
    "pandas",
    "datasets",
    "torch",
    "torchaudio",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "diffusers",
    "perth",
    "s3tokenizer",
    "conformer",
    "chatterbox",
    "chatterbox.tts_turbo",
    "kokoro",
    "misaki",
    "spacy",
    "onnx",
    "onnxruntime",
    "sklearn",
    "gradio",
    "fastapi",
    "uvicorn",
    "websockets",
    "tqdm",
    "src.audio_utils",
    "src.augment",
    "src.model",
    "src.generate_spoofs",
    "src.train",
    "src.eval",
    "app.main",
]


def _step(name: str) -> None:
    print(f"\n[preflight] {name}...", flush=True)


def run_hidden_backend_checks() -> None:
    """Detect libraries that silently degrade to None behind try/except ImportError."""
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"  OK  {label}", flush=True)
        else:
            msg = f"{label}: {detail or 'failed / None'}"
            print(f"  ERR {msg}", flush=True)
            failures.append(msg)

    import chatterbox.tts_turbo
    import datasets
    import diffusers.utils
    import kokoro
    import onnxruntime
    import onnxruntime.quantization
    import perth
    import transformers
    import transformers.utils

    check(
        "perth.PerthImplicitWatermarker callable",
        perth.PerthImplicitWatermarker is not None and callable(perth.PerthImplicitWatermarker),
        "None (likely missing pkg_resources from setuptools>=81; pin setuptools==80.9.0)",
    )
    check(
        "chatterbox.tts_turbo.ChatterboxTurboTTS",
        getattr(chatterbox.tts_turbo, "ChatterboxTurboTTS", None) is not None,
    )
    check("kokoro.KPipeline", getattr(kokoro, "KPipeline", None) is not None)
    check("transformers torch backend", bool(transformers.utils.is_torch_available()))
    check("transformers safetensors backend", bool(transformers.utils.is_safetensors_available()))
    for sym in ("AutoModel", "AutoModelForAudioClassification", "AutoFeatureExtractor", "WavLMModel", "Wav2Vec2ForSequenceClassification"):
        check(f"transformers.{sym}", getattr(transformers, sym, None) is not None)
    check("diffusers torch backend", bool(diffusers.utils.is_torch_available()))
    check("diffusers safetensors backend", bool(diffusers.utils.is_safetensors_available()))
    check("datasets.Audio & load_dataset", datasets.Audio is not None and datasets.load_dataset is not None)
    check(
        "onnxruntime InferenceSession & quantize_dynamic",
        onnxruntime.InferenceSession is not None and onnxruntime.quantization.quantize_dynamic is not None,
    )

    if failures:
        sys.exit("PREFLIGHT FAIL (silent optional-backend check):\n  " + "\n  ".join(failures))


def run_import_sweep() -> None:
    failures: list[str] = []
    for mod_name in REQUIRED_IMPORTS:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", None)
            ver_str = f"=={ver}" if ver else ""
            print(f"  OK  {mod_name}{ver_str}", flush=True)
        except Exception as exc:
            print(f"  ERR {mod_name}: {exc}", flush=True)
            failures.append(f"{mod_name}: {exc}")
    if failures:
        sys.exit("PREFLIGHT FAIL (import sweep):\n  " + "\n  ".join(failures))
    run_hidden_backend_checks()


def main() -> None:
    t_start = time.perf_counter()

    print("\n[preflight] step 0 — HF token & dataset-namespace check (scripts.hf_auth_check)...", flush=True)
    from scripts import hf_auth_check

    hf_auth_check.main()

    _step("1/7 System binaries (nvidia-smi, ffmpeg libopus, espeak-ng)")
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

    _step("2/7 Full import sweep (Sessions 1-4 modules + NumPy/PyArrow ABI check)")
    run_import_sweep()

    _step("3/7 PyTorch & CUDA availability")
    import torch

    print(f"  torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_version={torch.version.cuda}")
    if not torch.cuda.is_available():
        sys.exit("PREFLIGHT FAIL: torch.cuda.is_available() is False.")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"  VRAM: {free_b / 1e9:.1f} GB free / {total_b / 1e9:.1f} GB total")

    _step("4/7 Synthetic 6.0s reference clip (Chatterbox-Turbo requires > 5.0s)")
    tmpdir = Path(tempfile.mkdtemp(prefix="cloakbreak_preflight_"))
    ref_wav = tmpdir / "ref_6s.wav"
    t = np.linspace(0.0, 6.0, TARGET_SR * 6, endpoint=False, dtype=np.float32)
    wave = (0.25 * np.sin(2.0 * np.pi * 210.0 * t)).astype(np.float32)
    save_wav_mono(ref_wav, wave, TARGET_SR)

    _step("5/7 Load Chatterbox-Turbo on CUDA & synthesize 1 test clip")
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        from src.generate_spoofs import _NoOpWatermarker

        cb = ChatterboxTurboTTS.from_pretrained(device="cuda")
        cb.watermarker = _NoOpWatermarker()
        out_cb = cb.generate("Preflight voice check on Vast.", audio_prompt_path=str(ref_wav))
        arr_cb = out_cb.squeeze().detach().cpu().numpy()
        if arr_cb.size < 1000 or not np.isfinite(arr_cb).all():
            sys.exit("PREFLIGHT FAIL: Chatterbox produced empty or non-finite audio.")
        print(f"  Chatterbox-Turbo OK: {arr_cb.shape} @ {cb.sr} Hz")
        del cb
        torch.cuda.empty_cache()
    except Exception as exc:
        sys.exit(f"PREFLIGHT FAIL (Chatterbox): {exc}")

    _step("6/7 Load Kokoro-82M & synthesize 1 test clip")
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

    _step("7/7 Load microsoft/wavlm-base-plus CloakBreakDetector & run 1 CUDA forward pass")
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
    print(f"\nPREFLIGHT PASSED in {elapsed:.1f}s — ready for GPU work.", flush=True)


if __name__ == "__main__":
    main()
