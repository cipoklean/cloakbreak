from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.audio_utils import (
    TARGET_SR,
    load_wav_bytes_mono,
    load_wav_mono,
    normalize_peak,
    resample_audio,
    save_wav_mono,
)

SAMPLE_PROMPTS = [
    "Hey, can you confirm that wire transfer before the bank closes today?",
    "I lost my phone at the airport and I am calling from a colleague's number.",
    "Please send the verification code as soon as it arrives by text.",
    "We need to update the payroll account details before Friday afternoon.",
    "This is an urgent call regarding an unauthorized charge on your card.",
    "Let me read the new routing number slowly so you can double check it.",
    "I am stuck in traffic and need you to approve the invoice right away.",
    "Could you voice-confirm your full name and date of birth for security?",
    "The quarterly report looks solid, let us review the risk section next.",
    "Good morning everyone, thank you for joining the standup on time.",
    "Remember to rotate the staging credentials after the deployment finishes.",
    "Weather in Lagos is humid today with scattered rain expected tonight.",
]


def assign_speaker_split(speaker_id: str) -> str:
    """Deterministic speaker-disjoint split: ~70% train, 15% val, 15% test."""
    digest = int(hashlib.sha256(speaker_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if digest < 70:
        return "train"
    if digest < 85:
        return "val"
    return "test"


def _wav_duration_s(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return round(info.frames / float(info.samplerate), 3) if info.samplerate else 0.0


def _write_manifest_atomic(manifest_path: Path, records: List[Dict]) -> None:
    tmp_path = manifest_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    tmp_path.replace(manifest_path)


def synthesize_mock_clone(ref_audio: np.ndarray, seed: int) -> np.ndarray:
    """Fast CPU synthetic waveform for unit tests (no model download)."""
    rng = np.random.default_rng(seed)
    n = max(TARGET_SR * 2, min(ref_audio.shape[0], TARGET_SR * 4))
    t = np.linspace(0.0, n / TARGET_SR, n, endpoint=False, dtype=np.float32)
    f0 = float(rng.uniform(140.0, 260.0))
    wave = 0.25 * np.sin(2.0 * np.pi * f0 * t) + 0.12 * np.sin(2.0 * np.pi * (2.0 * f0) * t)
    wave += rng.normal(0.0, 0.015, size=n).astype(np.float32)
    return normalize_peak(wave, 0.85)


def generate_chatterbox_clip(model, text: str, ref_wav_path: Path, turbo: bool = True) -> tuple[np.ndarray, int]:
    if turbo:
        wav_tensor = model.generate(text, audio_prompt_path=str(ref_wav_path))
    else:
        wav_tensor = model.generate(text, audio_prompt_path=str(ref_wav_path), exaggeration=0.5, cfg_weight=0.5)
    arr = wav_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    return arr, int(model.sr)


def generate_kokoro_clip(pipeline, text: str, voice: str = "af_heart") -> tuple[np.ndarray, int]:
    chunks = []
    for _, _, audio in pipeline(text, voice=voice):
        chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        return np.zeros(TARGET_SR, dtype=np.float32), 24000
    return np.concatenate(chunks), 24000


def build_from_local_dir(
    real_dir: Path,
    out_dir: Path,
    max_chatterbox: int = 20,
    max_kokoro: int = 10,
    mock_generator: bool = True,
) -> Path:
    """Resumable local builder: skips WAV files that already exist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(real_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files found in {real_dir}")

    records: List[Dict] = []
    for idx, wav_path in enumerate(wavs):
        speaker_id = wav_path.stem.split("_")[0] or f"spk{idx}"
        split = assign_speaker_split(speaker_id)
        records.append({
            "id": f"real_{idx}",
            "path": str(wav_path.resolve()),
            "label": 0,
            "generator": "bonafide",
            "speaker_id": speaker_id,
            "split": split,
            "duration_s": _wav_duration_s(wav_path),
        })

    cb_dir = out_dir / "chatterbox"
    cb_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(max_chatterbox, len(wavs))):
        ref_path = wavs[i]
        speaker_id = ref_path.stem.split("_")[0] or f"spk{i}"
        split = assign_speaker_split(speaker_id)
        out_path = cb_dir / f"cb_{i:05d}.wav"
        if not out_path.exists():
            if not mock_generator:
                raise RuntimeError("Use --source hf for GPU Chatterbox generation")
            ref_audio, _ = load_wav_mono(ref_path, target_sr=TARGET_SR)
            synth = synthesize_mock_clone(ref_audio, seed=1000 + i)
            save_wav_mono(out_path, synth, TARGET_SR)
        records.append({
            "id": f"cb_{i}",
            "path": str(out_path.resolve()),
            "label": 1,
            "generator": "chatterbox_mock",
            "speaker_id": speaker_id,
            "split": split,
            "duration_s": _wav_duration_s(out_path),
        })

    kokoro_dir = out_dir / "kokoro_unseen"
    kokoro_dir.mkdir(parents=True, exist_ok=True)
    for i in range(max_kokoro):
        out_path = kokoro_dir / f"kokoro_{i:05d}.wav"
        if not out_path.exists():
            synth = synthesize_mock_clone(np.zeros(TARGET_SR * 2, dtype=np.float32), seed=5000 + i)
            save_wav_mono(out_path, synth, TARGET_SR)
        records.append({
            "id": f"kokoro_{i}",
            "path": str(out_path.resolve()),
            "label": 1,
            "generator": "kokoro_unseen_mock",
            "speaker_id": f"kokoro_voice_{i % 4}",
            "split": "test",
            "duration_s": _wav_duration_s(out_path),
        })

    manifest_path = out_dir / "manifest.jsonl"
    _write_manifest_atomic(manifest_path, records)
    return manifest_path


def build_from_hf_on_gpu(
    out_dir: Path,
    hf_dataset: str = "SpeechAntiSpoofingBenchmarks/InTheWild",
    max_real: int = 12000,
    max_chatterbox: int = 3000,
    max_kokoro: int = 600,
    use_turbo: bool = True,
) -> Path:
    """Resumable GPU dataset builder.

    Decodes raw audio bytes via soundfile (Audio(decode=False)) to avoid
    torchcodec dependencies in datasets>=4.0, and skips already-generated WAVs
    so interrupted runs never redo paid GPU work.
    """
    from datasets import Audio, load_dataset
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    real_dir = out_dir / "inthewild_wav"
    cb_dir = out_dir / "chatterbox"
    kokoro_dir = out_dir / "kokoro_unseen"
    for d in (real_dir, cb_dir, kokoro_dir):
        d.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(hf_dataset, split="train").cast_column("audio", Audio(decode=False))
    records: List[Dict] = []
    bonafide_refs: List[Dict] = []

    skipped_real = 0
    print(f"Exporting up to {max_real} InTheWild clips to {real_dir} (resumable)...", flush=True)
    for idx, row in enumerate(tqdm(ds, total=min(len(ds), max_real))):
        if idx >= max_real:
            break
        notes = json.loads(row["notes"]) if isinstance(row.get("notes"), str) else {}
        speaker_id = str(notes.get("speaker") or f"itw_spk_{idx}")
        label = int(row["label"])
        wav_path = real_dir / f"itw_{idx:06d}.wav"

        if wav_path.exists() and wav_path.stat().st_size > 100:
            dur = _wav_duration_s(wav_path)
            skipped_real += 1
        else:
            audio_info = row["audio"]
            raw_bytes = audio_info.get("bytes")
            if raw_bytes is not None:
                audio_16k, _ = load_wav_bytes_mono(raw_bytes, target_sr=TARGET_SR)
            elif audio_info.get("path"):
                audio_16k, _ = load_wav_mono(audio_info["path"], target_sr=TARGET_SR)
            else:
                continue
            dur = audio_16k.shape[0] / float(TARGET_SR)
            if dur < 1.0 or dur > 15.0:
                continue
            save_wav_mono(wav_path, audio_16k, TARGET_SR)

        if dur < 1.0 or dur > 15.0:
            continue
        split = assign_speaker_split(speaker_id)
        rec = {
            "id": f"itw_{idx}",
            "path": str(wav_path.resolve()),
            "label": label,
            "generator": "bonafide" if label == 0 else "inthewild_spoof",
            "speaker_id": speaker_id,
            "split": split,
            "duration_s": round(dur, 3),
        }
        records.append(rec)
        if label == 0 and 3.0 <= dur <= 10.0:
            bonafide_refs.append(rec)

    print(f"InTheWild ready: {len(records)} clips ({skipped_real} reused from disk).", flush=True)

    if max_chatterbox > 0 and bonafide_refs:
        missing_cb = [i for i in range(max_chatterbox) if not (cb_dir / f"cb_{i:05d}.wav").exists()]
        cb_model = None
        if missing_cb:
            if use_turbo:
                from chatterbox.tts_turbo import ChatterboxTurboTTS as CBClass
            else:
                from chatterbox.tts import ChatterboxTTS as CBClass
            cb_model = CBClass.from_pretrained(device="cuda")
            print(
                f"Generating {len(missing_cb)}/{max_chatterbox} Chatterbox clones "
                f"({max_chatterbox - len(missing_cb)} already on disk)...",
                flush=True,
            )
        else:
            print(f"All {max_chatterbox} Chatterbox clips already on disk; skipping synthesis.", flush=True)

        for i in tqdm(range(max_chatterbox)):
            ref = bonafide_refs[i % len(bonafide_refs)]
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            out_path = cb_dir / f"cb_{i:05d}.wav"
            try:
                if not out_path.exists():
                    wav, sr = generate_chatterbox_clip(cb_model, prompt, Path(ref["path"]), turbo=use_turbo)
                    wav_16k = resample_audio(wav, sr, TARGET_SR)
                    save_wav_mono(out_path, wav_16k, TARGET_SR)
                records.append({
                    "id": f"cb_{i}",
                    "path": str(out_path.resolve()),
                    "label": 1,
                    "generator": "chatterbox_turbo" if use_turbo else "chatterbox",
                    "speaker_id": ref["speaker_id"],
                    "split": ref["split"],
                    "duration_s": _wav_duration_s(out_path),
                })
            except Exception as exc:
                print(f"[warn] Chatterbox clip {i} skipped: {exc}", flush=True)
        if cb_model is not None:
            del cb_model

    if max_kokoro > 0:
        kokoro_voices = ["af_heart", "af_bella", "am_adam", "am_michael", "bf_emma", "bm_george"]
        missing_k = [i for i in range(max_kokoro) if not (kokoro_dir / f"kokoro_{i:05d}.wav").exists()]
        pipeline = None
        if missing_k:
            from kokoro import KPipeline

            pipeline = KPipeline(lang_code="a")
            print(
                f"Generating {len(missing_k)}/{max_kokoro} held-out Kokoro-82M test clips "
                f"({max_kokoro - len(missing_k)} already on disk)...",
                flush=True,
            )
        else:
            print(f"All {max_kokoro} Kokoro clips already on disk; skipping synthesis.", flush=True)

        for i in tqdm(range(max_kokoro)):
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            voice = kokoro_voices[i % len(kokoro_voices)]
            out_path = kokoro_dir / f"kokoro_{i:05d}.wav"
            try:
                if not out_path.exists():
                    wav, sr = generate_kokoro_clip(pipeline, prompt, voice=voice)
                    wav_16k = resample_audio(wav, sr, TARGET_SR)
                    save_wav_mono(out_path, wav_16k, TARGET_SR)
                records.append({
                    "id": f"kokoro_{i}",
                    "path": str(out_path.resolve()),
                    "label": 1,
                    "generator": "kokoro_82m_unseen",
                    "speaker_id": f"kokoro_{voice}",
                    "split": "test",
                    "duration_s": _wav_duration_s(out_path),
                })
            except Exception as exc:
                print(f"[warn] Kokoro clip {i} skipped: {exc}", flush=True)

    manifest_path = out_dir / "manifest.jsonl"
    _write_manifest_atomic(manifest_path, records)
    print(f"Wrote {len(records)} rows to {manifest_path}", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CloakBreak dataset & 2026 synthetic spoofs")
    parser.add_argument("--source", choices=["hf", "local"], default="hf")
    parser.add_argument("--local-real-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/cloakbreak_v1"))
    parser.add_argument("--hf-dataset", type=str, default="SpeechAntiSpoofingBenchmarks/InTheWild")
    parser.add_argument("--max-real", type=int, default=12000)
    parser.add_argument("--max-chatterbox", type=int, default=3000)
    parser.add_argument("--max-kokoro", type=int, default=600)
    parser.add_argument("--use-turbo", action="store_true", default=True)
    parser.add_argument("--mock-generator", action="store_true")
    args = parser.parse_args()

    if args.source == "local":
        if args.local_real_dir is None:
            raise SystemExit("--local-real-dir is required when --source local")
        build_from_local_dir(
            real_dir=args.local_real_dir,
            out_dir=args.out_dir,
            max_chatterbox=args.max_chatterbox,
            max_kokoro=args.max_kokoro,
            mock_generator=args.mock_generator,
        )
    else:
        build_from_hf_on_gpu(
            out_dir=args.out_dir,
            hf_dataset=args.hf_dataset,
            max_real=args.max_real,
            max_chatterbox=args.max_chatterbox,
            max_kokoro=args.max_kokoro,
            use_turbo=args.use_turbo,
        )


if __name__ == "__main__":
    main()
