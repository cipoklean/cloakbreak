from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.audio_utils import (
    TARGET_SR,
    load_wav_bytes_mono,
    load_wav_mono,
    normalize_loudness,
    resample_audio,
    save_wav_mono,
)

# Open, non-gated English sentence corpus (license: ODC-BY, ~1.53M sentences)
DEFAULT_TEXT_DATASET = "agentlans/high-quality-english-sentences"

# Reserved strictly for the live UI demo — NEVER used in training or evaluation.
DEMO_PROMPTS = [
    "Hey, can you confirm that wire transfer before the bank closes today?",
    "I lost my phone at the airport and I am calling from a colleague's number.",
    "Please send the verification code as soon as it arrives by text.",
    "We need to update the payroll account details before Friday afternoon.",
]

_CLEAN_SENTENCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 ,.'?!;-]+[.?!]$")


class _NoOpWatermarker:
    """Replaces Chatterbox's Perth watermarker during defensive training-data generation.

    Why: attackers using open-source Chatterbox can disable the watermark in one
    line. Training on unwatermarked clones forces the detector to learn synthesis
    artifacts rather than watermark presence.
    """

    def apply_watermark(self, wav: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        return wav


def normalize_sentence_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_clean_spoken_sentence(text: str, min_words: int = 8, max_words: int = 18) -> bool:
    s = " ".join(text.strip().split())
    if not s or "..." in s or "http" in s.lower() or "(" in s or ")" in s or '"' in s:
        return False
    if any(ch.isdigit() for ch in s):
        return False
    words = s.split(" ")
    if not (min_words <= len(words) <= max_words):
        return False
    return bool(_CLEAN_SENTENCE_RE.match(s))


def load_diverse_sentences_from_hf(
    total_needed: int,
    dataset_id: str = DEFAULT_TEXT_DATASET,
    seed: int = 42,
) -> List[str]:
    """Stream clean, diverse English sentences from an open HF text dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train", streaming=True)
    seen: set[str] = set()
    demo_keys = {normalize_sentence_key(p) for p in DEMO_PROMPTS}
    collected: List[str] = []

    for row in ds:
        raw = str(row.get("text") or "").strip()
        if not is_clean_spoken_sentence(raw):
            continue
        key = normalize_sentence_key(raw)
        if not key or key in seen or key in demo_keys:
            continue
        seen.add(key)
        collected.append(" ".join(raw.split()))
        if len(collected) >= total_needed:
            break

    if len(collected) < total_needed:
        raise RuntimeError(
            f"Only collected {len(collected)} clean sentences from {dataset_id}; needed {total_needed}."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(collected))
    return [collected[int(i)] for i in order]


def make_synthetic_test_sentences(total_needed: int) -> List[str]:
    """Deterministic unique sentences for offline CPU unit tests only."""
    subjects = ["The baker", "A traveler", "My neighbor", "The musician", "Our captain", "An architect"]
    verbs = ["carried", "repaired", "sketched", "inspected", "delivered", "described"]
    objects = ["the wooden lantern", "a brass compass", "the morning newspaper", "a ceramic bowl", "the garden map"]
    places = ["near the quiet harbor", "across the old bridge", "beside the stone library", "along the sunlit avenue"]
    out: List[str] = []
    idx = 0
    for s in subjects:
        for v in verbs:
            for o in objects:
                for p in places:
                    out.append(f"{s} {v} {o} {p} during batch number {idx}.")
                    idx += 1
                    if len(out) >= total_needed:
                        return out
    return out


def split_disjoint_sentence_pools(
    sentences: Sequence[str],
    n_chatterbox: int,
    n_kokoro: int,
) -> Tuple[List[str], List[str]]:
    """Return strictly disjoint sentence pools for Chatterbox and held-out Kokoro."""
    unique: List[str] = []
    seen: set[str] = set()
    for s in sentences:
        k = normalize_sentence_key(s)
        if k and k not in seen:
            seen.add(k)
            unique.append(s)

    needed = n_chatterbox + n_kokoro
    if len(unique) < needed:
        raise ValueError(f"Need at least {needed} unique sentences for disjoint pools, got {len(unique)}")

    cb_pool = unique[:n_chatterbox]
    kokoro_pool = unique[n_chatterbox : n_chatterbox + n_kokoro]
    cb_keys = {normalize_sentence_key(s) for s in cb_pool}
    kokoro_keys = {normalize_sentence_key(s) for s in kokoro_pool}
    if cb_keys & kokoro_keys:
        raise ValueError("Sentence leakage detected between Chatterbox and Kokoro pools")
    return cb_pool, kokoro_pool


def assert_disjoint_text_splits(records: Sequence[Dict]) -> None:
    """Hard invariant check preventing sentence leakage into held-out Kokoro or across splits."""
    train_texts = {
        normalize_sentence_key(r["text"])
        for r in records
        if r.get("split") in {"train", "val"} and r.get("text")
    }
    chatterbox_texts = {
        normalize_sentence_key(r["text"])
        for r in records
        if "chatterbox" in str(r.get("generator", "")) and r.get("text")
    }
    kokoro_texts = {
        normalize_sentence_key(r["text"])
        for r in records
        if "kokoro" in str(r.get("generator", "")) and r.get("text")
    }
    leak_train = train_texts & kokoro_texts
    leak_cb = chatterbox_texts & kokoro_texts
    if leak_train or leak_cb:
        raise ValueError(
            f"Sentence leakage detected: {len(leak_train)} Kokoro sentences overlap train/val, "
            f"{len(leak_cb)} overlap Chatterbox."
        )


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
    assert_disjoint_text_splits(records)
    tmp_path = manifest_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    tmp_path.replace(manifest_path)


def synthesize_mock_clone(ref_audio: np.ndarray, seed: int) -> np.ndarray:
    """Fast CPU synthetic waveform matched in duration & loudness for unit tests."""
    rng = np.random.default_rng(seed)
    n = max(int(TARGET_SR * 2.5), min(ref_audio.shape[0], TARGET_SR * 4))
    t = np.linspace(0.0, n / TARGET_SR, n, endpoint=False, dtype=np.float32)
    f0 = float(rng.uniform(140.0, 260.0))
    wave = 0.25 * np.sin(2.0 * np.pi * f0 * t) + 0.12 * np.sin(2.0 * np.pi * (2.0 * f0) * t)
    wave += rng.normal(0.0, 0.015, size=n).astype(np.float32)
    return normalize_loudness(wave)


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
        return np.zeros(int(TARGET_SR * 2.5), dtype=np.float32), 24000
    return np.concatenate(chunks), 24000


def build_from_local_dir(
    real_dir: Path,
    out_dir: Path,
    max_chatterbox: int = 20,
    max_kokoro: int = 10,
    mock_generator: bool = True,
    sentences: Sequence[str] | None = None,
) -> Path:
    """Resumable local builder with loudness normalization and disjoint sentences."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(real_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files found in {real_dir}")

    n_cb = min(max_chatterbox, len(wavs))
    pool_src = list(sentences) if sentences is not None else make_synthetic_test_sentences(n_cb + max_kokoro + 10)
    cb_sentences, kokoro_sentences = split_disjoint_sentence_pools(pool_src, n_cb, max_kokoro)

    norm_real_dir = out_dir / "real_norm"
    norm_real_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    for idx, wav_path in enumerate(wavs):
        speaker_id = wav_path.stem.split("_")[0] or f"spk{idx}"
        split = assign_speaker_split(speaker_id)
        norm_path = norm_real_dir / wav_path.name
        if not norm_path.exists():
            audio, sr = load_wav_mono(wav_path, target_sr=TARGET_SR)
            save_wav_mono(norm_path, normalize_loudness(audio), sr)
        records.append({
            "id": f"real_{idx}",
            "path": str(norm_path.relative_to(out_dir)),
            "label": 0,
            "generator": "bonafide",
            "watermarked": False,
            "speaker_id": speaker_id,
            "split": split,
            "duration_s": _wav_duration_s(norm_path),
        })

    cb_dir = out_dir / "chatterbox"
    cb_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_cb):
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
            "path": str(out_path.relative_to(out_dir)),
            "label": 1,
            "generator": "chatterbox_mock",
            "watermarked": False,
            "text": cb_sentences[i],
            "speaker_id": speaker_id,
            "split": split,
            "duration_s": _wav_duration_s(out_path),
        })

    kokoro_dir = out_dir / "kokoro_unseen"
    kokoro_dir.mkdir(parents=True, exist_ok=True)
    for i in range(max_kokoro):
        out_path = kokoro_dir / f"kokoro_{i:05d}.wav"
        if not out_path.exists():
            synth = synthesize_mock_clone(np.zeros(int(TARGET_SR * 2.5), dtype=np.float32), seed=5000 + i)
            save_wav_mono(out_path, synth, TARGET_SR)
        records.append({
            "id": f"kokoro_{i}",
            "path": str(out_path.relative_to(out_dir)),
            "label": 1,
            "generator": "kokoro_unseen_mock",
            "watermarked": False,
            "text": kokoro_sentences[i],
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
    text_dataset: str = DEFAULT_TEXT_DATASET,
    max_real: int = 12000,
    max_chatterbox: int = 3000,
    max_kokoro: int = 600,
    use_turbo: bool = True,
    disable_watermark: bool = True,
    min_duration_s: float = 2.5,
    max_duration_s: float = 10.0,
) -> Path:
    """Resumable GPU dataset builder with strict anti-shortcut controls.

    1. Diverse open sentences streamed from agentlans/high-quality-english-sentences (ODC-BY).
    2. Strictly disjoint sentence sets between Chatterbox and held-out Kokoro-82M.
    3. Chatterbox Perth watermark disabled by default for defensive training data.
    4. Identical RMS loudness normalization & duration band [2.5s, 10.0s] across real and fake.
    5. Throughput (clips/sec) logged live to flag underperforming GPU hosts.
    """
    from datasets import Audio, load_dataset
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    real_dir = out_dir / "inthewild_wav"
    cb_dir = out_dir / "chatterbox"
    kokoro_dir = out_dir / "kokoro_unseen"
    for d in (real_dir, cb_dir, kokoro_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Cache the exact deterministic sentence pools to disk so resumed runs use identical prompts
    prompt_cache = out_dir / "disjoint_prompts.json"
    if prompt_cache.exists():
        cached = json.loads(prompt_cache.read_text(encoding="utf-8"))
        cb_sentences = cached["chatterbox"]
        kokoro_sentences = cached["kokoro"]
        if len(cb_sentences) < max_chatterbox or len(kokoro_sentences) < max_kokoro:
            raise RuntimeError("Cached disjoint_prompts.json smaller than requested counts; delete it to regenerate.")
        print(
            f"Reusing cached disjoint prompt pools: {len(cb_sentences)} Chatterbox, {len(kokoro_sentences)} Kokoro.",
            flush=True,
        )
    else:
        print(f"Streaming {max_chatterbox + max_kokoro} unique sentences from {text_dataset} (ODC-BY)...", flush=True)
        all_sentences = load_diverse_sentences_from_hf(max_chatterbox + max_kokoro, dataset_id=text_dataset, seed=42)
        cb_sentences, kokoro_sentences = split_disjoint_sentence_pools(all_sentences, max_chatterbox, max_kokoro)
        prompt_cache.write_text(
            json.dumps({"source": text_dataset, "chatterbox": cb_sentences, "kokoro": kokoro_sentences}, indent=2),
            encoding="utf-8",
        )

    ds = load_dataset(hf_dataset, split="train").cast_column("audio", Audio(decode=False))
    records: List[Dict] = []
    bonafide_refs: List[Dict] = []

    skipped_real = 0
    print(f"Exporting up to {max_real} loudness-normalized InTheWild clips ({min_duration_s}-{max_duration_s}s)...", flush=True)
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
            if dur < min_duration_s or dur > max_duration_s:
                continue
            save_wav_mono(wav_path, normalize_loudness(audio_16k), TARGET_SR)

        if dur < min_duration_s or dur > max_duration_s:
            continue
        split = assign_speaker_split(speaker_id)
        rec = {
            "id": f"itw_{idx}",
            "path": str(wav_path.relative_to(out_dir)),
            "label": label,
            "generator": "bonafide" if label == 0 else "inthewild_spoof",
            "watermarked": False,
            "speaker_id": speaker_id,
            "split": split,
            "duration_s": round(dur, 3),
        }
        records.append(rec)
        if label == 0 and 3.0 <= dur <= 9.0:
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
            if disable_watermark:
                cb_model.watermarker = _NoOpWatermarker()
                print("[watermark] Chatterbox Perth watermarker DISABLED for defensive training data.", flush=True)
            else:
                print("[watermark] Chatterbox Perth watermarker ENABLED.", flush=True)
            print(
                f"Generating {len(missing_cb)}/{max_chatterbox} Chatterbox clones "
                f"({max_chatterbox - len(missing_cb)} already on disk)...",
                flush=True,
            )
        else:
            print(f"All {max_chatterbox} Chatterbox clips already on disk; skipping synthesis.", flush=True)

        t_cb = time.perf_counter()
        generated_cb = 0
        for i in tqdm(range(max_chatterbox)):
            ref = bonafide_refs[i % len(bonafide_refs)]
            prompt = cb_sentences[i]
            out_path = cb_dir / f"cb_{i:05d}.wav"
            try:
                if not out_path.exists():
                    wav, sr = generate_chatterbox_clip(cb_model, prompt, out_dir / ref["path"], turbo=use_turbo)
                    wav_16k = normalize_loudness(resample_audio(wav, sr, TARGET_SR))
                    if wav_16k.shape[0] > int(max_duration_s * TARGET_SR):
                        wav_16k = wav_16k[: int(max_duration_s * TARGET_SR)]
                    save_wav_mono(out_path, wav_16k, TARGET_SR)
                    generated_cb += 1
                dur = _wav_duration_s(out_path)
                if dur < min_duration_s:
                    continue
                records.append({
                    "id": f"cb_{i}",
                    "path": str(out_path.relative_to(out_dir)),
                    "label": 1,
                    "generator": "chatterbox_turbo" if use_turbo else "chatterbox",
                    "watermarked": not disable_watermark,
                    "text": prompt,
                    "speaker_id": ref["speaker_id"],
                    "split": ref["split"],
                    "duration_s": dur,
                })
            except Exception as exc:
                print(f"[warn] Chatterbox clip {i} skipped: {exc}", flush=True)
        if generated_cb > 0:
            elapsed_cb = max(time.perf_counter() - t_cb, 1e-6)
            cps = generated_cb / elapsed_cb
            print(f"[throughput] Chatterbox: {generated_cb} clips in {elapsed_cb:.1f}s ({cps:.2f} clips/sec)", flush=True)
            if cps < 0.5:
                print("[warn] Chatterbox throughput < 0.5 clips/sec — host GPU may be throttled.", flush=True)
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
                f"(disjoint sentences, unwatermarked)...",
                flush=True,
            )
        else:
            print(f"All {max_kokoro} Kokoro clips already on disk; skipping synthesis.", flush=True)

        t_k = time.perf_counter()
        generated_k = 0
        for i in tqdm(range(max_kokoro)):
            prompt = kokoro_sentences[i]
            voice = kokoro_voices[i % len(kokoro_voices)]
            out_path = kokoro_dir / f"kokoro_{i:05d}.wav"
            try:
                if not out_path.exists():
                    wav, sr = generate_kokoro_clip(pipeline, prompt, voice=voice)
                    wav_16k = normalize_loudness(resample_audio(wav, sr, TARGET_SR))
                    if wav_16k.shape[0] > int(max_duration_s * TARGET_SR):
                        wav_16k = wav_16k[: int(max_duration_s * TARGET_SR)]
                    save_wav_mono(out_path, wav_16k, TARGET_SR)
                    generated_k += 1
                dur = _wav_duration_s(out_path)
                if dur < min_duration_s:
                    continue
                records.append({
                    "id": f"kokoro_{i}",
                    "path": str(out_path.relative_to(out_dir)),
                    "label": 1,
                    "generator": "kokoro_82m_unseen",
                    "watermarked": False,
                    "text": prompt,
                    "speaker_id": f"kokoro_{voice}",
                    "split": "test",
                    "duration_s": dur,
                })
            except Exception as exc:
                print(f"[warn] Kokoro clip {i} skipped: {exc}", flush=True)
        if generated_k > 0:
            elapsed_k = max(time.perf_counter() - t_k, 1e-6)
            print(f"[throughput] Kokoro-82M: {generated_k} clips in {elapsed_k:.1f}s ({generated_k / elapsed_k:.2f} clips/sec)", flush=True)

    manifest_path = out_dir / "manifest.jsonl"
    _write_manifest_atomic(manifest_path, records)
    print(f"Wrote {len(records)} rows to {manifest_path} (disjoint sentence check PASSED).", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CloakBreak dataset & 2026 synthetic spoofs")
    parser.add_argument("--source", choices=["hf", "local"], default="hf")
    parser.add_argument("--local-real-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/cloakbreak_v1"))
    parser.add_argument("--hf-dataset", type=str, default="SpeechAntiSpoofingBenchmarks/InTheWild")
    parser.add_argument("--text-dataset", type=str, default=DEFAULT_TEXT_DATASET)
    parser.add_argument("--max-real", type=int, default=12000)
    parser.add_argument("--max-chatterbox", type=int, default=3000)
    parser.add_argument("--max-kokoro", type=int, default=600)
    parser.add_argument("--use-turbo", action="store_true", default=True)
    parser.add_argument("--keep-watermark", action="store_true", help="Keep Chatterbox Perth watermark (default: disabled)")
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
            text_dataset=args.text_dataset,
            max_real=args.max_real,
            max_chatterbox=args.max_chatterbox,
            max_kokoro=args.max_kokoro,
            use_turbo=args.use_turbo,
            disable_watermark=not args.keep_watermark,
        )


if __name__ == "__main__":
    main()
