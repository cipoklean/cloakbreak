"""Restore the CloakBreak dataset on a fresh Vast box (Session 2).

The HF dataset repo holds ONLY the synthetic spoofs (``chatterbox/*``,
``kokoro_unseen/*``), ``disjoint_prompts.json`` and ``manifest.jsonl`` — real
clips are never uploaded. This module downloads that snapshot, then
re-materializes the missing REAL clips from the public upstream datasets
(InTheWild + LibriSpeech clean) by manifest ID, replicating the Session-1
``build_from_hf_on_gpu`` decode path byte-for-byte via the shared helpers in
``src.generate_spoofs``. Synthetic clips are NEVER regenerated: any missing
chatterbox/kokoro path is a hard failure.

CLI: ``python -m src.restore_dataset --repo REPO --out-dir DIR``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import soundfile as sf

from src.generate_spoofs import (
    DEFAULT_CLEAN_REAL_DATASET,
    _export_real_clip,
    _inthewild_utt_id,
    summarize_manifest_counts,
)

MIN_FILE_BYTES = 100
DURATION_TOLERANCE_S = 0.02
ITW_GENERATORS = {"bonafide_wild", "inthewild_spoof"}
LS_GENERATOR = "bonafide_clean"
LS_SPLIT_MAP = {"train": "train.100", "val": "validation", "test": "test"}
_UTT_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _is_synthetic(generator: object) -> bool:
    g = str(generator or "").lower()
    return "chatterbox" in g or "kokoro" in g


def _read_manifest(manifest_path: Path) -> List[Dict]:
    """Read manifest.jsonl; resolve relative paths against the manifest dir."""
    base = manifest_path.parent
    rows: List[Dict] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        p = Path(str(row["path"]))
        if not p.is_absolute():
            row["path"] = str((base / p).resolve())
        rows.append(row)
    return rows


def _file_duration_s(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate) if info.samplerate else 0.0


def _ls_utt_id(row: Dict, idx: int) -> str | None:
    """Sanitize an upstream LibriSpeech row id exactly like the Session-1 builder."""
    raw = row.get("id")
    if raw is None:
        return None
    return _UTT_SANITIZE_RE.sub("_", str(raw))


def _restore_from_rows(
    rows_iter: Iterable[Dict],
    needed: Dict[str, Path],
    id_fn: Callable[[Dict, int], str | None],
    min_s: float,
    max_s: float,
) -> Dict[str, float]:
    """Scan upstream ``rows_iter`` and export every row whose ``id_fn`` key is in ``needed``.

    ``needed`` maps upstream-id -> manifest target path and is mutated: an entry
    is removed once its upstream row is seen (restored or not), so leftovers are
    the IDs that never appeared upstream. Returns ``{key: duration_s}`` for the
    clips actually written. Prints progress every 1000 scanned rows.
    """
    total_needed = len(needed)
    restored: Dict[str, float] = {}
    scanned = 0
    for idx, row in enumerate(rows_iter):
        scanned += 1
        if scanned % 1000 == 0:
            print(
                f"[restore] scanned {scanned} upstream rows; "
                f"restored {len(restored)}/{total_needed}",
                flush=True,
            )
        key = id_fn(row, idx)
        if key is not None and key in needed:
            wav_path = needed.pop(key)
            audio_info = row.get("audio") if isinstance(row, dict) else None
            dur = _export_real_clip(audio_info or {}, wav_path, min_s, max_s)
            if dur is not None:
                restored[key] = dur
        if not needed:
            break
    print(
        f"[restore] scan finished after {scanned} rows; restored {len(restored)}/{total_needed}",
        flush=True,
    )
    return restored


def _resolve_hf_split(hf_dataset: str, hf_split: str) -> str:
    """Same auto-resolution as the Session-1 builder (prefer 'test' when train is absent)."""
    if hf_split != "auto":
        return hf_split
    from datasets import get_dataset_split_names

    avail = list(get_dataset_split_names(hf_dataset))
    if "test" in avail and "train" not in avail:
        return "test"
    if "train" in avail:
        return "train"
    return avail[0]


def _restore_inthewild(
    hf_dataset: str,
    hf_split: str,
    needed: Dict[str, Path],
    seed: int,
    min_s: float,
    max_s: float,
) -> Dict[str, float]:
    """Re-stream InTheWild (non-streaming, shuffled) exactly like Session 1."""
    from datasets import Audio, load_dataset

    resolved = _resolve_hf_split(hf_dataset, hf_split)
    print(
        f"[restore] loading {hf_dataset} split='{resolved}' for {len(needed)} InTheWild clips...",
        flush=True,
    )
    ds = load_dataset(hf_dataset, split=resolved).cast_column("audio", Audio(decode=False))
    ds = ds.shuffle(seed=seed)
    return _restore_from_rows(ds, needed, _inthewild_utt_id, min_s, max_s)


def _restore_librispeech(
    clean_real_dataset: str,
    needed_by_split: Dict[str, Dict[str, Path]],
    seed: int,
    min_s: float,
    max_s: float,
) -> Dict[str, Dict[str, float]]:
    """Re-stream LibriSpeech clean splits (streaming + shuffle, same as Session 1)."""
    from datasets import Audio, load_dataset

    results: Dict[str, Dict[str, float]] = {}
    for target_split, hf_split in LS_SPLIT_MAP.items():
        needed = needed_by_split.get(target_split) or {}
        if not needed:
            continue
        wanted = dict(needed)
        print(
            f"[restore] streaming {clean_real_dataset} clean/{hf_split} "
            f"for {len(wanted)} LibriSpeech {target_split} clips...",
            flush=True,
        )
        ds = load_dataset(clean_real_dataset, "clean", split=hf_split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        ds = ds.shuffle(seed=seed, buffer_size=1500)
        restored = _restore_from_rows(ds, needed, _ls_utt_id, min_s, max_s)
        results[target_split] = restored
        missing = sorted(set(wanted) - set(restored))
        if missing:
            examples = ", ".join(missing[:5])
            raise RuntimeError(
                f"[restore] FAIL: {len(missing)} LibriSpeech '{target_split}' clips were not found "
                f"in {clean_real_dataset} clean/{hf_split} before the stream ended "
                f"(e.g. {examples}). The manifest cannot be fully restored."
            )
    return results


def _download_snapshot(repo_id: str, out_dir: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    print(f"[restore] snapshot_download {repo_id} -> {out_dir} (resumable)...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        token=token,
        ignore_patterns=["smoke_check/*"],
    )
    print(f"[restore] snapshot ready in {out_dir}", flush=True)


def _verify_manifest(rows: List[Dict], expected_rows: int | None) -> List[str]:
    """Check every row: file exists (>100B) and duration matches manifest ±0.02s."""
    failures: List[str] = []
    verified = 0
    for row in rows:
        p = Path(str(row["path"]))
        rid = row.get("id", p.name)
        if not p.exists() or p.stat().st_size <= MIN_FILE_BYTES:
            failures.append(f"missing or too small ({p.stat().st_size if p.exists() else 0}B): {rid} -> {p}")
            continue
        dur_manifest = row.get("duration_s")
        if dur_manifest is not None:
            try:
                dur_file = _file_duration_s(p)
            except Exception as exc:
                failures.append(f"unreadable audio {rid} -> {p}: {exc}")
                continue
            if abs(dur_file - float(dur_manifest)) > DURATION_TOLERANCE_S:
                failures.append(
                    f"duration mismatch {rid} -> {p}: file={dur_file:.3f}s manifest={float(dur_manifest):.3f}s"
                )
                continue
        verified += 1
    if verified != len(rows):
        failures.insert(0, f"verified {verified}/{len(rows)} manifest rows")
    if expected_rows is not None and len(rows) != int(expected_rows):
        failures.append(f"manifest row count {len(rows)} != expected {int(expected_rows)}")
    return failures


def restore_dataset(
    repo_id: str,
    out_dir: Path,
    hf_dataset: str = "SpeechAntiSpoofingBenchmarks/InTheWild",
    clean_real_dataset: str = DEFAULT_CLEAN_REAL_DATASET,
    hf_split: str = "auto",
    seed: int = 42,
    expected_rows: int | None = None,
    skip_download: bool = False,
    skip_upstream: bool = False,
    min_duration: float = 2.5,
    max_duration: float = 10.0,
    splits: Iterable[str] | None = None,
) -> int:
    """Download snapshot, restore missing real clips, verify. Returns process exit code."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        token = os.environ.get("HF_TOKEN", "").strip() or None
        _download_snapshot(repo_id, out_dir, token)

    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        print(
            f"[restore] FAIL: {manifest_path} not found"
            + ("" if skip_download else f" after downloading {repo_id}"),
            file=sys.stderr,
            flush=True,
        )
        return 2

    rows = _read_manifest(manifest_path)
    if not rows:
        print(f"[restore] FAIL: {manifest_path} is empty", file=sys.stderr, flush=True)
        return 2

    if splits is not None:
        wanted_splits = {str(s).strip() for s in splits if str(s).strip()}
        if wanted_splits:
            rows = [r for r in rows if str(r.get("split") or "") in wanted_splits]
            per_split: Dict[str, int] = {}
            for r in rows:
                s = str(r.get("split") or "")
                per_split[s] = per_split.get(s, 0) + 1
            print(
                f"[restore] --splits {sorted(wanted_splits)}: "
                f"{len(rows)} manifest rows selected ({per_split})",
                flush=True,
            )
            if not rows:
                print(
                    f"[restore] WARN: --splits {sorted(wanted_splits)} matched 0 manifest rows",
                    file=sys.stderr,
                    flush=True,
                )

    missing_synth = [
        r for r in rows
        if _is_synthetic(r.get("generator"))
        and not (Path(str(r["path"])).exists() and Path(str(r["path"])).stat().st_size > MIN_FILE_BYTES)
    ]
    if missing_synth:
        examples = ", ".join(str(r.get("path")) for r in missing_synth[:5])
        print(
            f"[restore] FAIL: missing {len(missing_synth)} synthetic clips (e.g. {examples}); "
            "Session 2 never regenerates spoofs — re-check the HF dataset repo.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    def _missing_real(row: Dict) -> bool:
        p = Path(str(row["path"]))
        return not (p.exists() and p.stat().st_size > MIN_FILE_BYTES)

    missing_real = [r for r in rows if not _is_synthetic(r.get("generator")) and _missing_real(r)]
    print(f"[restore] manifest rows: {len(rows)}; missing real clips: {len(missing_real)}", flush=True)

    if missing_real and not skip_upstream:
        itw_needed: Dict[str, Path] = {}
        ls_needed: Dict[str, Dict[str, Path]] = {s: {} for s in LS_SPLIT_MAP}
        other_missing: List[Dict] = []
        for r in missing_real:
            gen = str(r.get("generator") or "")
            target = Path(str(r["path"]))
            if gen in ITW_GENERATORS:
                itw_needed[target.stem] = target
            elif gen == LS_GENERATOR:
                split = str(r.get("split") or "")
                if split in ls_needed:
                    ls_needed[split][target.stem] = target
                else:
                    other_missing.append(r)
            else:
                other_missing.append(r)

        if itw_needed:
            wanted = dict(itw_needed)
            restored = _restore_inthewild(hf_dataset, hf_split, itw_needed, seed, min_duration, max_duration)
            leftover = sorted(set(wanted) - set(restored))
            if leftover:
                print(
                    f"[restore] FAIL: {len(leftover)} InTheWild clips not found upstream "
                    f"(e.g. {', '.join(leftover[:5])}).",
                    file=sys.stderr,
                    flush=True,
                )
                return 2

        if any(ls_needed.values()):
            _restore_librispeech(clean_real_dataset, ls_needed, seed, min_duration, max_duration)

        if other_missing:
            print(
                f"[restore] FAIL: {len(other_missing)} real clips have generators that cannot be "
                f"restored upstream (e.g. {other_missing[0].get('generator')}).",
                file=sys.stderr,
                flush=True,
            )
            return 2
    elif missing_real:
        print(
            f"[restore] {len(missing_real)} real clips missing and --skip-upstream set; "
            "verification will report them.",
            flush=True,
        )

    failures = _verify_manifest(rows, expected_rows)
    if failures:
        print(f"[restore] FAIL: dataset verification failed ({len(failures)} issue(s)):", file=sys.stderr, flush=True)
        for f in failures[:20]:
            print(f"  - {f}", file=sys.stderr, flush=True)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr, flush=True)
        return 2

    print(json.dumps(summarize_manifest_counts(rows), indent=2), flush=True)
    print(f"[restore] manifest rows: {len(rows)} (verified {len(rows)})", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore CloakBreak dataset from HF snapshot + upstream reals")
    parser.add_argument("--repo", type=str, required=True, help="HF dataset repo id (e.g. Cipoklean/cloakbreak-data-v1)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Dataset directory containing manifest.jsonl")
    parser.add_argument("--hf-dataset", type=str, default="SpeechAntiSpoofingBenchmarks/InTheWild")
    parser.add_argument("--hf-split", type=str, default="auto", help="InTheWild split override (default: auto)")
    parser.add_argument("--clean-real-dataset", type=str, default=DEFAULT_CLEAN_REAL_DATASET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true", help="Do not snapshot_download (manifest must exist)")
    parser.add_argument("--skip-upstream", action="store_true", help="Verify only; never touch upstream datasets")
    parser.add_argument("--min-duration", type=float, default=2.5)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated manifest splits to restore/verify (e.g. 'val,test'); default: all",
    )
    args = parser.parse_args()

    expected = args.expected_rows
    if expected is None:
        env_expected = os.environ.get("EXPECTED_MANIFEST_ROWS", "").strip()
        if env_expected:
            try:
                expected = int(env_expected)
            except ValueError:
                print(f"[restore] FAIL: EXPECTED_MANIFEST_ROWS={env_expected!r} is not an integer", file=sys.stderr)
                sys.exit(2)

    code = restore_dataset(
        repo_id=args.repo,
        out_dir=args.out_dir,
        hf_dataset=args.hf_dataset,
        clean_real_dataset=args.clean_real_dataset,
        hf_split=args.hf_split,
        seed=args.seed,
        expected_rows=expected,
        skip_download=args.skip_download,
        skip_upstream=args.skip_upstream,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        splits=[s for s in args.splits.split(",")] if args.splits else None,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
