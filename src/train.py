from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.audio_utils import TARGET_SR, load_wav_mono, normalize_loudness, pad_or_crop
from src.augment import TrainCodecAugmenter, has_ffmpeg, precompute_opus_cache
from src.eval import compute_eer_and_auroc
from src.generate_spoofs import assert_disjoint_text_splits
from src.model import (
    DEFAULT_BACKBONE_ID,
    DEFAULT_WINDOW_SAMPLES,
    CloakBreakDetector,
    export_to_onnx,
    quantize_onnx_int8,
)

SEED_BASE = 42


class BudgetExceeded(RuntimeError):
    """Raised when ``deadline_unix`` passes mid-training; last.pt is already saved."""

    def __init__(
        self,
        stage: str,
        epoch: int,
        global_steps: int,
        elapsed_sec: float,
        uploaded: bool = True,
    ) -> None:
        self.stage = stage
        self.epoch = epoch
        self.global_steps = global_steps
        self.elapsed_sec = elapsed_sec
        self.uploaded = uploaded
        super().__init__(
            f"training deadline exceeded at stage={stage} epoch={epoch} global_steps={global_steps}"
        )


def _seed_dataloader_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed_worker"):
        info.dataset.reseed_worker(worker_seed)


class SpoofAudioDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
        augmenter: TrainCodecAugmenter | None = None,
        opus_cache_map: Dict[str, str] | None = None,
        random_crop: bool = False,
    ) -> None:
        self.rows = rows
        self.window_samples = window_samples
        self.augmenter = augmenter
        self.opus_cache_map = opus_cache_map or {}
        self.random_crop = random_crop
        self.rng = np.random.default_rng(42)

    def reseed_worker(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        if self.augmenter is not None:
            self.augmenter.reseed(seed + 17)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        src_path = row["path"]
        wav, sr = load_wav_mono(src_path, target_sr=TARGET_SR)
        wav = pad_or_crop(wav, self.window_samples, random_crop=self.random_crop, rng=self.rng)

        cached_opus = None
        if self.augmenter is not None and src_path in self.opus_cache_map:
            opus_wav, _ = load_wav_mono(self.opus_cache_map[src_path], target_sr=TARGET_SR)
            cached_opus = pad_or_crop(opus_wav, self.window_samples, random_crop=self.random_crop, rng=self.rng)

        if self.augmenter is not None:
            wav = self.augmenter(wav, sr=sr, cached_opus_waveform=cached_opus)
        else:
            wav = normalize_loudness(wav)
        return torch.from_numpy(wav), torch.tensor(float(row["label"]), dtype=torch.float32)


def export_from_checkpoint(
    best_ckpt: Path,
    out_dir: Path,
    backbone_id: str = DEFAULT_BACKBONE_ID,
    use_tiny_test_backbone: bool = False,
) -> tuple[Path, Path]:
    """Load a trained ``cloakbreak_best.pt`` and export FP32 + INT8 ONNX artifacts."""
    best_ckpt = Path(best_ckpt)
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    meta_backbone = ckpt.get("backbone_id", backbone_id)
    use_tiny = bool(use_tiny_test_backbone) or meta_backbone == "tiny-test"
    model = CloakBreakDetector(backbone_id=meta_backbone, use_tiny_test_backbone=use_tiny)
    model.load_state_dict(ckpt["state_dict"])
    model.to("cpu").eval()
    fp32_onnx = export_to_onnx(model, out_dir / "cloakbreak_fp32.onnx")
    int8_onnx = quantize_onnx_int8(fp32_onnx, out_dir / "cloakbreak_int8.onnx")
    return fp32_onnx, int8_onnx


def resolve_resume_checkpoint(
    spec: str,
    out_dir: Path,
    hf_repo: str | None = None,
    hf_prefix: str = "checkpoints",
    token: str | None = None,
) -> Path | None:
    """Resolve ``--resume-from`` to a local checkpoint path (or None for fresh start).

    ``"none"`` -> None; an existing local path -> itself; ``"auto"`` ->
    ``out_dir/last.pt`` when present, else download ``{hf_prefix}/last.pt`` (and
    ``cloakbreak_best.pt`` for best-EER tracking) from the HF model repo.
    """
    spec = (spec or "none").strip()
    if spec.lower() in {"", "none", "no", "off", "false", "0"}:
        return None
    local = Path(spec)
    if local.exists():
        print(f"[resume] using local checkpoint {local}", flush=True)
        return local
    if spec.lower() != "auto":
        raise FileNotFoundError(
            f"--resume-from '{spec}' is not an existing file; use 'auto', 'none', or a real path."
        )
    last_local = Path(out_dir) / "last.pt"
    if last_local.exists():
        print(f"[resume] found local checkpoint {last_local}", flush=True)
        return last_local
    if hf_repo:
        from src import hub

        prefix = hf_prefix.strip("/")
        remote = hub.download_checkpoint(hf_repo, f"{prefix}/last.pt", Path(out_dir), token=token)
        if remote is not None:
            # hf_hub_download preserves the repo path (out_dir/<prefix>/last.pt);
            # move it to out_dir/last.pt so the next 'auto' resume finds it locally.
            local_last = Path(out_dir) / "last.pt"
            if Path(remote) != local_last:
                Path(remote).replace(local_last)
                remote = local_last
            best_remote = hub.download_checkpoint(
                hf_repo, f"{prefix}/cloakbreak_best.pt", Path(out_dir), token=token
            )
            if best_remote is not None:
                local_best = Path(out_dir) / "cloakbreak_best.pt"
                if not local_best.exists() and Path(best_remote) != local_best:
                    Path(best_remote).replace(local_best)
            print(f"[resume] downloaded {hf_repo}:{prefix}/last.pt -> {remote}", flush=True)
            return remote
    print("[resume] no checkpoint found — starting fresh", flush=True)
    return None


def train_model(
    manifest_path: Path,
    out_dir: Path,
    backbone_id: str = DEFAULT_BACKBONE_ID,
    epochs_stage1: int = 2,
    epochs_stage2: int = 6,
    batch_size: int = 24,
    lr_stage1: float = 3e-4,
    lr_stage2: float = 2e-5,
    unfreeze_layers: int = 6,
    use_tiny_test_backbone: bool = False,
    precompute_opus: bool = True,
    opus_cache_dir: Path | None = None,
    num_workers: int | None = None,
    max_steps: int | None = None,
    device_str: str = "cuda",
    resume_from: Path | None = None,
    deadline_unix: float | None = None,
    checkpoint_callback: Callable[[List[Path], Dict], None] | None = None,
    eta_after_steps: int = 50,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")
    train_t0 = time.time()

    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p = Path(r["path"])
        if not p.is_absolute():
            r["path"] = str((manifest_path.parent / p).resolve())
        rows.append(r)
    assert_disjoint_text_splits(rows)

    train_rows = [r for r in rows if r.get("split") == "train" and "kokoro" not in r.get("generator", "")]
    val_rows = [r for r in rows if r.get("split") == "val" and "kokoro" not in r.get("generator", "")]
    if not train_rows or not val_rows:
        mid = max(1, int(len(rows) * 0.8))
        train_rows, val_rows = rows[:mid], rows[mid:]

    train_spks = {r.get("speaker_id") for r in train_rows}
    val_spks = {r.get("speaker_id") for r in val_rows}
    overlap = train_spks & val_spks
    if overlap and len(rows) > 10:
        print(f"[warn] {len(overlap)} overlapping speakers between train/val", flush=True)

    n_pos = sum(1 for r in train_rows if int(r["label"]) == 1)
    n_neg = max(1, len(train_rows) - n_pos)
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)

    if num_workers is None:
        workers = min(os.cpu_count() or 2, 8) if device.type == "cuda" else 0
    else:
        workers = max(0, num_workers)

    opus_cache_map: Dict[str, str] = {}
    if precompute_opus and has_ffmpeg():
        cache_dir = opus_cache_dir or (manifest_path.parent / "opus_cache_16k")
        t_cache = time.perf_counter()
        opus_cache_map = precompute_opus_cache(
            [r["path"] for r in train_rows],
            cache_dir=cache_dir,
            bitrate_kbps=16,
            num_workers=max(1, min(os.cpu_count() or 2, 12)),
        )
        elapsed_c = max(time.perf_counter() - t_cache, 1e-6)
        print(
            f"[cache] Opus 16k cache ready for {len(opus_cache_map)} train clips in {elapsed_c:.1f}s "
            f"({len(opus_cache_map) / elapsed_c:.1f} clips/sec).",
            flush=True,
        )

    augmenter = TrainCodecAugmenter(allow_inline_opus=False, seed=42)
    train_ds = SpoofAudioDataset(train_rows, augmenter=augmenter, opus_cache_map=opus_cache_map, random_crop=True)
    val_ds = SpoofAudioDataset(val_rows, augmenter=None, random_crop=False)

    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "worker_init_fn": _seed_dataloader_worker if workers > 0 else None,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    steps_per_epoch = len(train_loader)

    model = CloakBreakDetector(backbone_id=backbone_id, use_tiny_test_backbone=use_tiny_test_backbone).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_eer = 101.0
    best_ckpt = out_dir / "cloakbreak_best.pt"
    last_ckpt = out_dir / "last.pt"
    history_path = out_dir / "train_history.json"
    history: List[Dict] = []
    global_steps = 0

    resume_payload: Dict | None = None
    resume_stage: str | None = None
    resume_next_epoch = 1
    if resume_from is not None:
        resume_payload = torch.load(resume_from, map_location="cpu", weights_only=True)
        if not isinstance(resume_payload, dict) or resume_payload.get("version") != 1:
            raise ValueError(f"Unsupported checkpoint format in {resume_from} (expected version=1).")
        ckpt_backbone = resume_payload.get("backbone_id")
        if ckpt_backbone != model.backbone_id:
            raise ValueError(
                f"Resume backbone mismatch: checkpoint backbone_id={ckpt_backbone!r} "
                f"vs current {model.backbone_id!r}."
            )
        cfg = resume_payload.get("config") or {}
        ckpt_unfreeze = cfg.get("unfreeze_layers")
        if ckpt_unfreeze is not None and int(ckpt_unfreeze) != int(unfreeze_layers):
            raise ValueError(
                f"Resume config mismatch: checkpoint unfreeze_layers={ckpt_unfreeze} "
                f"vs current {unfreeze_layers}."
            )
        best_eer = float(resume_payload.get("best_eer", 101.0))
        history = [dict(h) for h in (resume_payload.get("history") or [])]
        global_steps = int(resume_payload.get("global_steps", 0))
        model.load_state_dict(resume_payload["model"])
        resume_stage = str(resume_payload.get("stage") or "")
        resume_next_epoch = int(resume_payload.get("next_epoch", 1))
        if resume_stage not in {"stage1_head", "stage2_finetune"}:
            raise ValueError(f"Resume checkpoint has unknown stage {resume_stage!r}.")
        print(
            f"[resume] stage={resume_stage} next_epoch={resume_next_epoch} "
            f"global_steps={global_steps} best_eer={best_eer:.2f}",
            flush=True,
        )

    def _ckpt_info(stage_name: str, ep: int, next_epoch: int, partial: bool, best_updated: bool) -> Dict:
        return {
            "stage": stage_name,
            "epoch": ep,
            "next_epoch": next_epoch,
            "partial_epoch": partial,
            "global_steps": global_steps,
            "best_eer": best_eer,
            "best_updated": best_updated,
        }

    ckpt_upload_failures = 0

    def _save_last_ckpt(
        stage_name: str,
        ep: int,
        next_epoch: int,
        partial: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        best_updated: bool = False,
    ) -> bool:
        """Write last.pt (weights_only-safe primitives + tensors) + train_history.json.

        Returns True when the checkpoint_callback succeeded (or none is set).
        Two consecutive callback failures raise RuntimeError so a lost box can
        never silently lose progress.
        """
        nonlocal ckpt_upload_failures
        payload = {
            "version": 1,
            "stage": stage_name,
            "next_epoch": int(next_epoch),
            "partial_epoch": bool(partial),
            "global_steps": int(global_steps),
            "best_eer": float(best_eer),
            "history": history,
            "backbone_id": model.backbone_id,
            "config": {
                "unfreeze_layers": int(unfreeze_layers),
                "batch_size": int(batch_size),
                "window_samples": int(DEFAULT_WINDOW_SAMPLES),
            },
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        torch.save(payload, last_ckpt)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        files: List[Path] = [last_ckpt, history_path]
        if best_updated and best_ckpt.exists():
            files.insert(1, best_ckpt)
        info = _ckpt_info(stage_name, ep, next_epoch, partial, best_updated)
        if checkpoint_callback is None:
            return True
        try:
            checkpoint_callback(files, info)
        except Exception as exc:
            ckpt_upload_failures += 1
            print(
                f"[warn] checkpoint upload failed ({ckpt_upload_failures}/2 consecutive): {exc}",
                flush=True,
            )
            if ckpt_upload_failures >= 2:
                raise RuntimeError(
                    "checkpoint upload failed twice in a row — stopping so a lost box "
                    "can't silently lose progress; last.pt is still on local disk, "
                    "rerun to resume"
                )
            return False
        ckpt_upload_failures = 0
        return True

    def run_stage(
        epochs: int,
        lr: float,
        stage_name: str,
        start_epoch: int = 1,
        resume_state: Dict | None = None,
        later_epochs: int = 0,
    ) -> None:
        nonlocal best_eer, global_steps
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            scaler.load_state_dict(resume_state["scaler"])

        stage_steps = 0
        eta_printed = False
        for ep in range(start_epoch, epochs + 1):
            if max_steps is not None and global_steps >= max_steps:
                break
            # Deterministic per-epoch torch seed so resumes replay the same order.
            torch.manual_seed(SEED_BASE + len(history) + 1)
            model.train()
            losses = []
            t_ep = time.perf_counter()
            steps = 0
            for x, y in train_loader:
                if max_steps is not None and global_steps >= max_steps:
                    break
                x = x.to(device, non_blocking=device.type == "cuda")
                y = y.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits, _ = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.item()))
                steps += 1
                stage_steps += 1
                global_steps += 1

                if deadline_unix is not None and time.time() > deadline_unix:
                    uploaded_ok = _save_last_ckpt(
                        stage_name, ep, next_epoch=ep, partial=True,
                        optimizer=optimizer, scheduler=scheduler, best_updated=False,
                    )
                    raise BudgetExceeded(
                        stage_name, ep, global_steps, time.time() - train_t0, uploaded=uploaded_ok
                    )

                if stage_steps == eta_after_steps and not eta_printed:
                    eta_printed = True
                    ep_elapsed = max(time.perf_counter() - t_ep, 1e-6)
                    sps = steps / ep_elapsed
                    remaining = (
                        (epochs - ep) * steps_per_epoch
                        + max(0, steps_per_epoch - steps)
                        + later_epochs * steps_per_epoch
                    )
                    if max_steps is not None:
                        remaining = min(remaining, max(0, max_steps - global_steps))
                    eta_sec = remaining / max(sps, 1e-6)
                    print(
                        f"[eta] stage={stage_name} steps_per_sec={sps:.2f} "
                        f"steps_per_epoch={steps_per_epoch} remaining_steps={int(remaining)} "
                        f"train_eta_min={eta_sec / 60.0:.1f} "
                        "(training only; excludes per-epoch validation; "
                        "stage2 is slower than stage1 once backbone layers unfreeze)",
                        flush=True,
                    )
                    if deadline_unix is not None and time.time() + eta_sec > deadline_unix:
                        over_min = (time.time() + eta_sec - deadline_unix) / 60.0
                        print(
                            f"[eta] WARNING: projected training ETA exceeds --deadline-unix "
                            f"by ~{over_min:.1f} min — the budget guard will stop training early.",
                            flush=True,
                        )
                    rate_env = os.environ.get("VAST_RATE_PER_HR", "").strip()
                    rate_val: float | None = None
                    if rate_env:
                        try:
                            rate_val = float(rate_env)
                            print(
                                f"[eta] projected training cost ~${(eta_sec / 3600.0) * rate_val:.2f} "
                                f"at ${rate_val:.2f}/hr",
                                flush=True,
                            )
                        except ValueError:
                            rate_val = None
                    cost_str = f"${(eta_sec / 3600.0) * rate_val:.2f}" if rate_val is not None else "NA"
                    key_line = (
                        f"[KEY] eta stage={stage_name} steps_per_s={sps:.2f} "
                        f"train_eta_min={eta_sec / 60.0:.1f} projected_cost={cost_str}"
                    )
                    print(key_line, flush=True)
                    key_file = os.environ.get("CB_KEY_LINES_FILE", "").strip()
                    if key_file:
                        try:
                            key_path = Path(key_file)
                            key_path.parent.mkdir(parents=True, exist_ok=True)
                            with key_path.open("a", encoding="utf-8") as fh:
                                fh.write(key_line + "\n")
                        except OSError as exc:
                            print(f"[warn] could not append KEY line to {key_file}: {exc}", flush=True)
            if steps == 0:
                break
            scheduler.step()
            ep_sec = max(time.perf_counter() - t_ep, 1e-6)
            steps_per_sec = steps / ep_sec

            model.eval()
            val_labels: List[int] = []
            val_scores: List[float] = []
            with torch.inference_mode():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=device.type == "cuda")
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        logits, _ = model(x)
                    probs = torch.sigmoid(logits).float().cpu().numpy().tolist()
                    val_scores.extend(probs)
                    val_labels.extend([int(v) for v in y.numpy().tolist()])

            metrics = compute_eer_and_auroc(val_labels, val_scores)
            entry = {
                "stage": stage_name,
                "epoch": ep,
                "steps": steps,
                "global_steps": global_steps,
                "train_loss": round(float(np.mean(losses)), 4),
                "steps_per_sec": round(steps_per_sec, 2),
                "epoch_sec": round(ep_sec, 1),
                "num_workers": workers,
                **metrics,
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)

            eer_val = metrics["eer_pct"]
            best_updated = False
            if not best_ckpt.exists() or np.isnan(eer_val) or eer_val <= best_eer:
                best_eer = eer_val if not np.isnan(eer_val) else best_eer
                torch.save(
                    {"backbone_id": model.backbone_id, "state_dict": model.state_dict(), "val_metrics": metrics},
                    best_ckpt,
                )
                best_updated = True

            _save_last_ckpt(
                stage_name, ep, next_epoch=ep + 1, partial=False,
                optimizer=optimizer, scheduler=scheduler, best_updated=best_updated,
            )

    # ---- stage orchestration (fresh or resumed) ----
    stage1_start = 1
    stage1_resume: Dict | None = None
    run_stage1 = epochs_stage1 > 0
    stage2_start = 1
    stage2_resume: Dict | None = None
    run_stage2 = epochs_stage2 > 0

    if resume_payload is not None:
        if resume_stage == "stage1_head":
            if resume_next_epoch <= epochs_stage1:
                stage1_start = resume_next_epoch
                stage1_resume = {
                    "optimizer": resume_payload["optimizer"],
                    "scheduler": resume_payload["scheduler"],
                    "scaler": resume_payload["scaler"],
                }
            else:
                run_stage1 = False  # stage1 already finished -> straight to stage2
        else:  # stage2_finetune
            run_stage1 = False
            if resume_next_epoch <= epochs_stage2:
                stage2_start = resume_next_epoch
                stage2_resume = {
                    "optimizer": resume_payload["optimizer"],
                    "scheduler": resume_payload["scheduler"],
                    "scaler": resume_payload["scaler"],
                }
            else:
                run_stage2 = False  # training already complete -> export only

    model.freeze_backbone()
    if run_stage1:
        run_stage(
            epochs_stage1, lr_stage1, "stage1_head",
            start_epoch=stage1_start, resume_state=stage1_resume, later_epochs=epochs_stage2,
        )

    model.unfreeze_top_layers(num_layers=unfreeze_layers)
    if run_stage2:
        run_stage(
            epochs_stage2, lr_stage2, "stage2_finetune",
            start_epoch=stage2_start, resume_state=stage2_resume, later_epochs=0,
        )

    if not best_ckpt.exists():
        raise FileNotFoundError(
            f"No trained checkpoint at {best_ckpt} — cannot export ONNX "
            "(epochs may be 0 or every epoch was skipped by max_steps)."
        )
    fp32_onnx, int8_onnx = export_from_checkpoint(
        best_ckpt, out_dir, backbone_id=backbone_id, use_tiny_test_backbone=use_tiny_test_backbone
    )

    summary = {
        "best_ckpt": str(best_ckpt),
        "fp32_onnx": str(fp32_onnx),
        "int8_onnx": str(int8_onnx),
        "best_val_eer_pct": best_eer,
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CloakBreak WavLM deepfake detector")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/run1"))
    parser.add_argument("--backbone-id", type=str, default=DEFAULT_BACKBONE_ID)
    parser.add_argument("--epochs-stage1", type=int, default=2)
    parser.add_argument("--epochs-stage2", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--unfreeze-layers", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tiny-test", action="store_true")
    parser.add_argument("--no-precompute-opus", action="store_true")
    parser.add_argument("--resume-from", type=str, default="none",
                        help="'none' | 'auto' (out_dir/last.pt then HF) | path to last.pt")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HF model repo for per-epoch checkpoint uploads (needs HF_TOKEN)")
    parser.add_argument("--hf-prefix", type=str, default="checkpoints",
                        help="Path prefix inside the HF model repo for checkpoints")
    parser.add_argument("--deadline-unix", type=float, default=None,
                        help="Unix timestamp; training stops cleanly (exit 3) past it")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training; export ONNX from out_dir/cloakbreak_best.pt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_only:
        best = out_dir / "cloakbreak_best.pt"
        fp32, int8 = export_from_checkpoint(
            best, out_dir, backbone_id=args.backbone_id, use_tiny_test_backbone=args.tiny_test
        )
        print(f"[export-only] fp32={fp32} int8={int8}", flush=True)
        return

    if args.manifest is None:
        raise SystemExit("--manifest is required unless --export-only is used.")

    hf_repo = args.hf_repo
    token = os.environ.get("HF_TOKEN", "").strip() or None
    checkpoint_callback = None
    if hf_repo:
        if not token:
            raise SystemExit("--hf-repo requires HF_TOKEN to be set.")
        from src import hub

        hub.ensure_model_repo(hf_repo, token=token)
        hf_prefix = args.hf_prefix.strip("/")

        def checkpoint_callback(files: List[Path], info: Dict) -> None:
            msg = (
                f"CloakBreak ckpt {info.get('stage')} ep={info.get('epoch')} "
                f"next={info.get('next_epoch')} gs={info.get('global_steps')}"
            )
            hub.upload_files(hf_repo, files, f"{hf_prefix}/", token=token, commit_message=msg)

    resume_path = resolve_resume_checkpoint(
        args.resume_from, out_dir, hf_repo=hf_repo, hf_prefix=args.hf_prefix, token=token
    )

    try:
        train_model(
            manifest_path=args.manifest,
            out_dir=out_dir,
            backbone_id=args.backbone_id,
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2,
            batch_size=args.batch_size,
            unfreeze_layers=args.unfreeze_layers,
            use_tiny_test_backbone=args.tiny_test,
            precompute_opus=not args.no_precompute_opus,
            num_workers=args.num_workers,
            max_steps=args.max_steps,
            device_str=args.device,
            resume_from=resume_path,
            deadline_unix=args.deadline_unix,
            checkpoint_callback=checkpoint_callback,
        )
    except BudgetExceeded as exc:
        if exc.uploaded:
            uploaded = f" and uploaded to {hf_repo}" if hf_repo else ""
            print(
                f"[budget] MAX_HOURS deadline reached at stage={exc.stage} epoch={exc.epoch} "
                f"global_steps={exc.global_steps} — checkpoint saved{uploaded}. "
                "Rerun the same command to resume (--resume-from auto).",
                flush=True,
            )
        else:
            print(
                f"[budget] MAX_HOURS deadline reached at stage={exc.stage} epoch={exc.epoch} "
                f"global_steps={exc.global_steps} — checkpoint saved locally but "
                "upload FAILED — last.pt is only on local disk; do not destroy the box "
                "until it is uploaded. Rerun the same command to resume (--resume-from auto).",
                flush=True,
            )
        sys.exit(3)


if __name__ == "__main__":
    main()
