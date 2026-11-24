from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

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
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")

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

    model = CloakBreakDetector(backbone_id=backbone_id, use_tiny_test_backbone=use_tiny_test_backbone).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_eer = 101.0
    best_ckpt = out_dir / "cloakbreak_best.pt"
    history = []
    global_steps = 0

    def run_stage(epochs: int, lr: float, stage_name: str) -> None:
        nonlocal best_eer, global_steps
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        for ep in range(1, epochs + 1):
            if max_steps is not None and global_steps >= max_steps:
                break
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
                global_steps += 1
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
            if not best_ckpt.exists() or np.isnan(eer_val) or eer_val <= best_eer:
                best_eer = eer_val if not np.isnan(eer_val) else best_eer
                torch.save(
                    {"backbone_id": model.backbone_id, "state_dict": model.state_dict(), "val_metrics": metrics},
                    best_ckpt,
                )

    model.freeze_backbone()
    if epochs_stage1 > 0:
        run_stage(epochs_stage1, lr_stage1, "stage1_head")

    model.unfreeze_top_layers(num_layers=unfreeze_layers)
    if epochs_stage2 > 0:
        run_stage(epochs_stage2, lr_stage2, "stage2_finetune")

    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to("cpu").eval()
    fp32_onnx = export_to_onnx(model, out_dir / "cloakbreak_fp32.onnx")
    int8_onnx = quantize_onnx_int8(fp32_onnx, out_dir / "cloakbreak_int8.onnx")

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
    parser.add_argument("--manifest", type=Path, required=True)
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
    args = parser.parse_args()

    train_model(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
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
    )


if __name__ == "__main__":
    main()
