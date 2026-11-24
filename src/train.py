from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.audio_utils import TARGET_SR, load_wav_mono, pad_or_crop
from src.augment import TrainCodecAugmenter
from src.eval import compute_eer_and_auroc
from src.model import (
    DEFAULT_BACKBONE_ID,
    DEFAULT_WINDOW_SAMPLES,
    CloakBreakDetector,
    export_to_onnx,
    quantize_onnx_int8,
)


class SpoofAudioDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
        augmenter: TrainCodecAugmenter | None = None,
        random_crop: bool = False,
    ) -> None:
        self.rows = rows
        self.window_samples = window_samples
        self.augmenter = augmenter
        self.random_crop = random_crop
        self.rng = np.random.default_rng(42)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        wav, sr = load_wav_mono(row["path"], target_sr=TARGET_SR)
        wav = pad_or_crop(wav, self.window_samples, random_crop=self.random_crop, rng=self.rng)
        if self.augmenter is not None:
            wav = self.augmenter(wav, sr=sr)
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
    allow_opus: bool = True,
    device_str: str = "cuda",
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows = [r for r in rows if r.get("split") == "train" and "kokoro" not in r.get("generator", "")]
    val_rows = [r for r in rows if r.get("split") == "val" and "kokoro" not in r.get("generator", "")]
    if not train_rows or not val_rows:
        mid = max(1, int(len(rows) * 0.8))
        train_rows, val_rows = rows[:mid], rows[mid:]

    train_spks = {r.get("speaker_id") for r in train_rows}
    val_spks = {r.get("speaker_id") for r in val_rows}
    overlap = train_spks & val_spks
    if overlap and len(rows) > 10:
        print(f"[warn] {len(overlap)} overlapping speakers between train/val")

    n_pos = sum(1 for r in train_rows if int(r["label"]) == 1)
    n_neg = max(1, len(train_rows) - n_pos)
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)

    augmenter = TrainCodecAugmenter(allow_opus=allow_opus, seed=42)
    train_ds = SpoofAudioDataset(train_rows, augmenter=augmenter, random_crop=True)
    val_ds = SpoofAudioDataset(val_rows, augmenter=None, random_crop=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CloakBreakDetector(backbone_id=backbone_id, use_tiny_test_backbone=use_tiny_test_backbone).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_eer = 101.0
    best_ckpt = out_dir / "cloakbreak_best.pt"
    history = []

    def run_stage(epochs: int, lr: float, stage_name: str) -> None:
        nonlocal best_eer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        for ep in range(1, epochs + 1):
            model.train()
            losses = []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
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
            scheduler.step()

            model.eval()
            val_labels: List[int] = []
            val_scores: List[float] = []
            with torch.inference_mode():
                for x, y in val_loader:
                    x = x.to(device)
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        logits, _ = model(x)
                    probs = torch.sigmoid(logits).float().cpu().numpy().tolist()
                    val_scores.extend(probs)
                    val_labels.extend([int(v) for v in y.numpy().tolist()])

            metrics = compute_eer_and_auroc(val_labels, val_scores)
            entry = {
                "stage": stage_name,
                "epoch": ep,
                "train_loss": round(float(np.mean(losses)), 4),
                **metrics,
            }
            history.append(entry)
            print(json.dumps(entry))

            eer_val = metrics["eer_pct"]
            if np.isnan(eer_val) or eer_val <= best_eer:
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tiny-test", action="store_true")
    parser.add_argument("--no-opus", action="store_true")
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
        allow_opus=not args.no_opus,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
