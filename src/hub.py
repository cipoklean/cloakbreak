"""Hugging Face Hub helpers for CloakBreak model/artifact uploads (Session 2+).

Used by ``src.train`` (per-epoch checkpoint uploads + resume downloads) and by
the ``python -m src.hub upload-artifacts`` CLI called from session scripts.
The HF token is read from ``HF_TOKEN`` and never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError, RepositoryNotFoundError

MODEL_REPO_FILES = ("cloakbreak_fp32.onnx", "cloakbreak_int8.onnx")
MODEL_REPO_CKPTS = ("cloakbreak_best.pt", "train_summary.json", "train_history.json")


def _env_token() -> str | None:
    tok = os.environ.get("HF_TOKEN", "").strip()
    return tok or None


def ensure_model_repo(repo_id: str, token: str | None = None) -> None:
    """Create the private model repo if needed (idempotent)."""
    api = HfApi(token=token if token is not None else _env_token())
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    print(f"[hub] model repo ready: {repo_id}", flush=True)


def upload_files(
    repo_id: str,
    files: Iterable[Path],
    path_prefix: str,
    token: str | None = None,
    commit_message: str = "CloakBreak artifacts",
) -> None:
    """Upload ``files`` into ``repo_id`` under ``path_prefix`` in one commit."""
    paths = [Path(f) for f in files]
    if not paths:
        print("[hub] upload_files: nothing to upload", flush=True)
        return
    prefix = path_prefix.strip("/")
    prefix = f"{prefix}/" if prefix else ""
    api = HfApi(token=token if token is not None else _env_token())
    operations = [
        CommitOperationAdd(path_in_repo=f"{prefix}{p.name}", path_or_fileobj=str(p))
        for p in paths
    ]
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=commit_message,
    )
    print(
        f"[hub] uploaded {len(paths)} file(s) -> {repo_id}:{prefix} ({commit_message})",
        flush=True,
    )


def download_checkpoint(
    repo_id: str,
    filename: str,
    local_dir: str | Path,
    token: str | None = None,
) -> Path | None:
    """Download ``filename`` from the model repo into ``local_dir``.

    Returns the local path, or None when the file/repo does not exist.
    """
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            local_dir=str(local_dir),
            token=token if token is not None else _env_token(),
        )
        return Path(path)
    except (EntryNotFoundError, RemoteEntryNotFoundError, RepositoryNotFoundError):
        return None


def _plan_upload_artifacts(args: argparse.Namespace) -> List[str]:
    """Return human-readable ``local -> repo:path`` plan lines."""
    prefix = args.prefix.strip("/")
    prefix = f"{prefix}/" if prefix else ""
    lines: List[str] = []

    ckpt_dir = Path(args.ckpt_dir)
    for name in MODEL_REPO_FILES:
        p = ckpt_dir / name
        if p.exists():
            lines.append(f"{p} -> {prefix}onnx/{p.name}")
    for name in MODEL_REPO_CKPTS:
        p = ckpt_dir / name
        if p.exists():
            lines.append(f"{p} -> {prefix}checkpoints/{p.name}")

    results_dir = Path(args.results_dir)
    if results_dir.exists():
        n = sum(1 for f in results_dir.rglob("*") if f.is_file())
        lines.append(f"{results_dir}/ ({n} files) -> {prefix}results/")
    log_dir = Path(args.log_dir)
    if log_dir.exists():
        for p in sorted(log_dir.glob("*.log")):
            lines.append(f"{p} -> {prefix}logs/{p.name}")
    return lines


def _run_upload_artifacts(args: argparse.Namespace) -> int:
    plan = _plan_upload_artifacts(args)
    prefix = args.prefix.strip("/")
    prefix = f"{prefix}/" if prefix else ""

    if args.dry_run:
        print(f"[hub] DRY RUN upload plan -> {args.repo} (prefix '{prefix}'):", flush=True)
        if plan:
            for line in plan:
                print(f"  {line}", flush=True)
        else:
            print("  (nothing to upload)", flush=True)
        return 0

    ensure_model_repo(args.repo)
    ckpt_dir = Path(args.ckpt_dir)

    onnx_files = [ckpt_dir / n for n in MODEL_REPO_FILES if (ckpt_dir / n).exists()]
    if onnx_files:
        upload_files(args.repo, onnx_files, f"{prefix}onnx/", commit_message="CloakBreak ONNX exports")

    ckpt_files = [ckpt_dir / n for n in MODEL_REPO_CKPTS if (ckpt_dir / n).exists()]
    if ckpt_files:
        upload_files(args.repo, ckpt_files, f"{prefix}checkpoints/", commit_message="CloakBreak checkpoints")

    results_dir = Path(args.results_dir)
    if results_dir.exists() and any(f.is_file() for f in results_dir.rglob("*")):
        api = HfApi(token=_env_token())
        api.upload_folder(
            folder_path=str(results_dir),
            repo_id=args.repo,
            repo_type="model",
            path_in_repo=f"{prefix}results",
            allow_patterns=["*"],
        )
        print(f"[hub] uploaded {results_dir}/ -> {args.repo}:{prefix}results/", flush=True)

    log_dir = Path(args.log_dir)
    logs = sorted(log_dir.glob("*.log")) if log_dir.exists() else []
    if logs:
        upload_files(args.repo, logs, f"{prefix}logs/", commit_message="CloakBreak session logs")

    if not plan:
        print("[hub] nothing to upload", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="CloakBreak HF Hub artifact helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload-artifacts", help="Upload ONNX/checkpoints/results/logs to the model repo")
    up.add_argument("--repo", type=str, required=True, help="HF model repo id")
    up.add_argument("--ckpt-dir", type=Path, required=True)
    up.add_argument("--results-dir", type=Path, required=True)
    up.add_argument("--log-dir", type=Path, required=True)
    up.add_argument("--prefix", type=str, default="", help="Repo path prefix, e.g. 'smoke_check/' or ''")
    up.add_argument("--dry-run", action="store_true", help="Print the upload plan without uploading")

    args = parser.parse_args()
    if args.command == "upload-artifacts":
        sys.exit(_run_upload_artifacts(args))


if __name__ == "__main__":
    main()
