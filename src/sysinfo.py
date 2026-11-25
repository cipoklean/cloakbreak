"""Effective CPU count that respects cgroup CPU quotas.

Containers on quota-limited hosts (e.g. Vast) report the host's vCPU count via
``os.cpu_count()``/``nproc`` while the cgroup quota is far lower; sizing worker
pools off the raw count oversubscribes and can stall the pipeline. This takes
the minimum of affinity, cgroup v2 ``cpu.max``, cgroup v1 CFS quota, and
``os.cpu_count()``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _v2_quota_cpus(root: Path) -> int | None:
    raw = _read_text(root / "cpu.max")
    if not raw:
        return None
    parts = raw.split()
    if len(parts) < 2 or parts[0] == "max":
        return None
    try:
        quota, period = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return math.ceil(quota / period)


def _v1_quota_cpus(root: Path) -> int | None:
    for sub in ("cpu", "cpu,cpuacct"):
        quota_s = _read_text(root / sub / "cpu.cfs_quota_us")
        period_s = _read_text(root / sub / "cpu.cfs_period_us")
        if quota_s is None or period_s is None:
            continue
        try:
            quota, period = float(quota_s), float(period_s)
        except ValueError:
            continue
        if quota > 0 and period > 0:  # -1 means unlimited
            return math.ceil(quota / period)
    return None


def effective_cpu_count(cgroup_root: str = "/sys/fs/cgroup") -> int:
    """Smallest CPU ceiling visible to this process; never below 1."""
    candidates: list[int] = []

    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass

    root = Path(cgroup_root)
    for fn in (_v2_quota_cpus, _v1_quota_cpus):
        try:
            n = fn(root)
        except OSError:
            n = None
        if n is not None:
            candidates.append(n)

    n = os.cpu_count()
    if n:
        candidates.append(int(n))

    if not candidates:
        return 1
    return max(1, min(candidates))


if __name__ == "__main__":
    print(effective_cpu_count())
