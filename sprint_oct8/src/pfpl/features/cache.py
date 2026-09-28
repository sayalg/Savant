"""Content-hashed shard cache (PLAN.md Sections 0.5.3, 7.1).

Two requirements from the HPC section drive this:

* **idempotent, resumable job arrays.** A wall-time kill must not cost the whole
  array. Each shard writes to `<name>.<shard>.h5` and a finished shard is skipped on
  re-submission, so re-running the array is always safe and always cheap.
* **large shards, not small files.** A parallel filesystem handles 20 files of 2 GB
  far better than 300,000 files of 100 kB, and node-local scratch staging only
  works if there is something big to stage.

The cache key is a content hash of (what was computed, from which inputs, with
which config), so changing the PLM, the pooling or the protein set produces a new
key rather than a silently stale tensor. The previous project lost a full benchmark
run to a shard-count mismatch that nothing detected; the manifest here records
`n_shards` and `merge` refuses to run if a shard is missing.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..paths import CACHE


def content_hash(*parts, length: int = 12) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:length]


def shard_range(n_items: int, shard_id: int, n_shards: int) -> np.ndarray:
    """Contiguous slices: neighbouring sequences have similar lengths after sorting,
    so contiguous shards pad less than strided ones."""
    if not 0 <= shard_id < n_shards:
        raise SystemExit(f"shard {shard_id} outside 0..{n_shards - 1}")
    bounds = np.linspace(0, n_items, n_shards + 1).astype(int)
    return np.arange(bounds[shard_id], bounds[shard_id + 1])


def slurm_shard() -> tuple[int, int]:
    """(shard_id, n_shards) from Slurm, with the trap that already cost this project.

    `sbatch --array=9` sets SLURM_ARRAY_TASK_COUNT to 1, not 10, so re-running a
    single shard of a 20-way array silently re-shards the whole input. That is why
    this returns the count explicitly and every caller prints it.
    """
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    cnt = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if tid is None:
        return 0, 1
    n = int(cnt) if cnt else 1
    i = int(tid)
    if n == 1 and i != 0:
        raise SystemExit(
            f"SLURM_ARRAY_TASK_ID={i} but SLURM_ARRAY_TASK_COUNT=1.\n"
            "  `sbatch --array=9` sets the count to 1, so this task would re-shard the\n"
            "  whole input into one piece and write it as shard 9. Pass the real\n"
            "  geometry explicitly:  --shard-id 9 --n-shards 20")
    return i, n


class ShardCache:
    """One logical array, written by N tasks, merged once."""

    def __init__(self, name: str, key: str, root: Path | None = None):
        self.name, self.key = name, key
        self.dir = Path(root or CACHE) / f"{name}_{key}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def shard_path(self, shard_id: int) -> Path:
        return self.dir / f"shard_{shard_id:04d}.npz"

    @property
    def merged_path(self) -> Path:
        return self.dir / "merged.npz"

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def done(self, shard_id: int) -> bool:
        p = self.shard_path(shard_id)
        return p.exists() and p.stat().st_size > 0

    def write_shard(self, shard_id: int, **arrays) -> Path:
        p = self.shard_path(shard_id)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        ## Atomic rename: a job killed mid-write leaves a .tmp, never a half shard
        ## that `done()` would accept.
        tmp.replace(p)
        return p

    def merge(self, n_shards: int, keys: list[str], meta: dict | None = None) -> Path:
        missing = [i for i in range(n_shards) if not self.done(i)]
        if missing:
            raise SystemExit(
                f"{len(missing)} of {n_shards} shards missing for {self.name} "
                f"(first: {missing[:8]}).\n"
                f"  Re-submit just those:  sbatch --array={','.join(map(str, missing[:20]))} "
                f"...\n  A merge with holes would silently produce a shorter array and "
                "every downstream index would be wrong.")
        out = {k: [] for k in keys}
        order = []
        for i in range(n_shards):
            z = np.load(self.shard_path(i), allow_pickle=False)
            order.append(z["index"])
            for k in keys:
                out[k].append(z[k])
        idx = np.concatenate(order)
        if len(np.unique(idx)) != len(idx):
            raise SystemExit(f"{self.name}: shards overlap ({len(idx) - len(np.unique(idx))} "
                             "duplicate indices) - the sharding geometry changed between runs")
        srt = np.argsort(idx)
        merged = {k: np.concatenate(v)[srt] for k, v in out.items()}
        merged["index"] = idx[srt]
        np.savez(self.merged_path, **merged)
        self.manifest_path.write_text(json.dumps(
            {"name": self.name, "key": self.key, "n_shards": n_shards,
             "n_items": int(len(idx)),
             "shapes": {k: list(v.shape) for k, v in merged.items()},
             **(meta or {})}, indent=1))
        print(f"  merged {n_shards} shards -> {self.merged_path} "
              f"({self.merged_path.stat().st_size / 1e9:.2f} GB)")
        return self.merged_path

    def load(self, mmap: bool = True):
        if not self.merged_path.exists():
            raise SystemExit(f"{self.merged_path} missing - the merge step has not run")
        return np.load(self.merged_path, mmap_mode="r" if mmap else None)


def verify_sample(cache: ShardCache, recompute, n: int = 8, tol: float = 1e-4,
                  key: str = "value") -> bool:
    """Phase 4 gate: re-extract a random sample and check it matches the cache.

    Deterministic re-extraction is the only thing that distinguishes "cached" from
    "cached from a config I have since changed".
    """
    z = cache.load(mmap=False)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(z["index"]), size=min(n, len(z["index"])), replace=False)
    ok = True
    for i in pick:
        got = np.asarray(recompute(int(z["index"][i])), np.float32)
        want = np.asarray(z[key][i], np.float32)
        if got.shape != want.shape or not np.allclose(got, want, atol=tol):
            print(f"  MISMATCH at index {int(z['index'][i])}: "
                  f"max|d|={np.abs(got - want).max() if got.shape == want.shape else 'shape'}")
            ok = False
    print(f"  cache verification on {len(pick)} samples: {'PASS' if ok else 'FAIL'}")
    return ok
