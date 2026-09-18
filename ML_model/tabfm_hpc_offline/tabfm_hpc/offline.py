"""Force fully-offline behaviour before any HF / torch import.

Import and call `go_offline()` at the very top of any entrypoint that runs on a
compute node. It flips the Hugging Face libraries into offline mode so a stray
`from_pretrained` can never try to reach the network and hang the job, and it
keeps every cache off the (small) home quota.
"""
from __future__ import annotations

import os
from pathlib import Path


def go_offline(cache_root: str | None = None) -> None:
    """Set env vars that disable all network access for HF + tokenizers.

    Parameters
    ----------
    cache_root : optional
        Directory for any HF cache/staging. Defaults to $HF_HOME if set, else a
        `.hf_home` next to the current working directory. Should live on
        scratch/project storage on HPC, never home.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # Avoid slow/failed symlink probes on parallel filesystems (Lustre/GPFS).
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    root = cache_root or os.environ.get("HF_HOME") or str(Path.cwd() / ".hf_home")
    os.environ["HF_HOME"] = root
    Path(root).mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str | None = None) -> str:
    """Pick a torch device string, honouring an explicit request."""
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
