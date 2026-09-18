"""Load a TabFM estimator from a LOCAL weights directory, fully offline.

The stock loading paths (`tabfm...load()` and `TabFM_HF.from_pretrained`) both
default to reaching Hugging Face. Here we always point them at a directory that
`download_weights.py` already populated, and we set `local_files_only=True`, so
nothing touches the network on a compute node.

The public TabFM API has moved a little between releases, so `load_backbone`
tries the documented entrypoints in order and raises a clear error if none fit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .offline import resolve_device


def _subfolder_for(task: str) -> str:
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    return task


def load_backbone(task: str, weights_dir: str, *, device: str | None = None,
                  dtype: str = "bfloat16") -> Any:
    """Load the pretrained TabFM backbone (not yet fitted) from local files.

    weights_dir is the directory passed to download_weights.py --dest, i.e. the
    one that contains the `classification/` and/or `regression/` subfolders.
    """
    import torch

    sub = _subfolder_for(task)
    root = Path(weights_dir).expanduser().resolve()
    ckpt = root / sub
    if not (ckpt / "model.safetensors").exists() and not (ckpt / "config.json").exists():
        raise FileNotFoundError(
            f"No weights found under {ckpt}. Did you run:\n"
            f"  python download_weights.py --task {task} --dest {root}")

    dev = resolve_device(device)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32, "none": None}.get(dtype, torch.bfloat16)

    errors: list[str] = []

    # Path 1: the library's own loader with an explicit local checkpoint.
    try:
        from tabfm import tabfm_v1_0_0_pytorch as backend  # type: ignore
        model = backend.load(model_type=task, checkpoint_path=str(ckpt),
                             device=dev, dtype=torch_dtype)
        return model
    except Exception as e:  # noqa: BLE001 - fall through to the HF mixin path
        errors.append(f"tabfm_v1_0_0_pytorch.load(checkpoint_path=...): {e!r}")

    # Path 2: the HF mixin, pointed at the local dir with the network disabled.
    try:
        from tabfm.src.pytorch.tabfm_v1_0_0 import TabFM_HF  # type: ignore
        model = TabFM_HF.from_pretrained(
            str(root), subfolder=sub, local_files_only=True,
            map_location=dev,
        )
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        model = model.to(dev)
        return model
    except Exception as e:  # noqa: BLE001
        errors.append(f"TabFM_HF.from_pretrained(local): {e!r}")

    raise RuntimeError(
        "Could not load TabFM from local weights. Tried:\n  - "
        + "\n  - ".join(errors)
        + "\nCheck that `pip install tabfm[pytorch]` succeeded and that the "
          "weights directory layout matches what download_weights.py produced.")


def build_estimator(task: str, weights_dir: str, *, device: str | None = None,
                    dtype: str = "bfloat16", **estimator_kwargs: Any) -> Any:
    """Return a fit-ready sklearn-style TabFM estimator wrapping the backbone."""
    from tabfm import TabFMClassifier, TabFMRegressor  # type: ignore

    model = load_backbone(task, weights_dir, device=device, dtype=dtype)
    if task == "classification":
        return TabFMClassifier(model=model, **estimator_kwargs)
    return TabFMRegressor(model=model, **estimator_kwargs)
