"""One loader per affinity source, all emitting the Section 5.1 schema.

Every loader returns a DataFrame with the `schema.COLUMNS` columns and has already
applied the type filter, so a source that cannot be shown to be free of IC50 never
reaches `harmonize.py`. Registration is by name so `configs/data/*.yaml` can list
sources without importing anything.

Adding a source means: write `load(...) -> DataFrame`, call
`schema.conform(df, "<name>")` on the way out, and register it here.
"""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> Callable:
    if name not in _REGISTRY:
        _load_all()
    if name not in _REGISTRY:
        raise SystemExit(f"unknown data source {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    _load_all()
    return sorted(_REGISTRY)


def _load_all() -> None:
    from . import (bindingdb, chembl, iuphar, local_csv,  # noqa: F401
               papyrus, pdbbind)


def uniprot_sequences():
    """accession -> sequence, from the Phase 1a cache.

    ChEMBL's target endpoint and IUPHAR's interactions.csv both carry an accession
    and NO sequence, so both loaders resolve through this one cache rather than
    each reaching for the network.
    """
    from ...paths import RAW
    import pandas as pd
    p = RAW / "uniprot" / "sequences.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    return dict(zip(d.accession.astype(str), d.sequence.astype(str)))
