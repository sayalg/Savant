"""Minimal, dependency-light data loading for tabular CSV/Parquet files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def load_xy(path: str, target: str) -> tuple[pd.DataFrame, "pd.Series"]:
    """Split a labelled table into features X and target y."""
    df = _read(path)
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in {path}. "
                       f"Columns: {list(df.columns)}")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def load_x(path: str, target: str | None = None) -> pd.DataFrame:
    """Load features only; drop target column if present (test set may lack it)."""
    df = _read(path)
    if target and target in df.columns:
        df = df.drop(columns=[target])
    return df
