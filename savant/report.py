"""The output documents Savant writes: JSON, CSV, or both.

JSON shape, which does not change with the number of targets or ligands:

    {
      "savant":  {...what produced this...},
      "targets": [
        {
          "id": "P02754",
          "sequence": "MTVPDRSEI...",
          "length": 212,
          "source": "target.fasta",
          "n_scored": 3,
          "n_reported": 3,
          "predictions": [
            {"rank": 1, "smiles": "...", "name": null,
             "pKd": 7.41, "Kd_nM": 38.9, "is_binder": true},
            ...
          ]
        }
      ]
    }

`targets` is always a list, even for the single target that is the usual case,
so a consumer needs one code path rather than two.

The protein sequence appears **once per target** and never once per prediction.
A 1,000-residue sequence repeated across a 100,000-ligand screen would be 100 MB
of the same string. The CSV follows the same rule: it carries `target_id` to
join on and no sequence column at all.

Predictions are ordered by rank, best first.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

## Column order of the CSV. `p_binder` is appended only when the bundle carries a
## classifier, so the shipped regressor-only models do not emit an all-empty
## column that looks like a failed prediction.
CSV_COLUMNS = ["target_id", "rank", "smiles", "name", "pKd", "Kd_nM", "is_binder"]


def _py(value):
    """numpy scalar -> plain Python, and NaN -> None so the JSON is valid.

    json.dump would otherwise emit bare NaN, which is not JSON and which many
    parsers reject.
    """
    if value is None:
        return None
    ## bool before int: Python's bool IS an int subclass, so checking int first
    ## turns True into 1 and the JSON gets 1/0 where it should have true/false.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if np.isnan(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _target_label(target, index):
    """A stable id for joining CSV rows back to a target.

    Falls back to a positional label so the column is never empty: a screen
    against an unnamed sequence still has something to group by.
    """
    return target.id or f"target_{index}"


def build_report(savant, target_results, precision=4, include_sequence=True,
                 top=None):
    """Assemble the JSON document.

    `target_results` is [(Target, scored DataFrame)], where each frame is what
    :meth:`savant.model.Savant.screen` returns: smiles, pKd, Kd_nM, rank,
    is_binder, p_binder.

    `top` keeps only the N best-ranked predictions per target. Truncation happens
    here rather than in the caller so that `n_scored` can still report how many
    ligands were actually run -- a file holding the top 10 of a 100,000-ligand
    screen should not claim the screen was 10 ligands.
    """
    targets = []
    for i, (target, scored) in enumerate(target_results, 1):
        n_scored = len(scored)
        if top:
            scored = scored.head(top)

        predictions = []
        for _, row in scored.iterrows():
            rec = {
                "rank": _py(row["rank"]),
                "smiles": row["smiles"],
                "name": row.get("name") or None,
                "pKd": round(float(row["pKd"]), precision),
                "Kd_nM": round(float(row["Kd_nM"]), precision),
                "is_binder": _py(row["is_binder"]),
            }
            if savant.classifier is not None:
                rec["p_binder"] = round(float(row["p_binder"]), precision)
            predictions.append(rec)

        entry = {
            "id": _target_label(target, i),
            "length": target.length,
            "source": target.source,
            "n_scored": n_scored,
            "n_reported": len(predictions),
        }
        if include_sequence:
            entry["sequence"] = target.sequence
        entry["predictions"] = predictions
        targets.append(entry)

    return {
        "savant": {
            "version": _version(),
            "model": savant.model,
            "protein_encoder": savant.esm_model,
            "ligand_encoder": "DeepChem/ChemBERTa-77M-MLM",
            "n_features": savant.n_features_in_,
            "prediction": "pKd = -log10(Kd[M]); Kd_nM = 10^(9 - pKd)",
            "binder_call": (
                "p_binder >= 0.5" if savant.classifier is not None
                else f"pKd > {savant.binder_pkd}"
            ),
            "n_targets": len(targets),
            "n_scored": sum(t["n_scored"] for t in targets),
            "n_reported": sum(t["n_reported"] for t in targets),
            "top": top,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "targets": targets,
    }


def build_table(savant, target_results, precision=4, top=None):
    """One flat DataFrame: a row per (target, ligand), ordered by rank.

    No sequence column -- see the module docstring. `target_id` is what joins a
    row back to its protein, and the JSON (or the input FASTA) holds the sequence.
    """
    columns = list(CSV_COLUMNS)
    if savant.classifier is not None:
        columns.append("p_binder")

    frames = []
    for i, (target, scored) in enumerate(target_results, 1):
        if top:
            scored = scored.head(top)
        part = scored.copy()
        part["target_id"] = _target_label(target, i)
        part["pKd"] = part["pKd"].astype(float).round(precision)
        part["Kd_nM"] = part["Kd_nM"].astype(float).round(precision)
        part["is_binder"] = part["is_binder"].astype(bool)
        if "p_binder" in columns:
            part["p_binder"] = part["p_binder"].astype(float).round(precision)
        if "name" not in part.columns:
            part["name"] = None
        frames.append(part[columns])

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _version():
    from savant import __version__
    return __version__


def write_report(report, path=None, indent=2):
    """Write the JSON to `path`, or to stdout when `path` is None or '-'.

    Returns the path written, or None for stdout. Writes via a temporary file in
    the same directory and replaces atomically, so an interrupted run leaves the
    previous output intact rather than a half-written file that looks complete.
    """
    text = json.dumps(report, indent=indent, ensure_ascii=False, allow_nan=False)

    if path is None or str(path) == "-":
        sys.stdout.write(text + "\n")
        return None
    return _atomic_write(Path(path), text + "\n")


def write_table(table, path=None):
    """Write the CSV to `path`, or to stdout when `path` is None or '-'."""
    text = table.to_csv(index=False, lineterminator="\n")

    if path is None or str(path) == "-":
        sys.stdout.write(text)
        return None
    return _atomic_write(Path(path), text)


def _atomic_write(out: Path, text: str) -> Path:
    if str(out.parent) not in ("", "."):
        out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    return out
