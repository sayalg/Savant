"""The compute ledger and the preregistration check (PLAN.md Sections 7.1, 7.4).

Twelve concurrent GPUs make it easy to overfit a validation set by sheer volume:
run three hundred configurations, report the best, and a dPearson of 0.02 found
after three hundred trials is weak evidence dressed as a result. Section 7.4's
mitigations are mechanical, so they are implemented mechanically here.

* every run appends one row to `reports/compute_ledger.csv` - config hash, split,
  seed, GPU-hours, and the metric it reported. The final report states the total
  count, so the multiplicity is visible rather than inferred.
* `check_preregistered()` compares a run's config against
  `configs/preregistration.yaml` and marks it `exploratory` when it falls outside
  the declared matrix. Exploratory runs are not forbidden - they are labelled, and
  a headline claim that rests on one has to say so.
"""
from __future__ import annotations

import csv
import os
import socket
import time
from pathlib import Path

from .paths import REPORTS

LEDGER = REPORTS / "compute_ledger.csv"
FIELDS = ["timestamp", "run_name", "config_hash", "stage", "variant", "plm_mode",
          "split", "seed", "terms", "gpu_hours", "n_params", "host", "job_id",
          "preregistered", "select_metric", "select_value", "test_pearson",
          "test_rho", "notes"]


def gpu_hours(seconds: float, n_gpu: int = 1) -> float:
    return round(seconds / 3600.0 * max(n_gpu, 1), 4)


def record(cfg, *, stage: str, elapsed_s: float, n_gpu: int = 1, **kw) -> dict:
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_name": kw.pop("run_name", cfg.get("run_name", "")),
        "config_hash": cfg.get("hash", ""),
        "stage": stage,
        "variant": cfg.get_path("model.variant", ""),
        "plm_mode": cfg.get_path("plm.mode", ""),
        "split": cfg.get_path("data.split", ""),
        "seed": cfg.get("seed", ""),
        "terms": ",".join(cfg.get_path("model.energy.terms", []) or []),
        "gpu_hours": gpu_hours(elapsed_s, n_gpu),
        "host": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "preregistered": kw.pop("preregistered", ""),
    }
    row |= {k: v for k, v in kw.items() if k in FIELDS}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def summary() -> dict:
    if not LEDGER.exists():
        return {"n_runs": 0, "gpu_hours": 0.0}
    rows = list(csv.DictReader(open(LEDGER)))
    gh = sum(float(r.get("gpu_hours") or 0) for r in rows)
    return {
        "n_runs": len(rows),
        "gpu_hours_total": round(gh, 2),
        "n_preregistered": sum(r.get("preregistered") == "True" for r in rows),
        "n_exploratory": sum(r.get("preregistered") == "False" for r in rows),
        "by_variant": _count(rows, "variant"),
        "by_plm_mode": _count(rows, "plm_mode"),
        "by_split": _count(rows, "split"),
        "multiplicity_note": (
            f"{len(rows)} configurations were trained. A dPearson of 0.02 selected "
            "post hoc from this many trials is weak evidence; Section 7.4 asks for a "
            "multiplicity-aware reading."),
    }


def _count(rows, key) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[r.get(key, "")] = out.get(r.get(key, ""), 0) + 1
    return out


def check_preregistered(cfg, prereg: dict | None = None) -> tuple[bool, str]:
    """Is this configuration inside the declared ablation matrix?"""
    if prereg is None:
        from .conf import load
        try:
            prereg = load("preregistration")
        except SystemExit:
            return False, "configs/preregistration.yaml missing"
    matrix = prereg.get("matrix", {}) or {}
    checks = {
        "model.variant": cfg.get_path("model.variant"),
        "plm.mode": cfg.get_path("plm.mode"),
        "data.split": cfg.get_path("data.split"),
        "model.energy.vdw_form": cfg.get_path("model.energy.vdw_form"),
    }
    outside = [f"{k}={v}" for k, v in checks.items()
               if k in matrix and v not in matrix[k]]
    seeds = prereg.get("seeds")
    if seeds and cfg.get("seed") not in seeds:
        outside.append(f"seed={cfg.get('seed')}")
    if outside:
        return False, "outside the pre-registered matrix: " + ", ".join(outside)
    cap = int(prereg.get("trial_cap", 0) or 0)
    if cap and summary().get("n_runs", 0) >= cap:
        return False, f"trial cap of {cap} reached; this run is exploratory"
    return True, "pre-registered"
