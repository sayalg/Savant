#!/usr/bin/env python3
"""Phase 10: score the SEALED benchmark. Once per model version.

PLAN.md Section 0.4.3: never tune, early-stop, or select models on the sealed
benchmark. This script enforces that mechanically - it records every scoring event
in `reports/benchmark_log.json` keyed by (model version, config hash), and refuses
to score the same version twice without `--rescore`, which is itself logged.

It also does the two things a Boltz-2 comparison needs to be fair (Section 9):

* leads with RANK metrics, because Boltz-2's affinity output is a log10 IC50-like
  quantity in uM, not a Kd-calibrated pK. Pearson and RMSE for Boltz-2 are reported
  only after a linear recalibration fit on rows DISJOINT from the test items;
* reports the low-overlap subset separately, since the benchmark's overlap with
  Boltz-2's own training sources (ChEMBL, BindingDB, PubChem, MF-PCBA) is the
  single biggest confound in the comparison.

    python scripts/phase10_benchmark.py --run runs/A_bottleneck_protein_cluster_...
    python scripts/phase10_benchmark.py --run <dir> --boltz2 work/boltz2_scores.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

## Imported FIRST, before torch / matplotlib / transformers: importing
## pfpl.paths redirects every HOME-bound cache and TMPDIR under the project
## root, and some of those libraries read their cache variable at import time.
from pfpl import paths as _paths  # noqa: E402,F401


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pfpl import conf, paths, preflight  # noqa: E402
from pfpl.eval import metrics as M  # noqa: E402

LOG = paths.REPORTS / "benchmark_log.json"


def already_scored(version: str) -> list:
    if not LOG.exists():
        return []
    return [e for e in json.loads(LOG.read_text()) if e["version"] == version]


def record(entry: dict) -> None:
    hist = json.loads(LOG.read_text()) if LOG.exists() else []
    hist.append(entry)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(hist, indent=1, default=str))


def recalibrate(scores: np.ndarray, y: np.ndarray, fit_mask: np.ndarray):
    """Linear recalibration fit on rows disjoint from the ones being scored.

    Boltz-2 emits a log10 IC50-like value in uM. Reporting its Pearson on a pK
    target without recalibration makes it look worse than it is; fitting the
    recalibration on the SAME rows being scored makes it look better than it is.
    Both are wrong, and Section 9 asks for the disjoint version.
    """
    if fit_mask.sum() < 10:
        return None, "too few disjoint rows to fit a recalibration"
    a, b = np.polyfit(scores[fit_mask], y[fit_mask], 1)
    return (lambda s: a * np.asarray(s, float) + b), f"pK ~ {a:+.3f}*score {b:+.3f}"


def overlap_flags(bench: pd.DataFrame) -> pd.Series:
    """Rows whose protein or ligand appears in our own corpus, as an overlap proxy.

    Section 5.3.4 wants the benchmark checked against Boltz-2's reported training
    sources (ChEMBL, BindingDB, PubChem, MF-PCBA). Our harmonised corpus is built
    from the first two, so corpus membership is a usable lower bound on that overlap
    - and it is the part we can actually measure. It is a proxy and is labelled one.
    """
    p = paths.WORK / "pairs.parquet"
    if not p.exists():
        return pd.Series(False, index=bench.index)
    corpus = pd.read_parquet(p, columns=["sequence_hash", "inchikey_flat"])
    seqs = set(corpus.sequence_hash.astype(str))
    ligs = set(corpus.inchikey_flat.astype(str))
    return (bench.sequence_hash.astype(str).isin(seqs)
            & bench.inchikey_flat.astype(str).isin(ligs))


def main() -> None:
    ap = argparse.ArgumentParser()
    conf.add_cfg_args(ap)
    ap.add_argument("--run", required=True, help="a run directory with best.pt")
    ap.add_argument("--boltz2", default=None,
                    help="CSV of Boltz-2 scores. Omit it when the benchmark already "
                         "carries them in a column (this project's does).")
    ## NOT "pred": that is OUR model's column in the scored frame, and defaulting to
    ## it would silently compare the model against itself and report dPearson 0.
    ap.add_argument("--boltz2-col", default="boltz_affinity_pic50",
                    help="the Boltz-2 score column, in --boltz2 or in the benchmark")
    ap.add_argument("--rescore", action="store_true",
                    help="score a version that has been scored before; logged as such")
    a = ap.parse_args()

    run = Path(a.run)
    if not (run / "best.pt").exists():
        raise SystemExit(f"{run}/best.pt not found")
    info = json.loads((run / "run.json").read_text())
    version = f"{info['name']}@{info['config_hash']}"

    prev = already_scored(version)
    if prev and not a.rescore:
        raise SystemExit(
            f"model version {version} has already been scored on the sealed benchmark "
            f"({len(prev)} time(s), first on {prev[0]['timestamp']}).\n\n"
            "PLAN.md Section 0.4.3: the benchmark is scored ONCE per model version.\n"
            "Scoring it repeatedly and keeping the best result is selection on the "
            "test set,\nwhich is the thing the seal exists to prevent.\n\n"
            "If a bug genuinely invalidated the first scoring, pass --rescore; the "
            "event is\nlogged and must be disclosed in the final report.")

    bench_p = paths.WORK / "benchmark_standardized.csv"
    if not bench_p.exists():
        raise SystemExit(f"{bench_p} missing - run scripts/phase0_verify.py --benchmark")
    bench = pd.read_csv(bench_p)
    scoreable = bench.get("scoreable", pd.Series(True, index=bench.index)).astype(bool)
    if "pK" not in bench.columns:
        raise SystemExit("the benchmark has no pK column, so it cannot be scored")
    n_excluded = int((~scoreable).sum())
    bench = bench[scoreable & bench.pK.notna()].reset_index(drop=True)
    print(f"  benchmark: {len(bench):,} scoreable rows"
          + (f" ({n_excluded:,} excluded as non-Kd/Ki)" if n_excluded else ""))

    ## ---- our model -----------------------------------------------------------
    import torch
    from pfpl.data.datasets import (ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
                                    PairDataset, ProteinStore, make_loader)
    from pfpl.features.embed_esm import POCKET_EXTRA_DIM
    from pfpl.models.affinity import PFPL
    from pfpl.train.loop import Trainer

    cfg = conf.Cfg(json.loads((run / "config.json").read_text()))
    prot = (bench.drop_duplicates("sequence_hash")[["sequence_hash", "sequence"]]
            .reset_index(drop=True))
    prot["protein_idx"] = np.arange(len(prot))
    bench = bench.merge(prot[["sequence_hash", "protein_idx"]], on="sequence_hash")
    bench["pK"] = bench.pK.astype(float)
    bench["weight"] = 1.0
    bench["relation"] = bench.get("relation", "=")

    store = None
    if str(cfg.plm.mode) != "none":
        ## Benchmark proteins are not in the training protein table, so their
        ## embeddings are extracted here rather than read from the cache.
        from pfpl.features.embed_esm import Embedder, pocket_extra_features
        preflight.warn_if_cpu("benchmark embedding extraction")
        emb = Embedder(str(cfg.plm.mode), want_contacts=True)
        d = paths.WORK / f"benchmark_plm_{cfg.plm.mode}"
        d.mkdir(parents=True, exist_ok=True)
        vecs, extras, lengths = [], [], []
        for s in prot.sequence:
            r = emb.encode([s])[0]
            vecs.append(r["emb"])
            extras.append(pocket_extra_features(r.get("contact_centrality"),
                                                s[:len(r["emb"])]).astype(np.float16))
            lengths.append(len(r["emb"]))
        off = np.zeros(len(prot) + 1, np.int64)
        off[1:] = np.cumsum(lengths)
        np.memmap(d / "emb.f16", dtype=np.float16, mode="w+",
                  shape=(int(off[-1]), emb.dim))[:] = np.concatenate(vecs)
        np.memmap(d / "pocket_extra.f16", dtype=np.float16, mode="w+",
                  shape=(int(off[-1]), POCKET_EXTRA_DIM))[:] = np.concatenate(extras)
        np.save(d / "offsets.npy", off)
        np.save(d / "lengths.npy", np.asarray(lengths, np.int64))
        (d / "meta.json").write_text(json.dumps({"dim": emb.dim,
                                                 "n_proteins": len(prot)}))
        store = ProteinStore(d)

    model = PFPL(cfg, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
                 pocket_extra_dim=POCKET_EXTRA_DIM)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    trainer = Trainer(model, cfg, run / "benchmark", stage="affinity")
    bench = bench.rename(columns={"pK": "pK"})
    ds = PairDataset(bench, prot, store, int(cfg.data.get("max_sequence_length", 1022)),
                     int(cfg.data.get("max_atoms", 64)),
                     need_plm=str(cfg.plm.mode) != "none")
    t0 = time.time()
    preds = trainer.predict(make_loader(ds, 16, False, 2))
    infer_s = (time.time() - t0) / max(len(preds), 1)

    ## ---- assemble -------------------------------------------------------------
    df = bench.iloc[preds.pair_idx.to_numpy()].copy().reset_index(drop=True)
    df["pred"] = preds.pred.to_numpy()
    df["logvar"] = preds.logvar.to_numpy()
    df["y"] = df.pK.to_numpy()
    df["in_corpus"] = overlap_flags(df).to_numpy()
    low = ~df.in_corpus
    print(f"  overlap proxy: {int(df.in_corpus.sum()):,} of {len(df):,} rows share both "
          f"protein and ligand with our corpus; low-overlap subset n={int(low.sum()):,}")

    reports = {"model": M.evaluate(df, f"pfpl {info['name']}", logvar="logvar")}
    if low.sum() >= 20:
        reports["model_low_overlap"] = M.evaluate(
            df[low], "pfpl (low-overlap subset)", logvar="logvar",
            note="rows whose protein AND ligand are both absent from our corpus")

    ## ---- Boltz-2 --------------------------------------------------------------
    extra: dict = {"inference_seconds_per_complex": round(infer_s, 4),
                   "note_cost": "no folding step; Section 9 asks for this next to the "
                                "pose-based pipelines' cost including folding"}
    if a.boltz2 or a.boltz2_col in df.columns:
        ## This project's benchmark carries its Boltz-2 scores in the SAME file
        ## (bench_scored.csv has boltz_affinity_pic50), and phase0 preserves every
        ## original column, so the usual case needs no join at all.
        if not a.boltz2 and a.boltz2_col in df.columns:
            j = df.rename(columns={a.boltz2_col: "boltz"})
            j = j[j.boltz.notna()]
            key = f"(in-file column {a.boltz2_col!r})"
            print(f"  Boltz-2: {len(j):,} rows from the benchmark's own "
                  f"{a.boltz2_col!r} column")
        else:
            b = pd.read_csv(a.boltz2)
            key = next((k for k in ("inchikey_full", "inchikey", "benchmark_id",
                                    "smiles", "ligand_smiles", "ligand_smiles_std")
                        if k in b.columns and k in df.columns), None)
            if key is None:
                print("  cannot join Boltz-2 scores: no shared key between "
                      f"{list(b.columns)[:8]}... and the benchmark")
                j = None
            else:
                j = df.merge(
                    b[[key, a.boltz2_col]].rename(columns={a.boltz2_col: "boltz"}),
                    on=key, how="inner")
                print(f"  Boltz-2: joined {len(j):,} rows on {key!r}")
        if j is not None and len(j):
            reports["boltz2_rank"] = M.evaluate(
                j.assign(pred=j.boltz), "Boltz-2 (rank metrics only)",
                scale_free=True,
                note="log10 IC50-like output in uM, NOT a Kd-calibrated pK; Pearson "
                     "and RMSE are meaningless without recalibration")
            ## Recalibrate on a disjoint half, score the other.
            rng = np.random.default_rng(0)
            fit = rng.random(len(j)) < 0.5
            cal, desc = recalibrate(j.boltz.to_numpy(), j.y.to_numpy(), fit)
            if cal is not None:
                jj = j[~fit].copy()
                jj["pred"] = cal(jj.boltz.to_numpy())
                reports["boltz2_recalibrated"] = M.evaluate(
                    jj, "Boltz-2 (recalibrated, disjoint fit)",
                    note=f"linear recalibration fit on the other half: {desc}")
                ours = j[~fit].copy()
                d = M.paired_bootstrap_delta(ours.y.to_numpy(), ours.pred.to_numpy(),
                                             cal(ours.boltz.to_numpy()))
                extra["paired_vs_boltz2"] = d
                print(f"  paired dPearson (us - Boltz-2) on {int((~fit).sum()):,} rows: "
                      f"{d['delta']:+.4f} CI [{d['lo']:+.4f},{d['hi']:+.4f}]")
            extra["boltz2_join_key"] = key
            extra["boltz2_n"] = int(len(j))

    M.print_table(reports, f"SEALED BENCHMARK - {version}")
    outd = run / "benchmark"
    outd.mkdir(parents=True, exist_ok=True)
    df.to_csv(outd / "predictions.csv", index=False)
    M.write(reports, outd / "reports.json", extra=extra)

    record({"version": version, "run": str(run), "timestamp": time.strftime("%FT%T"),
            "rescore": bool(a.rescore), "n_rows": int(len(df)),
            "pearson": reports["model"].pearson,
            "per_target_rho": reports["model"].per_target_spearman_median,
            "low_overlap_n": int(low.sum()), **extra})
    print(f"\n  logged to {LOG}. This version is now SEALED - scoring it again needs "
          "--rescore and must be disclosed.")


if __name__ == "__main__":
    main()
