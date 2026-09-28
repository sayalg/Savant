#!/usr/bin/env python3
"""Phase 1: affinity harvesting and harmonisation (PLAN.md Section 8, Phase 1).

One loader per source into the Section 5.1 schema, then cross-source dedupe,
ligand standardisation, sequence resolution, quarantine, and aggregation to one
pK per (protein, ligand) pair.

Runs on the LOGIN NODE or in a job - it is pandas and RDKit, no GPU. Outputs:

    work/measurements.parquet   every surviving measurement, schema.COLUMNS
    work/pairs.parquet          one row per (protein, ligand): the training table
    work/proteins.parquet       unique sequences, indexed by protein_idx
    work/proteins.fasta         for mmseqs (headers are p<protein_idx>)
    reports/dropped_types.csv   Section 5.2: counts only, never values
    reports/phase_1.md          the numbers, next to Section 4.4's expectations

Gate: dedupe verified on hand-checked pairs, no benchmark leakage,
`tests/test_labels.py` passes.

    python scripts/phase1_harmonize.py
    python scripts/phase1_harmonize.py -o data.sources.papyrus.enabled=true
    python scripts/phase1_harmonize.py --inspect-papyrus
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

from pfpl import conf, paths  # noqa: E402
from pfpl.data import harmonize, quarantine, schema, sources  # noqa: E402
from pfpl.data.standardize import (clean_sequence, is_peptide_like,  # noqa: E402
                                   sequence_hash, standardize_frame)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def resolve(p: str | None) -> str | None:
    """Relative config paths resolve under $PFPL_WORK (default <root>/work).

    Absolute paths are honoured as given - that is the one place a staged read-only
    dataset (a licensed PDBbind tree, a shared BindingDB dump) may legitimately live
    outside the project root.
    """
    if not p:
        return p
    q = Path(p)
    return str(q if q.is_absolute() else paths.WORK / q)


def load_sources(cfg) -> tuple[pd.DataFrame, dict]:
    frames, per_source, skipped = [], {}, {}
    for name, spec in (cfg.data.sources or {}).items():
        if not spec.get("enabled", False):
            print(f"  - {name}: disabled")
            continue
        loader = sources.get(spec.get("loader", name))
        kw = {k: v for k, v in spec.items()
              if k not in ("enabled", "loader", "optional")}
        if "path" in kw:
            kw["path"] = resolve(kw["path"])
        if spec.get("loader") == "local_csv":
            kw["name"] = name
        log(f"loading {name}")
        try:
            df = loader(**kw)
        except SystemExit as e:
            ## An OPTIONAL source that is absent is a legitimate state, not a bug:
            ## PDBbind is licensed and cannot be fetched, so the corpus build must
            ## survive without it. Every other enabled source that fails IS an
            ## error - it means acquisition did not finish - and still aborts.
            if not spec.get("optional", False):
                raise
            skipped[name] = str(e).strip().splitlines()[0]
            print(f"\n  *** SKIPPING OPTIONAL SOURCE {name!r} ***")
            for ln in str(e).splitlines():
                print(f"      {ln}")
            print()
            continue
        frames.append(df)
        per_source[name] = {
            "rows": int(len(df)),
            "types": df["type"].value_counts().astype(int).to_dict(),
            "relations": df["relation"].value_counts().astype(int).to_dict(),
        }
    if not frames:
        raise SystemExit(
            "no data source produced any rows. Acquire them first:\n"
            "    python scripts/phase1a_acquire.py --all\n"
            f"Available loaders: {sources.available()}")
    return pd.concat(frames, ignore_index=True), per_source, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    conf.add_cfg_args(ap)
    ap.add_argument("--inspect-papyrus", action="store_true",
                    help="print a Papyrus release's real schema and exit")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap the row count")
    ap.add_argument("--skip-quarantine", action="store_true",
                    help="DEBUG ONLY. Never use for a run whose numbers you will quote.")
    a = ap.parse_args()

    if a.inspect_papyrus:
        from pfpl.data.sources.papyrus import inspect
        inspect()
        return

    cfg = conf.from_args(a)
    paths.ensure()
    print("=" * 78)
    print("PHASE 1 - affinity harvesting and harmonisation")
    print("=" * 78)

    raw, per_source, skipped = load_sources(cfg)
    if a.limit:
        raw = raw.sample(min(a.limit, len(raw)), random_state=0).reset_index(drop=True)
    log(f"{len(raw):,} measurements from {len(per_source)} source(s)")
    schema.assert_label_policy(raw, "concatenated sources")

    ## ---- standardisation ---------------------------------------------------
    log("standardising ligands and sequences")
    raw = standardize_frame(raw)
    drop_reasons = pd.concat([
        raw.loc[raw._ligand_status != "ok", "_ligand_status"].value_counts(),
        raw.loc[raw._seq_status != "ok", "_seq_status"].value_counts()])
    before = len(raw)
    raw = raw[(raw._ligand_status == "ok") & (raw._seq_status == "ok")].copy()
    log(f"standardisation kept {len(raw):,} / {before:,}")

    if cfg.data.get("drop_peptides", True):
        uniq = raw.drop_duplicates("inchikey_full")[["inchikey_full", "ligand_smiles_std"]]
        pep = {k for k, s in zip(uniq.inchikey_full, uniq.ligand_smiles_std)
               if is_peptide_like(s)}
        n = int(raw.inchikey_full.isin(pep).sum())
        raw = raw[~raw.inchikey_full.isin(pep)]
        drop_reasons["peptide_like"] = n
        log(f"dropped {n:,} peptide-like rows -> {len(raw):,}")

    maxlen = int(cfg.data.get("max_sequence_length", 1022))
    too_long = int((raw.sequence.str.len() > maxlen).sum())
    if too_long:
        ## Section 5.2: crop rather than drop, and REPORT the count so the paper can
        ## state it. Metrics are stratified by length bin in Phase 6.
        log(f"{too_long:,} rows have sequences > {maxlen} aa; they will be cropped "
            f"to the PLM window and reported by length bin")

    ## ---- dedupe and cross-type ---------------------------------------------
    log("cross-source dedupe")
    dedup = harmonize.dedupe_measurements(raw)
    cross = harmonize.cross_type_table(dedup)

    ## ---- quarantine --------------------------------------------------------
    qpath = ROOT / cfg.quarantine.path
    if a.skip_quarantine:
        print("\n  *** QUARANTINE SKIPPED. Any number from this corpus is invalid for")
        print("      reporting. Section 0.4.3 / 5.3.1. ***\n")
    elif qpath.exists():
        q = quarantine.load(qpath)
        mode = str(cfg.quarantine.get("mode", "strict"))
        log(f"applying {mode} benchmark quarantine")
        if mode == "strict":
            bench = pd.read_csv(paths.WORK / "benchmark_standardized.csv")
            drop = quarantine.apply_strict(
                dedup, bench, q["keys"],
                identity=float(cfg.quarantine.get("identity_threshold", 0.30)),
                tanimoto=float(cfg.quarantine.get("tanimoto_threshold", 0.90)))
        else:
            drop = quarantine.apply_exact(dedup, q["keys"])
        dedup = dedup[~drop]
        drop_reasons["quarantined"] = int(drop.sum())
    else:
        raise SystemExit(
            f"{qpath} missing. Quarantine keys must exist BEFORE the corpus is built,\n"
            "or the training set may contain the benchmark - which is exactly how the\n"
            "previous project's headline result came to be measured on a protein that\n"
            "was in its own training fold.\n\n"
            "    python scripts/phase0_verify.py --benchmark <file>\n\n"
            "For a throwaway debugging corpus only: --skip-quarantine")

    ## ---- aggregate ----------------------------------------------------------
    log("aggregating to pairs")
    pairs = harmonize.aggregate_pairs(
        dedup, label_types=tuple(cfg.label.label_types))
    ceiling = harmonize.noise_ceiling(pairs)
    print(f"  noise ceiling from our own replicates: {json.dumps(ceiling)}")

    ## ---- protein table ------------------------------------------------------
    prot = (pairs.drop_duplicates("sequence_hash")[["sequence_hash", "sequence", "uniprot"]]
            .reset_index(drop=True))
    prot["protein_idx"] = np.arange(len(prot))
    prot["length"] = prot.sequence.str.len()
    pairs = pairs.merge(prot[["sequence_hash", "protein_idx"]], on="sequence_hash")
    lig = pairs.drop_duplicates("inchikey_full")[["inchikey_full"]].reset_index(drop=True)
    lig["ligand_idx"] = np.arange(len(lig))
    pairs = pairs.merge(lig, on="inchikey_full")

    minlig = int(cfg.data.get("min_ligands_per_target", 1))
    if minlig > 1:
        n = pairs.groupby("protein_idx").size()
        keep = pairs.protein_idx.map(n) >= minlig
        log(f"min_ligands_per_target={minlig}: {len(pairs):,} -> {int(keep.sum()):,}")
        pairs = pairs[keep]

    ## ---- write --------------------------------------------------------------
    paths.WORK.mkdir(parents=True, exist_ok=True)
    dedup.to_parquet(paths.WORK / "measurements.parquet", index=False)
    pairs.reset_index(drop=True).to_parquet(paths.WORK / "pairs.parquet", index=False)
    prot.to_parquet(paths.WORK / "proteins.parquet", index=False)
    if len(cross):
        cross.to_csv(paths.REPORTS / "ki_vs_kd_same_pair.csv", index=False)
    with open(paths.WORK / "proteins.fasta", "w") as fh:
        for i, s in zip(prot.protein_idx, prot.sequence):
            fh.write(f">p{i}\n{s}\n")
    pd.DataFrame({"reason": drop_reasons.index, "n": drop_reasons.to_numpy()}).to_csv(
        paths.REPORTS / "dropped_types.csv", index=False)

    ## ---- report -------------------------------------------------------------
    summary = {
        "per_source": per_source,
        "measurements_after_dedupe": int(len(dedup)),
        "pairs": int(len(pairs)),
        "proteins": int(len(prot)),
        "ligands": int(pairs.inchikey_full.nunique()),
        "types": pairs["type"].value_counts().astype(int).to_dict(),
        "censored": int(pairs.censored.sum()),
        "sequences_over_window": too_long,
        "median_ligands_per_protein": float(pairs.groupby("protein_idx").size().median()),
        "max_ligands_per_protein": int(pairs.groupby("protein_idx").size().max()),
        "noise_ceiling": ceiling,
        "label_types": list(cfg.label.label_types),
        "quarantine_mode": (None if a.skip_quarantine
                            else str(cfg.quarantine.get("mode", "strict"))),
        "skipped_optional_sources": skipped,
        "has_crystal_source": "pdbbind" in per_source,
        "config_hash": cfg.hash,
    }
    (paths.WORK / "corpus_meta.json").write_text(json.dumps(summary, indent=1))
    _write_report(summary, drop_reasons, cross)

    if "pdbbind" in skipped:
        print("\n" + "!" * 78)
        print("PDBbind was NOT available, so this corpus has NO crystal-linked rows.")
        print("Three separate consequences, and the second is the one that matters:")
        print("  1. no source-weight-1.0 labels (the highest-quality measurements)")
        print("  2. NO DISTOGRAM ANCHOR. PLAN.md Section 2.1 lists it as a REQUIRED")
        print("     mitigation; without it the affinity loss can push probability mass")
        print("     toward each pair's energy minimum and converge on an")
        print("     over-attractive, geometrically impossible solution that scores")
        print("     well on pK. A physics claim from such a run is NOT supported.")
        print("  3. no Stage 3 crystal fine-tune")
        print("Wave 1 and every baseline are unaffected. Stage it and re-run Phase 1")
        print("before making any Wave 2 claim.")
        print("!" * 78)

    print("\n" + "=" * 78)
    print("GATE - Phase 1")
    ok_leak = True
    if not a.skip_quarantine:
        try:
            quarantine.assert_clean(pairs, qpath)
            print("  no benchmark pair in the corpus     : PASS")
        except AssertionError as e:
            ok_leak = False
            print(f"  benchmark leakage                   : FAIL\n    {e}")
    print(f"  label policy (Kd/Ki only)           : PASS "
          f"({summary['types']})")
    print(f"  corpus                              : {summary['pairs']:,} pairs, "
          f"{summary['proteins']:,} proteins, {summary['ligands']:,} ligands")
    print("  Section 4.4 expected 'several hundred thousand Kd/Ki pairs, "
          "Ki-dominated'.")
    print("  Compare the line above and record the answer in DATA_MANIFEST.md.")
    print("  Next:  python -m pytest tests/test_labels.py tests/test_harmonize.py")
    print("=" * 78)
    sys.exit(0 if ok_leak else 1)


def _write_report(summary: dict, drops: pd.Series, cross: pd.DataFrame) -> None:
    L = ["# Phase 1 - affinity harvesting and harmonisation", ""]
    if summary.get("skipped_optional_sources"):
        L += ["> **Optional sources skipped:** "
              + ", ".join(f"`{k}`" for k in summary["skipped_optional_sources"]),
              ">",
              "> Without `pdbbind` there are no crystal-linked labels, **no distogram "
              "anchor** (Section 2.1 lists it as required) and no Stage 3 fine-tune.", ""]
    L += [
         "Kd and Ki only. IC50/EC50 are excluded at every loader "
         "(PLAN.md Section 0.4 rule 1).", "", "## Corpus", "",
         f"- measurements after dedupe: **{summary['measurements_after_dedupe']:,}**",
         f"- unique (protein, ligand) pairs: **{summary['pairs']:,}**",
         f"- proteins: **{summary['proteins']:,}**  ligands: **{summary['ligands']:,}**",
         f"- types: {summary['types']}",
         f"- censored (>/<): {summary['censored']:,} (one-sided loss, Section 5.2)",
         f"- sequences over the {1022}-aa PLM window: {summary['sequences_over_window']:,} "
         "(cropped; metrics reported by length bin)",
         f"- ligands per protein: median {summary['median_ligands_per_protein']:.0f}, "
         f"max {summary['max_ligands_per_protein']:,}", "",
         "Section 4.4 expects several hundred thousand Kd/Ki pairs, Ki-dominated, "
         "after dedupe against ChEMBL. Compare.", "", "## Per source", ""]
    for k, v in summary["per_source"].items():
        L.append(f"- `{k}`: {v['rows']:,} rows, types {v['types']}, "
                 f"relations {v['relations']}")
    L += ["", "## Dropped", "",
          "| reason | n |", "|---|---|"]
    for r, n in drops.items():
        L.append(f"| {r} | {int(n):,} |")
    L += ["", "Counts only - Section 5.2 forbids carrying the dropped values forward.",
          "", "## Noise ceiling (measured on our own replicates)", "",
          "```json", json.dumps(summary["noise_ceiling"], indent=1), "```", "",
          "Appendix A asks for this to be quantified rather than borrowed. An RMSE "
          "below the implied sd is fitting noise.", ""]
    if len(cross):
        d = cross["delta"]
        L += ["## Ki vs Kd on identical pairs", "",
              f"- n = {len(cross):,}", f"- median Ki - Kd = {d.median():+.3f} pK",
              f"- |delta| > 1 log unit: {(d.abs() > 1).mean():.1%}", "",
              "This is the evidence for or against pooling Ki with Kd "
              "(`label_types: [Kd, Ki]`). The Kd-only ablation is in "
              "`configs/preregistration.yaml`.", ""]
    (paths.REPORTS / "phase_1.md").write_text("\n".join(L))
    print(f"  wrote {paths.REPORTS / 'phase_1.md'}")


if __name__ == "__main__":
    main()
