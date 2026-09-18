#!/usr/bin/env python
"""LIT-PCBA: download, build, contamination-audit and score a real screening benchmark.

    python prep_litpcba.py download          # phase 1 - NEEDS NETWORK (login node)
    python prep_litpcba.py build             # parse .smi + attach UniProt sequences
    python prep_litpcba.py audit             # overlap against input.csv (the training corpus)
    python prep_litpcba.py score --bundle ../work_screening/savant_screening_*.pkl

Why this set
------------
The v3 benchmark is 20 targets x ~25 ligands and **88.6% binders**, so per-target
enrichment saturates near 1.13 and every tool lands between 0.97 and 1.14 - the metric
cannot separate them. LIT-PCBA is 15 targets, 7,761 confirmed actives and 382,674
confirmed inactives from 149 dose-response PubChem bioassays, with hit rates from 5e-5 to
0.05 (below 3e-3 for most targets). At a 0.5% hit rate EF@1% can reach 100, which is where
a screening claim is actually testable.

Its inactives are *measured* inactives, not property-matched decoys - which is the reason
to prefer it over DUD-E, whose decoy construction has documented bias that inflates ML
scores.

Two caveats to carry into the write-up
--------------------------------------
1. LIT-PCBA has its own documented issues: "Data Leakage and Redundancy in the LIT-PCBA
   Benchmark" (arXiv 2507.21404, 2025). Cite it rather than let a reviewer raise it -
   leakage is already a theme of this work, so engaging with it helps.
2. **Four of the 15 target sets test a phenotype this model cannot represent.** ESR_ago
   and ESR_antago are the same protein (ESR1, P03372) with opposite functional outcomes;
   ADRB2 and OPRK1 are agonist sets. A sequence+SMILES *affinity* model receives identical
   protein input for ESR_ago and ESR_antago and cannot distinguish agonism from
   antagonism by construction. Report those separately or exclude them - do not average
   them into a headline number and let a reviewer find it.

Layout after `download`:

    data/AVE_unbiased/<TARGET>/active.smi      "SMILES SID", no header
                              /inactive.smi
                              /*_T.smi, *_V.smi   the authors' own train/validation split
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"
CORPUS = HERE.parent.parent / "input" / "input.csv"

## MUST be https. The site advertises these as http://, but that 301-redirects to
## lab.drugdesign.unistra.fr, which 404s - so anything that follows redirects (urllib,
## curl -L, wget) lands on a dead page and reports "404 Not Found" for a file that is
## actually there. Straight to https on the original host returns 200.
##   AVE_unbiased.tgz  57,396,648 bytes
##   full_data.tgz     53,808,890 bytes
URLS = {"ave": "https://drugdesign.unistra.fr/LIT-PCBA/Files/AVE_unbiased.tgz",
        "full": "https://drugdesign.unistra.fr/LIT-PCBA/Files/full_data.tgz"}
BIOACTIVITIES = "https://drugdesign.unistra.fr/LIT-PCBA/Files/LIT-PCBA_bioactivities.xlsx"
GZIP_MAGIC = b"\x1f\x8b"

# Directory name in the archive -> (UniProt accession, human-readable, assay phenotype).
# Phenotype matters: "agonist"/"antagonist" sets are functional, not pure affinity.
TARGETS = {
    "ADRB2":      ("P07550", "Beta-2 adrenergic receptor",        "agonist"),
    "ALDH1":      ("P00352", "Aldehyde dehydrogenase 1A1",        "inhibitor"),
    "ESR1_ago":   ("P03372", "Estrogen receptor alpha",           "agonist"),
    "ESR1_ant":   ("P03372", "Estrogen receptor alpha",           "antagonist"),
    "FEN1":       ("P39748", "Flap endonuclease 1",               "inhibitor"),
    "GBA":        ("P04062", "Glucosylceramidase beta",           "inhibitor"),
    "IDH1":       ("O75874", "Isocitrate dehydrogenase 1",        "inhibitor"),
    "KAT2A":      ("Q92830", "Histone acetyltransferase KAT2A",   "inhibitor"),
    "MAPK1":      ("P28482", "MAP kinase 1 (ERK2)",               "inhibitor"),
    "MTORC1":     ("P42345", "mTOR",                              "inhibitor"),
    "OPRK1":      ("P41145", "Kappa opioid receptor",             "agonist"),
    "PKM2":       ("P14618", "Pyruvate kinase PKM",               "inhibitor"),
    "PPARG":      ("P37231", "PPAR gamma",                        "inhibitor"),
    "TP53":       ("P04637", "Cellular tumor antigen p53",        "inhibitor"),
    "VDR":        ("P11473", "Vitamin D receptor",                "inhibitor"),
}
FUNCTIONAL = {k for k, v in TARGETS.items() if v[2] != "inhibitor"}
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{}.fasta"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# phase 1 - network
# --------------------------------------------------------------------------- #

def _fetch(url: str, dest: Path, expect_gzip: bool = True):
    """wget -> curl -> urllib, whichever exists. Resumes a partial file where it can."""
    import shutil
    import subprocess

    if shutil.which("wget"):
        cmd = ["wget", "--continue", "--tries=3", "--timeout=60", "-O", str(dest), url]
    elif shutil.which("curl"):
        ## -C - resumes; no -L, because following the redirect is exactly what breaks this
        cmd = ["curl", "-C", "-", "--retry", "3", "--max-time", "1800",
               "-o", str(dest), url]
    else:
        cmd = None

    if cmd:
        log(f"  {cmd[0]} {url}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd[0]} exited {r.returncode}")
    else:
        log(f"  urllib {url}")
        urllib.request.urlretrieve(url, dest)

    ## A 404 HTML page saved as .tgz fails much later with a confusing tarfile error.
    ## Check the gzip magic bytes here, where the message can still be useful.
    if expect_gzip:
        if dest.stat().st_size < 1_000_000:
            head = dest.read_bytes()[:200]
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"got {len(head)} bytes, not an archive: {head[:120]!r}")
        with open(dest, "rb") as fh:
            if fh.read(2) != GZIP_MAGIC:
                dest.unlink(missing_ok=True)
                raise RuntimeError("downloaded file is not gzip - the URL returned HTML")


def cmd_download(args):
    DATA.mkdir(parents=True, exist_ok=True)
    url = URLS[args.variant]
    dest = DATA / Path(url).name
    if dest.exists() and not args.force:
        log(f"{dest.name} already present ({dest.stat().st_size/1e6:.0f} MB)")
    else:
        log(f"downloading {url}")
        try:
            _fetch(url, dest)
        except Exception as e:
            sys.exit(f"download failed: {type(e).__name__}: {e}\n"
                     f"Fetch it by hand and drop it in {DATA}:\n"
                     f"    wget {url}\n"
                     "Note the https:// - the http:// form the website prints redirects to "
                     "lab.drugdesign.unistra.fr and 404s.")
        log(f"  -> {dest} ({dest.stat().st_size/1e6:.0f} MB)")

    ## per-compound bioactivity table; small, and useful for provenance in the methods
    xl = DATA / Path(BIOACTIVITIES).name
    if not xl.exists():
        try:
            _fetch(BIOACTIVITIES, xl, expect_gzip=False)
        except Exception as e:
            log(f"  (optional) bioactivities table not fetched: {type(e).__name__}")

    ## The archive has NO wrapper directory - it drops the 15 target folders straight into
    ## the extraction root. Extracting into DATA directly would mix the ave and full
    ## variants, so each goes in its own subdirectory. _find_target_dirs uses rglob, so a
    ## wrapper level (if a future release adds one) is handled either way.
    target_dir = DATA / ("AVE_unbiased" if args.variant == "ave" else "full_data")
    if target_dir.exists() and any(target_dir.iterdir()) and not args.force:
        log(f"{target_dir.name}/ already extracted")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        log(f"extracting into {target_dir}")
        with tarfile.open(dest) as t:
            try:
                t.extractall(target_dir, filter="data")     # py>=3.12
            except TypeError:
                t.extractall(target_dir)
    dirs = sorted(_find_target_dirs(target_dir))
    log(f"target directories found: {len(dirs)}/{len(TARGETS)}")
    log(f"  {dirs}")

    # UniProt sequences
    seqs = {}
    for acc in sorted({v[0] for v in TARGETS.values()}):
        try:
            with urllib.request.urlopen(UNIPROT_FASTA.format(acc), timeout=60) as fh:
                txt = fh.read().decode()
            seq = "".join(l.strip() for l in txt.splitlines() if not l.startswith(">"))
            seqs[acc] = seq
            log(f"  {acc}: {len(seq)} residues")
        except Exception as e:
            log(f"  {acc}: FAILED {type(e).__name__}: {e}")
    pd.DataFrame({"uniprot_id": list(seqs), "sequence": list(seqs.values())}
                 ).to_csv(DATA / "uniprot_sequences.csv", index=False)
    log(f"wrote {DATA/'uniprot_sequences.csv'} ({len(seqs)} sequences)")
    log("NOTE: PKM2 is an isoform of PKM (P14618) and mTORC1 is a complex whose target "
        "protein is MTOR (P42345). Both use the canonical UniProt sequence - state that "
        "in the methods rather than leaving it implicit.")


# --------------------------------------------------------------------------- #
# phase 2 - offline
# --------------------------------------------------------------------------- #

def _read_smi(path: Path) -> pd.DataFrame:
    """LIT-PCBA .smi files are 'SMILES SID', whitespace separated, no header."""
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
            elif parts:
                rows.append((parts[0], ""))
    return pd.DataFrame(rows, columns=["smiles", "sid"])


def _norm(s: str) -> str:
    """Fold a directory name to letters only, uppercased, digits removed.

    The published table names these sets ESR_ago / ESR_antago / ALDH1 / PKM2, but the
    archive has been seen using ESR1_ago and similar. Dropping separators and digits makes
    ESR_ago, ESR1_ago and esr-ago all collapse to ESRAGO. Verified collision-free across
    the 15 keys: ADRB, ALDH, ESRAGO, ESRANTAGO, FEN, GBA, IDH, KATA, MAPK, MTORC, OPRK,
    PKM, PPARG, TP, VDR.
    """
    return "".join(c for c in s.upper() if c.isalpha())


## The AVE_unbiased release ships ONLY the authors' train/validation split -
## active_T.smi / active_V.smi / inactive_T.smi / inactive_V.smi - and no combined
## active.smi. T+V reproduces the published per-target counts exactly (ALDH1:
## 4020+1343 = 5,363 actives, 76577+25297 = 101,874 inactives), so both are read and the
## split is kept as a column rather than being thrown away.
SMI_VARIANTS = [("active.smi", 1, "all"), ("inactive.smi", 0, "all"),
                ("active_T.smi", 1, "T"), ("active_V.smi", 1, "V"),
                ("inactive_T.smi", 0, "T"), ("inactive_V.smi", 0, "V")]


def _has_ligand_files(p: Path) -> bool:
    return any((p / fn).exists() for fn, _, _ in SMI_VARIANTS)


## The published table and the archive disagree on the two ESR set names: the paper calls
## them ESR_ago / ESR_antago, the tarball ships ESR1_ago / ESR1_ant. Letter-folding alone
## does not bridge ESRANT vs ESRANTAGO, so the alternatives are listed explicitly. Keys
## follow the archive, since that is what has to be matched on disk.
ALIASES = {
    "ESR1_ago": ["ESR_ago", "ESR_agonist"],
    "ESR1_ant": ["ESR_ant", "ESR_antago", "ESR1_antago", "ESR_antagonist"],
}


def _find_target_dirs(root: Path) -> dict:
    """Map our TARGETS keys onto whatever the archive actually named its folders."""
    lookup = {_norm(k): k for k in TARGETS}
    for k, alts in ALIASES.items():
        for a in alts:
            lookup[_norm(a)] = k
    assert len(lookup) >= len(TARGETS), "normalised target names collide"
    found, unmapped = {}, []
    for p in [q for q in root.rglob("*") if q.is_dir() and _has_ligand_files(q)]:
        key = TARGETS.get(p.name) and p.name or lookup.get(_norm(p.name))
        if key is None:
            unmapped.append(p.name)
            continue
        found[key] = p
    if unmapped:
        log(f"  UNMAPPED directories (add to TARGETS to include): {sorted(unmapped)}")
    missing = sorted(set(TARGETS) - set(found))
    if missing:
        log(f"  targets with no directory found: {missing}")
    return found


def cmd_build(args):
    OUT.mkdir(parents=True, exist_ok=True)
    root = DATA / ("AVE_unbiased" if args.variant == "ave" else "full_data")
    if not root.exists():
        sys.exit(f"{root} not found - run `prep_litpcba.py download` first")
    seqs = pd.read_csv(DATA / "uniprot_sequences.csv").set_index("uniprot_id")["sequence"]

    dirs = _find_target_dirs(root)
    log(f"{len(dirs)} target directories mapped")
    frames = []
    for key, path in sorted(dirs.items()):
        acc, name, pheno = TARGETS[key]
        parts = []
        for fn, label, split in SMI_VARIANTS:
            f = path / fn
            if not f.exists():
                continue
            part = _read_smi(f)
            part["label"], part["split"] = label, split
            parts.append(part)
        if not parts:
            log(f"  {key}: no .smi files - skipped")
            continue
        d = pd.concat(parts, ignore_index=True)
        a = d[d.label == 1]; i = d[d.label == 0]
        d["target"] = key
        d["uniprot_id"] = acc
        d["target_name"] = name
        d["phenotype"] = pheno
        d["target_sequence"] = seqs.get(acc, "")
        if not d["target_sequence"].iloc[0]:
            log(f"  {key}: NO SEQUENCE for {acc} - dropped")
            continue
        frames.append(d)
        log(f"  {key:10s} {acc}  actives {len(a):>6,}  inactives {len(i):>8,}  "
            f"hit rate {len(a)/max(len(d),1):.2%}")

    all_df = pd.concat(frames, ignore_index=True)
    ## a compound can appear against several targets; the PAIR is the unit
    all_df = all_df.drop_duplicates(["target", "smiles"]).reset_index(drop=True)
    log(f"total {len(all_df):,} (target, ligand) pairs, "
        f"{all_df.smiles.nunique():,} unique ligands, {all_df.target.nunique()} targets")
    log(f"overall hit rate {all_df.label.mean():.4%}")

    fn = OUT / f"litpcba_pairs_{args.variant}.parquet"
    all_df.to_parquet(fn, index=False)
    log(f"wrote {fn}")

    summ = (all_df.groupby(["target", "uniprot_id", "phenotype"])
            .agg(n=("label", "size"), actives=("label", "sum")).reset_index())
    summ["hit_rate"] = summ.actives / summ.n
    ## The most EF@1% can be is min(1/hit_rate, 100) - every active inside the top 1%.
    ## Printing it stops anyone reading a low EF as failure when the set caps it.
    summ["ef_ceiling_at_1pct"] = np.minimum(1.0 / summ.hit_rate.clip(lower=1e-9), 100.0)
    summ.to_csv(OUT / f"litpcba_targets_{args.variant}.csv", index=False)
    print(summ.round(5).to_string(index=False))
    if FUNCTIONAL & set(all_df.target):
        log(f"REMINDER: {sorted(FUNCTIONAL & set(all_df.target))} are functional "
            "(agonist/antagonist) sets. ESR1_ago and ESR1_ant share one sequence, so an "
            "affinity model cannot separate them. Report separately or exclude.")


# --------------------------------------------------------------------------- #
# contamination
# --------------------------------------------------------------------------- #

def cmd_audit(args):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import inchi, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    fn = OUT / f"litpcba_pairs_{args.variant}.parquet"
    lit = pd.read_parquet(fn)
    lit_lig = sorted(lit.smiles.unique())
    log(f"{len(lit_lig):,} unique LIT-PCBA ligands")

    corpus = set()
    for ch in pd.read_csv(CORPUS, chunksize=300_000, usecols=["smiles"]):
        corpus.update(ch.smiles.unique())
    corpus = sorted(corpus)
    log(f"{len(corpus):,} unique training-corpus ligands")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def prep(smis, want_fp):
        keys, fps, keep = [], [], []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            keep.append(s)
            try:
                keys.append(inchi.MolToInchiKey(m)[:14])
            except Exception:
                keys.append(None)
            if want_fp:
                fps.append(gen.GetFingerprint(m))
        return keys, fps, keep

    log("fingerprinting the corpus (this is the slow step)")
    c_keys, c_fps, _ = prep(corpus, want_fp=args.tanimoto)
    c_key_set = {k for k in c_keys if k}

    log("fingerprinting LIT-PCBA")
    l_keys, l_fps, l_keep = prep(lit_lig, want_fp=args.tanimoto)

    exact = np.array([bool(k and k in c_key_set) for k in l_keys])
    res = pd.DataFrame({"smiles": l_keep, "skeleton_in_corpus": exact})

    if args.tanimoto:
        ## Actives only by default: 7.8k x 992k is ~hours, and a memorised ACTIVE is the
        ## contamination that actually inflates enrichment. Pass --tanimoto-all to widen.
        act = set(lit.loc[lit.label == 1, "smiles"])
        idx = [i for i, s in enumerate(l_keep) if args.tanimoto_all or s in act]
        log(f"max-Tanimoto for {len(idx):,} ligands against {len(c_fps):,} corpus "
            f"fingerprints - expect roughly {len(idx)*len(c_fps)/1e9:.1f}e9 comparisons")
        mx = np.full(len(l_keep), np.nan, np.float32)
        for n, i in enumerate(idx):
            mx[i] = max(DataStructs.BulkTanimotoSimilarity(l_fps[i], c_fps))
            if (n + 1) % 500 == 0:
                log(f"  {n+1:,}/{len(idx):,}")
        res["max_tanimoto_to_corpus"] = mx

    ## A compound is an ACTIVE if it is active against ANY target - .max(), not
    ## drop_duplicates(). drop_duplicates keeps the first row per SMILES, so a compound
    ## inactive against an alphabetically earlier target was labelled inactive and the
    ## actives set collapsed from 7,761 to 410. groupby-max reproduces the published
    ## count of 7,761 exactly, which is the check that it is right.
    lab = lit.groupby("smiles").label.max()
    res["label"] = lab.reindex(res.smiles).to_numpy()
    res.to_csv(OUT / f"litpcba_contamination_{args.variant}.csv", index=False)
    print("\n=== overlap with the training corpus ===")
    for name, sub in (("actives", res[res.label == 1]), ("inactives", res[res.label == 0]),
                      ("all", res)):
        if not len(sub):
            continue
        line = f"{name:>9}: n={len(sub):>7,}  exact skeleton match {sub.skeleton_in_corpus.mean():.2%}"
        if "max_tanimoto_to_corpus" in sub and sub.max_tanimoto_to_corpus.notna().any():
            v = sub.max_tanimoto_to_corpus.dropna()
            line += (f"  |  Tanimoto>=0.5 {100*(v>=0.5).mean():.1f}%  "
                     f"median {v.median():.3f}  max {v.max():.3f}")
        print(line)
    print("\nA high exact-match rate among ACTIVES is the number that would invalidate an")
    print("enrichment claim. Report it next to every EF figure.")


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def ef_at(y, s, frac):
    N, n = len(y), int(y.sum())
    if n == 0 or n == N:
        return np.nan
    k = max(1, int(round(frac * N)))
    return (y[np.argsort(-s)[:k]].sum() / k) / (n / N)


def bedroc(y, s, alpha=20.0):
    N, n = len(y), int(y.sum())
    if n == 0 or n == N:
        return np.nan
    ranks = np.where(y[np.argsort(-s)] == 1)[0] + 1
    Ra = n / N
    rie = np.sum(np.exp(-alpha * ranks / N)) / (Ra * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1))
    return (rie * Ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * Ra))
            + 1.0 / (1 - np.exp(alpha * (1 - Ra))))


def cmd_score(args):
    import joblib
    import xgboost as xgb
    from rdkit import Chem, RDLogger
    from rdkit.Chem import (Crippen, Descriptors, Lipinski, rdFingerprintGenerator,
                            rdMolDescriptors)
    from sklearn.metrics import roc_auc_score
    RDLogger.DisableLog("rdApp.*")

    ## Import torch/transformers FIRST, before any RDKit work. This is not a style
    ## choice: an environment problem here (a huggingface_hub/transformers version
    ## mismatch killed a run after 45 minutes of RDKit descriptors on the HPC, because
    ## the failing import sat after all the RDKit code) must surface in seconds, not
    ## after the slow, cacheable part has already run.
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        sys.exit(
            f"cannot import torch/transformers: {type(e).__name__}: {e}\n\n"
            "If this says something like \"cannot import name 'is_offline_mode' from "
            "'huggingface_hub'\", transformers and huggingface_hub have drifted out of "
            "sync - environment.yml pins only 'transformers>=4.40' with no upper bound, "
            "so pip is free to resolve a transformers build that expects a "
            "huggingface_hub API the installed one no longer has. Fix with:\n\n"
            "    pip install -U transformers huggingface_hub\n\n"
            "and if that alone does not resolve it, pin both together, e.g.:\n\n"
            "    pip install 'transformers==4.46.*' 'huggingface_hub>=0.25,<0.26'\n")

    b = joblib.load(args.bundle)
    log(f"bundle {Path(args.bundle).name}: {b['variant']}")
    clf = xgb.Booster(); clf.load_model(bytearray(b["classifier_json"]))
    RD_COLS, BITS = b["rdkit_columns"], b["ecfp_bits"]

    lit = pd.read_parquet(OUT / f"litpcba_pairs_{args.variant}.parquet")
    if args.exclude_functional:
        lit = lit[~lit.target.isin(FUNCTIONAL)].reset_index(drop=True)
        log(f"excluded functional sets; {lit.target.nunique()} targets remain")
    log(f"{len(lit):,} pairs, hit rate {lit.label.mean():.4%}")

    ligs = pd.Index(lit.smiles.unique())
    prots = pd.Index(lit.target_sequence.unique())
    log(f"{len(ligs):,} unique ligands, {len(prots)} unique sequences")

    ## Cached to disk, keyed by variant + functional-exclusion state, and verified by
    ## row count against the CURRENT ligand set before being trusted. This is the second
    ## half of the same fix: even with the import moved up, RDKit over 380k ligands is
    ## ~50 minutes, and nothing about re-running `score` (a different bundle, a retry
    ## after a cluster hiccup) should pay for it twice.
    tag = f"{args.variant}{'_nofunc' if args.exclude_functional else ''}"
    rdkit_cache = OUT / f"rdkit_cache_{tag}.npz"
    cached_ok = False
    if rdkit_cache.exists():
        z = np.load(rdkit_cache)
        if z["ok"].shape[0] == len(ligs):
            log(f"reusing {rdkit_cache.name}")
            d11, fd, fps, ok = z["d11"], z["fd"], z["fps"], z["ok"]
            cached_ok = True
        else:
            log(f"{rdkit_cache.name} has {z['ok'].shape[0]} rows, expected {len(ligs)} "
                "- recomputing")

    if not cached_ok:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=BITS)
        d11, full, fps, ok = [], [], [], []
        for i, s in enumerate(ligs):
            m = Chem.MolFromSmiles(s)
            ok.append(m is not None)
            if m is None:
                d11.append([0.0] * 11); full.append({}); fps.append(np.zeros(BITS)); continue
            mh = Chem.AddHs(m)
            d11.append([Descriptors.MolWt(mh), Crippen.MolLogP(mh), Lipinski.NumHDonors(mh),
                        Lipinski.NumHAcceptors(mh), Lipinski.NumRotatableBonds(mh),
                        rdMolDescriptors.CalcTPSA(mh), rdMolDescriptors.CalcNumRings(mh),
                        rdMolDescriptors.CalcNumAromaticRings(mh), mh.GetNumAtoms(),
                        mh.GetNumHeavyAtoms(), Chem.GetFormalCharge(mh)])
            full.append(Descriptors.CalcMolDescriptors(m))
            fps.append(gen.GetCountFingerprintAsNumPy(m))
            if (i + 1) % 50_000 == 0:
                log(f"  rdkit {i+1:,}/{len(ligs):,}")
        fd = (pd.DataFrame(full).astype(np.float32).replace([np.inf, -np.inf], np.nan)
                .fillna(0.0).reindex(columns=RD_COLS, fill_value=0.0).to_numpy(np.float32))
        d11 = np.asarray(d11, np.float32); fps = np.asarray(fps, np.float32)
        ok = np.asarray(ok)
        np.savez_compressed(rdkit_cache, d11=d11, fd=fd, fps=fps, ok=ok)
        log(f"wrote {rdkit_cache.name}")

    def embed(texts, hf, bs, ml, cache_name):
        cache = OUT / f"{cache_name}.npy"
        if cache.exists():
            a = np.load(cache, mmap_mode="r")
            if a.shape[0] == len(texts):
                log(f"  reuse {cache.name} {a.shape}")
                return np.asarray(a, np.float32)
            log(f"  {cache.name} has {a.shape[0]} rows, expected {len(texts)} - recomputing")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"  {hf} on {dev}, {len(texts):,} items")
        mdl = AutoModel.from_pretrained(hf, trust_remote_code=True).to(dev).eval()
        try:
            tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
        except Exception:
            tok = mdl.tokenizer
        order = np.argsort([len(t) for t in texts]); out = None
        with torch.inference_mode():
            for st in range(0, len(order), bs):
                sel = order[st:st + bs]
                bt = tok([texts[i] for i in sel], padding=True, truncation=True,
                         max_length=ml, return_tensors="pt")
                bt = {k: v.to(dev) for k, v in bt.items()}
                h = mdl(**bt).last_hidden_state
                mk = bt["attention_mask"].unsqueeze(-1).to(h.dtype)
                p = ((h * mk).sum(1) / mk.sum(1).clamp_min(1)).float().cpu().numpy()
                if out is None:
                    out = np.zeros((len(texts), p.shape[1]), np.float32)
                out[sel] = p
                if (st // bs) % 200 == 0:
                    log(f"    {min(st+bs, len(order)):,}/{len(order):,}")
        del mdl
        if dev == "cuda":
            torch.cuda.empty_cache()
        np.save(cache, out)
        log(f"  wrote {cache.name} {out.shape}")
        return out

    L = embed(list(ligs), b["ligand_encoder"], 512, 256, f"lig_emb_{tag}")
    P = embed(list(prots), b["protein_encoder"], 8, 1024, f"prot_emb_{tag}")
    AA = "ACDEFGHIKLMNPQRSTVWY"
    aac = np.array([[s.count(a) / len(s) for a in AA] + [len(s)] for s in prots], np.float32)

    li = ligs.get_indexer(lit.smiles); pi = prots.get_indexer(lit.target_sequence)
    keep = ok[li]
    lit = lit[keep].reset_index(drop=True); li, pi = li[keep], pi[keep]
    log(f"{len(lit):,} pairs after dropping unparseable ligands")

    scores = np.empty(len(lit), np.float32)
    CH = 200_000
    for a in range(0, len(lit), CH):
        z = min(a + CH, len(lit))
        X = np.hstack([L[li[a:z]], d11[li[a:z]], fd[li[a:z]], fps[li[a:z]],
                       P[pi[a:z]], aac[pi[a:z]]]).astype(np.float32)
        scores[a:z] = clf.inplace_predict(X)
        del X
        log(f"  scored {z:,}/{len(lit):,}")
    lit["savant_p_binder"] = scores

    ## 29.5% of LIT-PCBA actives have an exact InChIKey-skeleton match in the training
    ## corpus (per-target: 8% for MTORC1, 96% for OPRK1). Reporting raw enrichment on a
    ## set that overlapping is the leakage-inflated number this whole line of work exists
    ## to avoid, so both tables are produced and the clean one is the quotable one.
    contam_file = OUT / f"litpcba_contamination_{args.variant}.csv"
    seen = set()
    if contam_file.exists():
        c = pd.read_csv(contam_file)
        seen = set(c.loc[c.skeleton_in_corpus, "smiles"])
        log(f"contamination audit: {len(seen):,} corpus-overlapping compounds")
    else:
        log(f"NO contamination audit at {contam_file.name} - run `audit` first. "
            "The 'clean' table below will be missing and raw EF is not quotable.")

    def table(d, tag):
        rows = []
        for t, sub in d.groupby("target"):
            y = sub.label.to_numpy(); s = sub.savant_p_binder.to_numpy()
            if y.sum() < 5 or y.sum() == len(y):
                continue                      # too few actives to enrich against
            rows.append({"subset": tag, "target": t, "n": len(y), "actives": int(y.sum()),
                         "hit_rate": float(y.mean()), "roc_auc": roc_auc_score(y, s),
                         "ef_0.5": ef_at(y, s, 0.005), "ef_1": ef_at(y, s, 0.01),
                         "ef_5": ef_at(y, s, 0.05), "bedroc": bedroc(y, s),
                         "phenotype": TARGETS[t][2]})
        return pd.DataFrame(rows)

    parts = [table(lit, "all")]
    if seen:
        parts.append(table(lit[~lit.smiles.isin(seen)].reset_index(drop=True), "clean"))
    R = pd.concat(parts, ignore_index=True)
    R.to_csv(OUT / f"litpcba_scored_{args.variant}.csv", index=False)
    lit.to_parquet(OUT / f"litpcba_predictions_{args.variant}.parquet", index=False)

    for tag, sub in R.groupby("subset"):
        print(f"\n=== {tag} ===")
        print(sub.drop(columns=["subset"]).sort_values("ef_1", ascending=False)
                .round(4).to_string(index=False))
        inh = sub[sub.phenotype == "inhibitor"]
        if len(inh):
            print(f"inhibitor sets only ({len(inh)} targets): median EF@1% "
                  f"{inh.ef_1.median():.2f}, median ROC-AUC {inh.roc_auc.median():.3f}, "
                  f"median BEDROC {inh.bedroc.median():.3f}")
    print("\nEF@1% of 1.0 is random. Quote the 'clean' subset; report 'all' beside it and")
    print("let the gap between them be the contamination measurement.")
    print("Use the MEDIAN across targets, not the mean: ALDH1 holds 5,363 of the 7,761")
    print("actives and its EF ceiling is 20 while most targets' is 100, so a mean is")
    print("mostly a report on ALDH1.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("download", "build", "audit", "score"):
        s = sub.add_parser(name)
        s.add_argument("--variant", choices=["ave", "full"], default="ave",
                       help="ave = AVE-debiased sets (default, the ML-appropriate one)")
        if name == "download":
            s.add_argument("--force", action="store_true")
        if name == "audit":
            s.add_argument("--tanimoto", action="store_true",
                           help="also compute max ECFP4 Tanimoto (slow)")
            s.add_argument("--tanimoto-all", action="store_true",
                           help="Tanimoto for inactives too (very slow)")
        if name == "score":
            s.add_argument("--bundle", required=True)
            s.add_argument("--exclude-functional", action="store_true",
                           help="drop agonist/antagonist sets an affinity model cannot represent")
    args = ap.parse_args()
    {"download": cmd_download, "build": cmd_build,
     "audit": cmd_audit, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
