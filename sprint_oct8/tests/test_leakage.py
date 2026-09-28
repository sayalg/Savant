"""Split guarantees and benchmark quarantine (PLAN.md Sections 5.3, 6.5).

Every split in this project advertises a guarantee. These tests check that it
delivers it, on synthetic data where the answer is known and - when the real
artifacts exist - on the frozen splits themselves.

The cost of not having these: a protein-GROUPED split put 21 near-identical HIV-1
RT constructs in the fit fold while another RT sat in test at 0.94 k-mer
containment. Internal r came out at 0.580 and the same model scored 0.327
externally. Nothing in the pipeline noticed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pytest

from pfpl import paths
from pfpl.data import quarantine as Q
from pfpl.data import splits as SP


def fake_pairs(n=600, n_prot=40, n_scaf=25, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "protein_idx": rng.integers(0, n_prot, n),
        "pK": rng.normal(7, 1.5, n),
        "sequence_hash": [f"h{i}" for i in rng.integers(0, n_prot, n)],
        "inchikey_full": [f"K{i}-AAAAAAAA-N" for i in range(n)],
        "inchikey_flat": [f"K{i % n_scaf}" for i in range(n)],
        ## Distinct molecules: identical SMILES would make every Tanimoto 1.0 and
        ## the strict-quarantine test would pass for the wrong reason.
        "ligand_smiles_std": ["C" * (1 + i % 9) + "O" + "C" * (1 + (i // 9) % 7)
                              for i in range(n)],
    })


def test_grouped_split_holds_out_whole_groups():
    g = np.repeat(np.arange(50), 12)
    tr, te = SP.grouped_split(g, 0.2, 7)
    assert not (set(np.unique(g[tr])) & set(np.unique(g[te])))
    assert 0.05 < te.mean() < 0.45


def test_grouped_split_sizes_by_pairs_not_by_group_count():
    """Group sizes here are extremely skewed - one protein had 10,163 ligands and
    the median had 2 - so drawing 20% of the GROUPS yields anywhere from 2% to 60%
    of the rows."""
    g = np.concatenate([np.zeros(5000, int), np.arange(1, 201)])
    fracs = [SP.grouped_split(g, 0.2, s)[1].mean() for s in range(12)]
    assert all(f < 0.55 for f in fracs), \
        f"a pair-quota split still produced {max(fracs):.0%} of rows in test"


def test_three_way_folds_are_mutually_disjoint():
    g = np.repeat(np.arange(60), 10)
    tr, va, te = SP.three_way(g, 0.1, 0.2, 3)
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not (set(np.unique(g[a])) & set(np.unique(g[b])))


def test_double_disjoint_discards_the_single_axis_pairs():
    """Discarding is the point: a pair whose protein is held out but whose scaffold
    was trained on measures memorisation, and mixing the two is how 0.580 internal
    became 0.327 external."""
    rng = np.random.default_rng(0)
    pc = rng.integers(0, 40, 2000)
    sc = rng.integers(0, 30, 2000)
    tr, va, te = SP.double_disjoint(pc, sc, 0.1, 0.2, 5)
    assert not (set(np.unique(pc[tr])) & set(np.unique(pc[te])))
    assert not (set(np.unique(sc[tr])) & set(np.unique(sc[te])))
    assert (len(tr) - tr.sum() - va.sum() - te.sum()) > 0, \
        "nothing was discarded, so one of the axes is not really disjoint"


def test_leave_viral_out_moves_whole_clusters():
    pc = np.repeat(np.arange(20), 25)
    viral = pc < 4
    tr, va, te = SP.leave_viral_out(pc, viral)
    assert te[viral].all(), "a viral pair stayed in training"
    assert not (set(np.unique(pc[tr])) & set(np.unique(pc[te])))


def test_describe_raises_when_a_split_breaks_its_promise():
    g = np.arange(200) % 20
    tr = np.zeros(200, bool); tr[:150] = True
    te = np.zeros(200, bool); te[100:] = True          # deliberate overlap
    with pytest.raises(SystemExit, match="disjoint"):
        SP.describe("protein_cluster", tr, np.zeros(200, bool), te, g, strict=True)


def test_describe_warns_but_returns_when_not_strict():
    g = np.arange(200) % 20
    tr = np.zeros(200, bool); tr[:150] = True
    te = np.zeros(200, bool); te[100:] = True
    d = SP.describe("protein_cluster", tr, np.zeros(200, bool), te, g, strict=False)
    assert d["guarantee_ok"] is False


def test_clusters_from_mmseqs_rejects_an_out_of_sync_fasta(tmp_path):
    tsv = tmp_path / "c.tsv"
    tsv.write_text("p0\tp0\np0\tp1\n")
    assert list(SP.clusters_from_mmseqs(tsv, 2)) == [0, 0]
    with pytest.raises(SystemExit, match="out of sync"):
        SP.clusters_from_mmseqs(tsv, 5)


def test_quarantine_removes_exact_benchmark_pairs():
    pairs = fake_pairs()
    bench = pairs.sample(20, random_state=1)
    keys = Q.build_keys(bench)
    drop = Q.apply_exact(pairs, keys, verbose=False)
    assert drop.sum() >= 20
    with pytest.raises(AssertionError):
        Q.assert_clean_frame(pairs, keys)
    Q.assert_clean_frame(pairs[~drop], keys)


def test_quarantine_strict_needs_both_axes():
    """Section 5.3.1: homologous protein AND near-identical ligand. A held-out
    protein tested with a genuinely novel ligand must survive."""
    pairs = fake_pairs(n=200, n_prot=10, n_scaf=200)
    bench = pairs.iloc[:5].copy()
    keys = Q.build_keys(bench)
    clusters = pairs.protein_idx.to_numpy()
    drop = Q.apply_strict(pairs, bench, keys, protein_clusters=clusters, verbose=False)
    assert drop[:5].all(), "the exact benchmark pairs must go"
    same_protein_new_ligand = (pairs.protein_idx.isin(bench.protein_idx).to_numpy()
                               & ~pairs.inchikey_flat.isin(bench.inchikey_flat).to_numpy())
    kept = same_protein_new_ligand & ~drop
    assert kept.sum() > 0, \
        "strict quarantine removed rows that share only the protein axis; it should " \
        "require both (Section 5.3.1)"


def test_kmer_containment_is_symmetric_and_bounded():
    a, b = Q.kmers("A" * 50), Q.kmers("A" * 50)
    assert Q.containment(a, b) == 1.0
    assert Q.containment(Q.kmers("MKTAYIAKQRQ" * 6), Q.kmers("WWWWWWWWWW" * 6)) == 0.0
    assert Q.containment(set(), a) == 0.0


## Both artifacts are needed, and on a cluster they live in different places:
## splits/ is version-controlled under the repo while the corpus sits on $PFPL_WORK.
## Guarding on only one of them turns "PFPL_WORK is not set" into a test failure.
@pytest.mark.skipif(
    not ((paths.SPLITS / "index.json").exists()
         and (paths.WORK / "pairs.parquet").exists()),
    reason=f"needs both {paths.SPLITS / 'index.json'} and {paths.WORK / 'pairs.parquet'} "
           "(run phase1 + phase5, and set PFPL_WORK if the corpus is on scratch)")
def test_frozen_splits_still_satisfy_their_guarantees():
    """Re-check the real artifacts, not just the algorithm. A split file can go stale
    against a rebuilt corpus, and nothing else would notice."""
    import json
    idx = json.loads((paths.SPLITS / "index.json").read_text())
    pairs = pd.read_parquet(paths.WORK / "pairs.parquet")
    cl = np.load(paths.WORK / "protein_clusters.npy")[pairs.protein_idx.to_numpy()]
    for name, meta in idx.items():
        tr, va, te = SP.load(paths.SPLITS, name)
        assert len(tr) == len(pairs), \
            f"split {name} has {len(tr)} rows but the corpus has {len(pairs)} - the " \
            "corpus was rebuilt without rebuilding the splits"
        if name in ("protein_cluster", "double", "leave_viral_out"):
            assert not (set(np.unique(cl[tr])) & set(np.unique(cl[te]))), \
                f"split {name} has a protein cluster in both folds"
        assert not (tr & te).any()


@pytest.mark.skipif(not (paths.WORK / "pairs.parquet").exists(),
                    reason="corpus not built yet")
def test_corpus_contains_no_benchmark_pair():
    qp = ROOT / "splits" / "quarantine.json"
    if not qp.exists():
        pytest.skip("no quarantine keys")
    Q.assert_clean(pd.read_parquet(paths.WORK / "pairs.parquet"), qp)
