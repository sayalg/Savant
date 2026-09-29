"""Harmonisation, standardisation and the metric definitions.

Section 8's Phase 1 gate asks for the dedupe to be verified on hand-checked pairs.
These are those pairs, written down so the verification survives the afternoon it
was done in.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pytest

from pfpl.data import harmonize as H
from pfpl.data import schema as S
from pfpl.data.standardize import apply_mutations, clean_sequence, sequence_hash
from pfpl.eval import metrics as M

rdkit = pytest.importorskip("rdkit.Chem", reason="RDKit not installed")


def frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = S.conform(df, "test")
    ## conform() stamps source_db; keep the per-row value when the test set one, since
    ## the dedupe's source preference is exactly what some of these tests check.
    if "source_db" in df.columns:
        out["source_db"] = df["source_db"].to_numpy()
    return out


# ------------------------------------------------------------------ sequences

def test_clean_sequence_maps_ambiguity_codes():
    assert clean_sequence("mktayiakqrqisfvksh" * 3).isupper()
    assert "B" not in clean_sequence("MKTAYIAKQRQISFVKSHB" * 3)
    assert clean_sequence("M K T A Y I A K Q R Q I S F") is None      # too short
    assert clean_sequence("XXXXXXXXXX" * 5) is None                    # >10% unknown
    assert clean_sequence(None) is None


def test_sequence_hash_ignores_case_and_whitespace():
    assert sequence_hash(" mktayiakq ") == sequence_hash("MKTAYIAKQ")


def test_mutations_apply_or_are_refused():
    seq = "MKTAYIAKQR"
    assert apply_mutations(seq, "K2A") == ("MATAYIAKQR", True)
    assert apply_mutations(seq, "K2A,R10W")[0] == "MATAYIAKQW"
    ## A mutation that does not match the wild-type residue means the numbering is
    ## wrong; training on the wild-type sequence under a mutant's label would teach
    ## the model that two different affinities share one input.
    assert apply_mutations(seq, "A2V") == (None, False)
    assert apply_mutations(seq, "K99A") == (None, False)
    assert apply_mutations(seq, "gibberish") == (None, False)
    assert apply_mutations(seq, "") == (seq, True)


# -------------------------------------------------------------------- dedupe

def test_dedupe_removes_the_same_measurement_from_two_sources():
    rows = [
        dict(source_db="chembl", doc_id="PMID1", uniprot="P1", sequence_hash="h1",
             inchikey_full="AAAA-BBBB-N", type="Ki", pK=8.00, relation="="),
        dict(source_db="bindingdb", doc_id="PMID1", uniprot="P1", sequence_hash="h1",
             inchikey_full="AAAA-BBBB-N", type="Ki", pK=8.004, relation="="),
    ]
    out = H.dedupe_measurements(frame(rows), verbose=False)
    assert len(out) == 1
    assert out.source_db.iat[0] == "chembl", \
        "when a duplicate is found, the structure-linked / better-curated source wins"


def test_dedupe_keeps_genuinely_different_measurements():
    rows = [
        ## different paper
        dict(source_db="chembl", doc_id="PMID1", sequence_hash="h1",
             inchikey_full="A-B-N", type="Ki", pK=8.0, relation="="),
        dict(source_db="chembl", doc_id="PMID2", sequence_hash="h1",
             inchikey_full="A-B-N", type="Ki", pK=8.0, relation="="),
        ## same paper, different value
        dict(source_db="chembl", doc_id="PMID1", sequence_hash="h1",
             inchikey_full="A-B-N", type="Ki", pK=6.5, relation="="),
        ## same paper and value, different assay type
        dict(source_db="chembl", doc_id="PMID1", sequence_hash="h1",
             inchikey_full="A-B-N", type="Kd", pK=8.0, relation="="),
        ## same paper and value, different ligand
        dict(source_db="chembl", doc_id="PMID1", sequence_hash="h1",
             inchikey_full="C-D-N", type="Ki", pK=8.0, relation="="),
    ]
    out = H.dedupe_measurements(frame(rows), verbose=False)
    assert len(out) == 5


def test_dedupe_exempts_rows_with_no_document_id():
    """Rows with no doc_id cannot be shown to be duplicates of each other; dropping
    them would silently delete whole sources."""
    rows = [dict(source_db="local", doc_id=None, sequence_hash="h1",
                 inchikey_full="A-B-N", type="Kd", pK=7.0, relation="=")] * 4
    assert len(H.dedupe_measurements(frame(rows), verbose=False)) == 4


def test_doc_ids_are_normalised_to_one_vocabulary():
    """Each source spells the same paper differently. Wave 1 keyed on the raw
    string, so no key matched across sources and dedupe removed NOTHING."""
    for raw in ("19231178", "19231178.0", "PMID:19231178", "PMID: 19231178",
                "PMID:19231178;PMID:22168767"):
        assert H.normalize_doc_id(raw) == "PMID:19231178", raw
    assert H.normalize_doc_id("CHEMBL1134862") == "CHEMBL1134862"
    assert H.normalize_doc_id("PDBBIND:1a4k") == "PDBBIND:1A4K"
    for empty in (None, np.nan, "", "  ", "nan", "<NA>"):
        assert H.normalize_doc_id(empty) == ""


def test_dedupe_matches_across_the_real_id_spellings():
    rows = [
        dict(source_db="chembl", doc_id="CHEMBL1134862", uniprot="P1",
             sequence_hash="h1", inchikey_full="A-B-N", type="Ki", pK=8.0,
             relation="="),
        dict(source_db="bindingdb", doc_id="19231178.0", uniprot="P1",
             sequence_hash="h1", inchikey_full="A-B-N", type="Ki", pK=8.0,
             relation="="),
        dict(source_db="papyrus", doc_id="PMID:19231178;PMID:22168767", uniprot="P1",
             sequence_hash="h1", inchikey_full="A-B-N", type="Ki", pK=8.0,
             relation="="),
        dict(source_db="iuphar", doc_id="PMID:19231178", uniprot="P1",
             sequence_hash="h1", inchikey_full="A-B-N", type="Ki", pK=8.0,
             relation="="),
    ]
    out = H.dedupe_measurements(frame(rows), verbose=False)
    assert len(out) == 1, "one measurement re-curated four times is still one"
    assert out.source_db.iat[0] == "chembl"


def test_dedupe_never_merges_two_rows_from_the_same_source():
    """Two rows from ONE source are that source's own replicates. Merging them
    would destroy the replicate count the noise ceiling is measured from."""
    rows = [dict(source_db="chembl", doc_id="PMID:1", uniprot="P1",
                 sequence_hash="h1", inchikey_full="A-B-N", type="Ki", pK=8.0,
                 relation="=")] * 3
    assert len(H.dedupe_measurements(frame(rows), verbose=False)) == 3


def test_dedupe_catches_a_copy_with_no_shared_document_id():
    """ChEMBL document ids carry no PubMed id, so BindingDB's imported-from-ChEMBL
    rows can only be matched on the VALUE."""
    rows = [
        dict(source_db="chembl", doc_id="CHEMBL111", uniprot="P1",
             sequence_hash="h1", inchikey_full="A-B-N", type="Ki", pK=8.0,
             relation="="),
        dict(source_db="bindingdb", doc_id=None, uniprot="P1", sequence_hash="h1",
             inchikey_full="A-B-N", type="Ki", pK=8.0, relation="="),
    ]
    out = H.dedupe_measurements(frame(rows), verbose=False)
    assert len(out) == 1 and out.source_db.iat[0] == "chembl"
    ## but a different RELATION is a different measurement
    rows[1]["relation"] = ">"
    assert len(H.dedupe_measurements(frame(rows), verbose=False)) == 2


def test_canonicalize_proteins_collapses_constructs_of_one_accession():
    """One target entering as three constructs was three "proteins" in Wave 1."""
    full = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    rows = [
        dict(source_db="chembl", uniprot="P00533", sequence=full,
             sequence_hash=sequence_hash(full), inchikey_full="A-B-N", type="Ki",
             pK=8.0, relation="=", doc_id="PMID:1"),
        ## crystallised construct: the same protein, 8 residues short
        dict(source_db="bindingdb", uniprot="P00533", sequence=full[:-8],
             sequence_hash=sequence_hash(full[:-8]), inchikey_full="C-D-N",
             type="Ki", pK=7.0, relation="=", doc_id="PMID:2"),
        ## isoform suffix and whitespace are the same entry
        dict(source_db="pdbbind", uniprot=" p00533-2 ", sequence=full[:-4],
             sequence_hash=sequence_hash(full[:-4]), inchikey_full="E-F-N",
             type="Kd", pK=9.0, relation="=", doc_id="PDBBIND:1a4k"),
        ## a genuinely different target must NOT be touched
        dict(source_db="chembl", uniprot="Q99999", sequence="MGGKWSKSSVIGWPTVRER",
             sequence_hash=sequence_hash("MGGKWSKSSVIGWPTVRER"),
             inchikey_full="G-H-N", type="Ki", pK=6.0, relation="=",
             doc_id="PMID:3"),
    ]
    out = H.canonicalize_proteins(frame(rows), verbose=False)
    assert out.sequence_hash.nunique() == 2, "three constructs are one protein"
    p533 = out[out.uniprot.str.strip().str.upper().str.startswith("P00533")]
    assert (p533.sequence == full).all(), "the LONGEST construct wins"
    assert (p533.sequence_hash == sequence_hash(full)).all(), "hash must follow"
    assert out.sequence_hash.iat[3] == sequence_hash("MGGKWSKSSVIGWPTVRER")


def test_uniprot_accession_pattern_accepts_the_real_ones():
    """A hand-rolled `[A-NR-Z0-9]...` excludes O, P and Q - so P00533, the single
    most common target in this corpus, silently failed to canonicalise."""
    import re
    pat = re.compile(H.UNIPROT_ACC)
    for a in ("P00533", "Q99999", "O95180", "A0A123", "P0DTC2", "A0A0C5B5G6"):
        assert pat.fullmatch(a), a
    for bad in ("", "NOTANACC", "12345", "P0", "p00533"):
        assert not pat.fullmatch(bad), bad


def test_canonicalize_leaves_rows_without_an_accession_alone():
    a, b = "MKTAYIAKQRQ", "MKTAYIAKQ"
    rows = [dict(source_db="local", uniprot=None, sequence=a,
                 sequence_hash=sequence_hash(a), inchikey_full="A-B-N", type="Ki",
                 pK=8.0, relation="=", doc_id=None),
            dict(source_db="local", uniprot="", sequence=b,
                 sequence_hash=sequence_hash(b), inchikey_full="C-D-N", type="Ki",
                 pK=7.0, relation="=", doc_id=None)]
    out = H.canonicalize_proteins(frame(rows), verbose=False)
    assert list(out.sequence) == [a, b]


# ----------------------------------------------------------------- aggregate

def test_aggregate_takes_the_median_and_keeps_the_spread():
    rows = [dict(source_db="s", doc_id=f"D{i}", sequence_hash="h1",
                 inchikey_full="A-B-N", inchikey_flat="A", type="Ki", pK=v,
                 relation="=", sequence="M" * 30, ligand_smiles_std="CCO")
            for i, v in enumerate([6.0, 7.0, 9.0])]
    out = H.aggregate_pairs(frame(rows), verbose=False)
    assert len(out) == 1
    assert out.pK.iat[0] == 7.0
    assert out.pK_spread.iat[0] == 3.0
    assert out.n_measurements.iat[0] == 3
    assert bool(out.flag_high_spread.iat[0])
    assert out.weight.iat[0] < 0.8, "a >1-log-unit spread must be down-weighted"


def test_aggregate_prefers_exact_over_censored():
    rows = [
        dict(source_db="s", doc_id="D1", sequence_hash="h1", inchikey_full="A-B-N",
             type="Ki", pK=7.0, relation="=", sequence="M" * 30, ligand_smiles_std="CCO"),
        dict(source_db="s", doc_id="D2", sequence_hash="h1", inchikey_full="A-B-N",
             type="Ki", pK=5.0, relation=">", sequence="M" * 30, ligand_smiles_std="CCO"),
    ]
    out = H.aggregate_pairs(frame(rows), verbose=False)
    assert len(out) == 1
    assert not bool(out.censored.iat[0]), \
        "a '>' alongside a measured value is a failed assay, not information"
    assert out.pK.iat[0] == 7.0


def test_aggregate_keeps_a_censored_only_pair():
    rows = [dict(source_db="s", doc_id="D2", sequence_hash="h2", inchikey_full="C-D-N",
                 type="Ki", pK=5.0, relation=">", sequence="M" * 30,
                 ligand_smiles_std="CCO")]
    out = H.aggregate_pairs(frame(rows), verbose=False)
    assert len(out) == 1 and bool(out.censored.iat[0])


def test_aggregate_respects_label_types():
    rows = [
        dict(source_db="s", doc_id="D1", sequence_hash="h1", inchikey_full="A-B-N",
             type="Ki", pK=7.0, relation="=", sequence="M" * 30, ligand_smiles_std="CCO"),
        dict(source_db="s", doc_id="D2", sequence_hash="h2", inchikey_full="C-D-N",
             type="Kd", pK=8.0, relation="=", sequence="M" * 30, ligand_smiles_std="CCO"),
    ]
    assert len(H.aggregate_pairs(frame(rows), ("Kd", "Ki"), verbose=False)) == 2
    kd_only = H.aggregate_pairs(frame(rows), ("Kd",), verbose=False)
    assert len(kd_only) == 1 and kd_only.type.iat[0] == "Kd"


def test_noise_ceiling_reports_the_implied_sd():
    rows = []
    for p in range(30):
        for v in (7.0, 7.6):
            rows.append(dict(source_db="s", doc_id=f"D{p}{v}", sequence_hash=f"h{p}",
                             inchikey_full=f"K{p}-B-N", type="Ki", pK=v, relation="=",
                             sequence="M" * 30, ligand_smiles_std="CCO"))
    out = H.aggregate_pairs(frame(rows), verbose=False)
    nc = H.noise_ceiling(out)
    assert nc["n_replicated"] == 30
    assert abs(nc["median_spread_pK"] - 0.6) < 1e-6
    assert nc["implied_sd_pK"] > 0


# ------------------------------------------------------------------- metrics

def test_per_target_rho_is_undefined_for_a_constant_prediction():
    """A constant within a target has no within-target ordering, so reporting a
    per-target rho for it would be reporting a division by zero. The previous
    project's 'oracle rho' column was exactly this mistake."""
    df = pd.DataFrame({"protein_idx": [0] * 12, "y": np.linspace(5, 9, 12)})
    tr = pd.DataFrame({"y": np.linspace(5, 9, 40)})
    b = M.blind_baselines(tr, df)
    assert b["const"].pooled_only and np.isnan(b["const"].per_target_spearman_median)
    assert b["target_mean"].pooled_only


def test_quantile_calibrator_is_monotone_and_cannot_change_rank():
    rng = np.random.default_rng(0)
    y = rng.normal(7, 1.5, 500)
    score = y * 31.4 - 1000           # an arbitrary ranking scale
    cal = M.quantile_calibrator(score, y)
    out = cal(score)
    assert M.spearman(y, out) == pytest.approx(M.spearman(y, score), abs=1e-9)
    assert abs(float(np.mean(out)) - float(np.mean(y))) < 0.2
    assert (np.diff(cal(np.sort(score))) >= -1e-9).all()


def test_rmse_is_undefined_for_an_uncalibrated_scale_free_model():
    """A ranking score has no units; an RMSE against pK would fail a gate for the
    wrong reason. One such model reported an offset slope of +27.3."""
    df = pd.DataFrame({"protein_idx": np.repeat(np.arange(10), 12),
                       "y": np.random.default_rng(0).normal(7, 1.5, 120)})
    df["pred"] = df.y * 100
    r = M.evaluate(df, "ranking", scale_free=True)
    assert np.isnan(r.rmse)
    assert np.isnan(r.offset_slope), "an uncalibrated slope reports arbitrary units"
    r2 = M.evaluate(df, "ranking", scale_free=True,
                    calibrator=M.quantile_calibrator(df.pred, df.y))
    assert np.isfinite(r2.rmse) and r2.rmse < 0.5


def test_shortcut_flag_fires_when_ligand_only_is_close():
    a, b = M.Report("model"), M.Report("ligand_only")
    a.pearson, b.pearson = 0.51, 0.50
    assert M.shortcut_flag(a, b, 0.02)["shortcut"] is True
    a.pearson = 0.60
    assert M.shortcut_flag(a, b, 0.02)["shortcut"] is False


def test_tier1_gate_needs_every_condition():
    def rep(p, rho=0.3):
        r = M.Report("x")
        r.pearson, r.per_target_spearman_median = p, rho
        return r
    rng = np.random.default_rng(0)
    y = rng.normal(7, 1.5, 800)
    good = y + rng.normal(0, 0.8, 800)
    poor = y + rng.normal(0, 1.6, 800)
    res = M.tier1_gate(rep(0.88, 0.5), rep(0.60), rep(0.40, 0.2), rep(0.30),
                       y, good, poor)
    assert res["tier1_pass"], res
    ## same model, but it no longer beats the ligand-only floor on per-target rho
    res2 = M.tier1_gate(rep(0.88, 0.1), rep(0.60), rep(0.40, 0.2), rep(0.30),
                        y, good, poor)
    assert not res2["tier1_pass"]


def test_paired_bootstrap_delta_detects_no_difference():
    rng = np.random.default_rng(1)
    y = rng.normal(7, 1.5, 600)
    a = y + rng.normal(0, 1.0, 600)
    d = M.paired_bootstrap_delta(y, a, a.copy())
    assert abs(d["delta"]) < 1e-9 and not d["excludes_zero"]


def test_enrichment_matches_a_hand_computation():
    y = np.concatenate([np.full(10, 9.0), np.full(90, 5.0)])
    perfect = y.copy()
    ef = M.enrichment(y, perfect, k=10, q=0.9)
    assert ef["ef"] == pytest.approx(10.0)
    assert ef["recall"] == pytest.approx(1.0)


def test_phase1_path_collapses_a_realistic_multi_source_overlap():
    """The Wave 1 failure, end to end: one target and one measurement arriving from
    four sources in four spellings produced 4 measurements and 3 "proteins"."""
    full = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
    construct, iso = full[:-12], full[:-6]
    rows = [
        ## --- one Ki, curated by four sources, three sequence variants -----------
        dict(source_db="chembl", doc_id="CHEMBL1134862", uniprot="P00533",
             sequence=full, inchikey_full="AAA-BBB-N", type="Ki", pK=8.30,
             relation="="),
        dict(source_db="bindingdb", doc_id="19231178.0", uniprot="P00533",
             sequence=construct, inchikey_full="AAA-BBB-N", type="Ki", pK=8.30,
             relation="="),
        dict(source_db="papyrus", doc_id="PMID:19231178;PMID:22168767",
             uniprot="P00533", sequence=full, inchikey_full="AAA-BBB-N", type="Ki",
             pK=8.302, relation="="),
        dict(source_db="pdbbind", doc_id="PDBBIND:1m17", uniprot="P00533-2",
             sequence=iso, inchikey_full="AAA-BBB-N", type="Ki", pK=8.30,
             relation="="),
        ## --- a second, genuinely independent measurement of the same pair -------
        dict(source_db="chembl", doc_id="CHEMBL999", uniprot="P00533",
             sequence=full, inchikey_full="AAA-BBB-N", type="Ki", pK=7.10,
             relation="="),
    ]
    df = frame(rows)
    df["sequence_hash"] = [sequence_hash(s) for s in df["sequence"]]

    d = H.canonicalize_proteins(df, verbose=False)
    assert d.sequence_hash.nunique() == 1, "one target, not three"

    d = H.dedupe_measurements(d, verbose=False)
    assert len(d) == 2, "four curations of one measurement, plus one real replicate"
    assert d.source_db.iat[0] == "pdbbind", "structure-linked source wins the tie"
    assert set(np.round(d.pK, 2)) == {8.30, 7.10}

    pairs = H.aggregate_pairs(d, verbose=False)
    assert len(pairs) == 1
    assert pairs.n_measurements.iat[0] == 2, \
        "the replicate count must reflect real replicates, not re-curated copies"
    assert np.isclose(pairs.pK.iat[0], 7.70)
