"""The label policy. PLAN.md Section 6.5 requires this to run at ingestion and in CI.

Section 0.4 rule 1: "Never use IC50 or EC50 in any form: not as labels, auxiliary
heads, features, sample weights, filters, or model-selection criteria."

The rule is easy to state and easy to violate by omission. The previous sprint's
merge step wrote

    if kd_ki_only and t:

so a source with no type column produced an empty `t`, the filter never ran, and
1.8M untyped rows entered the corpus tagged "unknown". These tests exist because
that bug was invisible in every downstream artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pytest

from pfpl.data import schema as S


def test_normalize_type_accepts_the_spellings_we_see():
    s = pd.Series(["Kd", "KD", "kd", "K_D", "Kd (nM)", "Ki", "KI", "ki", "pKi", "pKd"])
    out = S.normalize_type(s)
    assert set(out) == {"Kd", "Ki"}
    assert list(out[:5]) == ["Kd"] * 5


def test_normalize_type_rejects_every_forbidden_type():
    s = pd.Series(list(S.FORBIDDEN_TYPES))
    out = S.normalize_type(s)
    assert (out == "").all(), f"a forbidden type was accepted: {set(out) - {''}}"


def test_normalize_type_rejects_missing_and_unknown():
    s = pd.Series(["", None, np.nan, "unknown", "potency", "n/a", "Kdx", "KD50"])
    assert (S.normalize_type(s) == "").all()


def test_classify_rejections_separates_forbidden_from_unrecognised():
    s = pd.Series(["Ki", "IC50", "wibble", ""])
    why = S.classify_rejections(s)
    assert why.iloc[0] == "kept"
    assert "forbidden" in why.iloc[1]
    assert why.iloc[2] == "unrecognised"
    assert why.iloc[3] == "missing"


def test_untyped_source_is_refused_not_guessed():
    """The exact bug from the previous sprint: no type column must be fatal."""
    df = pd.DataFrame({"smiles": ["CCO"], "sequence": ["MKT"], "value_nM": [10.0]})
    with pytest.raises(SystemExit) as e:
        S.require_typed(df, "input_csv")
    msg = str(e.value)
    assert "assume-type" in msg, "the refusal must name the escape hatch"
    assert "IC50" in msg, "the refusal must say why"


def test_assumed_type_is_recorded_and_validated():
    df = pd.DataFrame({"smiles": ["CCO"], "value_nM": [10.0]})
    t = S.require_typed(df, "input_csv", "Kd")
    assert (t == "Kd").all()
    assert t.attrs["type_source"] == "declared", \
        "a declared type must be distinguishable from a read one, or the corpus " \
        "cannot be re-filtered later"
    with pytest.raises(SystemExit):
        S.require_typed(df, "input_csv", "IC50")


def test_assert_label_policy_catches_a_disallowed_type():
    df = pd.DataFrame({"type": ["Kd", "Ki", "IC50"], "pK": [7, 8, 9]})
    with pytest.raises(AssertionError, match="disallowed"):
        S.assert_label_policy(df)


def test_assert_label_policy_catches_a_leaked_column():
    """A column NAME that leaks an excluded assay is a design error even if unused:
    something upstream carried the value along and a later join could read it."""
    df = pd.DataFrame({"type": ["Kd"], "pK": [7.0], "ic50_nM": [100.0]})
    with pytest.raises(AssertionError, match="excluded assay type"):
        S.assert_label_policy(df)
    df2 = pd.DataFrame({"type": ["Kd"], "pK": [7.0], "IC 50": [100.0]})
    with pytest.raises(AssertionError):
        S.assert_label_policy(df2)


def test_assert_label_policy_catches_an_unknown_relation():
    df = pd.DataFrame({"type": ["Kd"], "relation": ["<="], "pK": [7.0]})
    with pytest.raises(AssertionError, match="relation"):
        S.assert_label_policy(df)


def test_relation_is_stored_in_pK_space():
    """pK = -log10(Kd) is monotonically DECREASING, so a concentration bound flips.

    Getting this wrong is invisible - right row count, right label, and only the
    DIRECTION of the one-sided censored loss is backwards, so the model is taught
    the opposite bound for those pairs.
    """
    s = pd.Series([">", "<", "=", ">=", "<=", "~", None])
    out = S.flip_relation(s)
    assert list(out) == ["<", ">", "=", "<", ">", "=", "="]
    assert set(out) <= set(S.ALLOWED_RELATIONS)


def test_censored_hinge_matches_the_pK_convention():
    torch = pytest.importorskip("torch")
    from pfpl.train.loop import censored_hinge
    target = torch.tensor([6.0, 6.0])
    w = torch.ones(2)

    ## rel ">" (pK ABOVE 6): a prediction of 8 is consistent, 4 violates it.
    gt = torch.tensor([1, 1])
    assert float(censored_hinge(torch.tensor([8.0, 8.0]), target, gt, w)) == 0.0
    assert float(censored_hinge(torch.tensor([4.0, 4.0]), target, gt, w)) > 0

    ## rel "<" (pK BELOW 6): the mirror image.
    lt = torch.tensor([2, 2])
    assert float(censored_hinge(torch.tensor([4.0, 4.0]), target, lt, w)) == 0.0
    assert float(censored_hinge(torch.tensor([8.0, 8.0]), target, lt, w)) > 0

    ## exact rows contribute nothing to this term
    assert float(censored_hinge(torch.tensor([99.0, 99.0]), target,
                                torch.tensor([0, 0]), w)) == 0.0


def test_every_loader_flips_or_documents_its_relation():
    """A concentration-space source that forgets to flip is the silent failure."""
    import re
    src = ROOT / "src" / "pfpl" / "data" / "sources"
    offenders = []
    for f in src.glob("*.py"):
        if f.name in ("__init__.py", "local_csv.py", "pdbbind.py"):
            continue          # local_csv passes through; pdbbind flips inline
        text = f.read_text(encoding="utf-8")
        if '"relation"' not in text and "relation=" not in text:
            continue
        if "flip_relation" not in text:
            offenders.append(f.name)
    assert not offenders, (
        f"{offenders} set `relation` without flip_relation; if the source really "
        "reports in pK space, say so in a comment and add it to this exemption list")


def test_papyrus_drops_aggregates_averaged_with_ic50(tmp_path):
    """A Papyrus row is a MEAN over every measurement of a pair. Flagged Ki AND IC50,
    its pchembl_value_Mean is IC50-contaminated and must never become a label."""
    import lzma
    from pfpl.data.sources.papyrus import load
    cols = ["Activity_ID", "Quality", "SMILES", "InChIKey", "target_id", "accession",
            "all_doc_ids", "type_IC50", "type_EC50", "type_KD", "type_Ki",
            "type_other", "relation", "pchembl_value_Mean"]
    rows = [["a", "High", "CCO", "K1", "P1_WT", "P1", "d", "0", "0", "0", "1", "0", "=", "7.0"],
            ["b", "High", "CCN", "K2", "P1_WT", "P1", "d", "1", "0", "0", "1", "0", "=", "8.0"],
            ["c", "High", "CCC", "K3", "P1_WT", "P1", "d", "0", "0", "1", "1", "0", "=", "6.0"],
            ["d", "High", "CCS", "K4", "P1_WT", "P1", "d", "0", "0", "1", "0", "0", ">", "5.0"]]
    bio = tmp_path / "05.7++_combined_set_without_stereochemistry.tsv.xz"
    with lzma.open(bio, "wt") as fh:
        fh.write("\n".join("\t".join(r) for r in [cols, *rows]) + "\n")
    tgt = tmp_path / "05.7_combined_set_protein_targets.tsv.xz"
    with lzma.open(tgt, "wt") as fh:
        fh.write("target_id\tSequence\nP1_WT\tMKTAYIAKQR\n")
    out = load(path=str(bio), targets=str(tgt))
    assert sorted(out["source_record_id"]) == ["a", "d"], \
        "only pure-Ki / pure-Kd aggregates may survive"
    assert not any("50" in c for c in out.columns)
    ## ">" on a concentration is "<" in pK space
    assert out.set_index("source_record_id").loc["d", "relation"] == "<"
    assert out["sequence"].notna().all()


def test_pK_round_trip():
    for nm in (0.1, 1.0, 10.0, 1e3, 1e6):
        assert np.isclose(S.from_pK(S.to_pK(nm)), nm, rtol=1e-9)
    assert np.isclose(S.to_pK(1.0), 9.0)          # 1 nM -> pK 9
    assert np.isclose(S.to_pK(1e9), 0.0)          # 1 M  -> pK 0


def test_pK_rejects_nonpositive_concentrations():
    out = S.to_pK([0.0, -1.0, np.nan, 5.0])
    assert np.isnan(out[:3]).all() and np.isfinite(out[3])


## The ONE key allowed to name an excluded assay: PLAN.md Section 4.2's switch for
## using an IC50-only complex's STRUCTURE with the affinity loss masked and the
## value never read. Nothing else may appear in an active config line.
_ALLOWED_IC50_KEY = "USE_IC50_ONLY_STRUCTURES"


def test_no_config_mentions_an_excluded_assay():
    """Section 0.4 rule 1 covers filters and model-selection criteria too."""
    bad = []
    for p in (ROOT / "configs").rglob("*.yaml"):
        for line in p.read_text().upper().splitlines():
            code = line.split("#", 1)[0].strip()
            if not code or code.startswith(_ALLOWED_IC50_KEY):
                continue
            if any(f in code for f in ("IC50", "EC50")):
                bad.append(f"{p.name}: {line.strip()}")
    assert not bad, f"an excluded assay type appears in an active config line: {bad}"


def test_ic50_only_structures_defaults_to_off():
    """Section 4.2: the DEFAULT excludes such complexes entirely, including from
    geometry pretraining, so no IC50-associated record enters training in any form."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    assert cfg["label"]["use_ic50_only_structures"] is False


def test_conform_fills_pK_from_value_and_back():
    df = pd.DataFrame({"type": ["Kd"], "value_nM": [10.0]})
    out = S.conform(df, "test")
    assert np.isclose(out.pK.iat[0], 8.0)
    df2 = pd.DataFrame({"type": ["Ki"], "pK": [8.0]})
    out2 = S.conform(df2, "test")
    assert np.isclose(float(out2.value_nM.iat[0]), 10.0)
    assert out2.relation.iat[0] == "="
    assert list(out.columns) == S.COLUMNS
