"""Cache, config and path resolution.

The path tests exist because this project removed `cd "$(dirname "$0")/.."` from
every Slurm script. The claim "nothing depends on the current directory" has to be
checkable, or it decays the first time someone adds a relative `open()`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from pfpl import conf
from pfpl.features.cache import ShardCache, content_hash, shard_range, slurm_shard


# --------------------------------------------------------------------- paths

def test_root_resolves_without_the_cwd():
    """The whole point of removing `cd` from the sbatch scripts."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s'); "
         "from pfpl import paths; print(paths.ROOT)" % (ROOT / "src")],
        capture_output=True, text=True, cwd=Path(sys.executable).parent,
        env={**os.environ, "PFPL_ROOT": "", "SLURM_SUBMIT_DIR": ""})
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()) == ROOT, \
        f"resolved to {out.stdout.strip()!r} when run from another directory"


def test_everything_is_contained_under_the_root():
    """The user's requirement: the project runs entirely inside PFPL_ROOT.

    On this cluster $HOME is quota'd and a compute node's /tmp is node-local and
    reclaimed when the job ends, so an unredirected HuggingFace, Triton or tempfile
    path is a job that dies hours in, or a shard that silently loses its scratch.
    """
    from pfpl import paths
    assert paths.escapes() == [], (
        f"these point outside {paths.ROOT}: {paths.escapes()}")
    for name in ("WORK", "DATA", "CACHE", "RUNS", "LOGS", "SPLITS", "REPORTS", "TMP"):
        Path(getattr(paths, name)).resolve().relative_to(paths.ROOT)


def test_contain_reclaims_a_preset_temp_dir(monkeypatch, tmp_path):
    """Pure setdefault is not enough: Windows and many login shells pre-set TMP and
    TEMP, and tempfile reads TMPDIR, TEMP, TMP in that order."""
    from pfpl import paths
    monkeypatch.setenv("TMP", str(tmp_path))          # outside the root
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.delenv("TMPDIR", raising=False)
    applied = paths.contain()
    assert "TMP" in applied and "TEMP" in applied and "TMPDIR" in applied
    for v in ("TMP", "TEMP", "TMPDIR"):
        Path(os.environ[v]).resolve().relative_to(paths.ROOT)


def test_contain_respects_a_value_already_inside_the_root(monkeypatch):
    from pfpl import paths
    inside = paths.WORK / "my-own-cache"
    monkeypatch.setenv("TORCH_HOME", str(inside))
    paths.contain()
    assert os.environ["TORCH_HOME"] == str(inside)


def test_allow_external_is_an_explicit_opt_out(monkeypatch):
    """A read-only site-wide model cache is a legitimate reason to point outside."""
    from pfpl import paths
    monkeypatch.setenv("PFPL_ALLOW_EXTERNAL", "HF_HOME")
    monkeypatch.setenv("HF_HOME", "/shared/models")
    paths.contain()
    assert os.environ["HF_HOME"] == "/shared/models"
    assert not any("HF_HOME" in e for e in paths.escapes())


def test_no_slurm_script_uses_a_cd_out_of_dirname():
    """The specific pattern the user asked to be removed, and why: Slurm copies a
    --wrap body to a spool dir, a symlinked launcher resolves to the link, and a
    container bind-mount rewrites the prefix - in each case $0 is not the script."""
    offenders = []
    for p in (ROOT / "slurm").glob("*"):
        if p.is_file():
            for i, line in enumerate(p.read_text().splitlines(), 1):
                s = line.strip()
                if s.startswith("#"):
                    continue
                if "dirname" in s and s.startswith("cd "):
                    offenders.append(f"{p.name}:{i}: {s}")
    assert not offenders, offenders


def test_no_bare_cd_in_slurm_scripts():
    offenders = []
    for p in (ROOT / "slurm").glob("*"):
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or not s.startswith("cd "):
                continue
            offenders.append(f"{p.name}:{i}: {s}")
    assert not offenders, f"a `cd` crept back into a Slurm script: {offenders}"


def test_scripts_do_not_open_relative_paths():
    """Every artifact path goes through pfpl.paths, which is absolute."""
    import re
    bad = []
    pat = re.compile(r"""(?:open|read_text|write_text|read_csv|read_parquet)\(\s*["']"""
                     r"""(?!/)[\w.]+/""")
    for p in (ROOT / "scripts").glob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line) and "paths." not in line:
                bad.append(f"{p.name}:{i}: {line.strip()}")
    assert not bad, bad


# -------------------------------------------------------------------- config

def test_config_loads_and_composes_groups():
    cfg = conf.load("config")
    assert cfg.label.target == "pKd"
    assert set(cfg.label.label_types) == {"Kd", "Ki"}
    assert cfg.model.variant == "A"
    assert cfg.plm.mode == "esm2_bottleneck"
    assert "vdw" in cfg.model.energy.terms


def test_group_override_replaces_the_group_wholesale():
    """`-o model=variant_c` must not leave variant_a's energy block behind."""
    cfg = conf.load("config", ["model=variant_c"])
    assert cfg.model.variant == "C"
    cfg2 = conf.load("config", ["plm=none"])
    assert cfg2.plm.mode == "none"


def test_inherit_flattens_rather_than_nesting():
    """Hydra's `defaults` would produce cfg.model.model.*; `_inherit` is the flat
    version for within-group inheritance."""
    cfg = conf.load("config", ["model=variant_b"])
    assert cfg.model.variant == "B"
    assert "model" not in cfg.model, "a group inherited itself into a nested key"
    assert cfg.model.d_model == 256, "the inherited defaults were lost"


def test_dotted_override_and_coercion():
    cfg = conf.load("config", ["seed=99", "model.topk=16",
                               "model.energy.residual=true",
                               "model.energy.terms=['vdw','hbond']"])
    assert cfg.seed == 99 and isinstance(cfg.seed, int)
    assert cfg.model.topk == 16
    assert cfg.model.energy.residual is True
    assert cfg.model.energy.terms == ["vdw", "hbond"]


def test_config_hash_is_order_independent_and_change_sensitive():
    a = conf.load("config", ["seed=1", "model.topk=32"])
    b = conf.load("config", ["model.topk=32", "seed=1"])
    c = conf.load("config", ["seed=1", "model.topk=64"])
    assert a.hash == b.hash
    assert a.hash != c.hash


def test_unknown_config_and_malformed_override_fail_loudly():
    with pytest.raises(SystemExit):
        conf.load("does_not_exist")
    with pytest.raises(SystemExit, match="key=value"):
        conf.load("config", ["nonsense"])


def test_preregistration_declares_the_matrix():
    p = conf.load("preregistration")
    assert p.get("trial_cap"), "Section 7.4 requires a trial cap"
    assert "model.variant" in p["matrix"]
    assert "C" in p["matrix"]["model.variant"], \
        "the no-physics control must be in the declared matrix or Tier 1 has no gate"
    assert "none" in p["matrix"]["plm.mode"], \
        "Section 6.7 asks for plm.mode=none to be run, not assumed"


# --------------------------------------------------------------------- cache

def test_shard_range_covers_everything_exactly_once():
    n, k = 1000, 7
    seen = np.concatenate([shard_range(n, i, k) for i in range(k)])
    assert len(seen) == n and len(np.unique(seen)) == n
    with pytest.raises(SystemExit):
        shard_range(n, 7, 7)


def test_content_hash_is_stable_and_sensitive():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_slurm_shard_refuses_the_ambiguous_single_task_case(monkeypatch):
    """`sbatch --array=9` sets SLURM_ARRAY_TASK_COUNT to 1, not 10. A silent
    shard-count mismatch has already cost this project one full benchmark run."""
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "9")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "1")
    with pytest.raises(SystemExit, match="shard-id"):
        slurm_shard()
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "20")
    assert slurm_shard() == (9, 20)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    assert slurm_shard() == (0, 1)


def test_shard_cache_merge_is_ordered_and_refuses_holes(tmp_path):
    c = ShardCache("t", "k", tmp_path)
    for s in range(3):
        idx = np.arange(s * 4, s * 4 + 4)
        c.write_shard(s, index=idx, value=idx.astype(np.float32) * 2)
    c.merge(3, ["value"])
    z = c.load(mmap=False)
    assert list(z["index"]) == list(range(12))
    assert list(z["value"]) == [i * 2 for i in range(12)]

    c2 = ShardCache("u", "k", tmp_path)
    c2.write_shard(0, index=np.arange(4), value=np.zeros(4, np.float32))
    with pytest.raises(SystemExit, match="missing"):
        c2.merge(3, ["value"])


def test_shard_cache_detects_overlapping_shards(tmp_path):
    """Two shards claiming the same indices means the geometry changed between runs."""
    c = ShardCache("v", "k", tmp_path)
    for s in range(2):
        c.write_shard(s, index=np.arange(4), value=np.zeros(4, np.float32))
    with pytest.raises(SystemExit, match="overlap"):
        c.merge(2, ["value"])


def test_shard_write_is_atomic(tmp_path):
    c = ShardCache("w", "k", tmp_path)
    assert not c.done(0)
    c.write_shard(0, index=np.arange(2), value=np.zeros(2, np.float32))
    assert c.done(0)
    assert not list(c.dir.glob("*.tmp.npz")), "a temp file survived the rename"


def test_a_cache_from_a_different_corpus_is_refused(tmp_path, monkeypatch):
    """Shards are indexed by protein_idx/ligand_idx, which Phase 1 reassigns on
    every rebuild. A stale shard looks identical on disk and the resume logic would
    skip it as "already present", pairing embeddings with the WRONG protein."""
    import numpy as np
    import pandas as pd
    import pfpl.paths as P
    from pfpl.features import cache as C

    monkeypatch.setattr(P, "WORK", tmp_path)

    def corpus(n_prot):
        pd.DataFrame({"protein_idx": np.arange(n_prot),
                      "sequence_hash": [f"h{i}" for i in range(n_prot)]}
                     ).to_parquet(tmp_path / "proteins.parquet")
        pd.DataFrame({"protein_idx": np.arange(20) % n_prot,
                      "ligand_idx": np.arange(20)}).to_parquet(tmp_path / "pairs.parquet")

    d = tmp_path / "cache" / "plm_x"
    corpus(10)
    first = C.corpus_fingerprint()
    assert C.check_stamp(d, "embeddings") is True          # stamps it
    assert C.check_stamp(d, "embeddings") is True          # same corpus, still fine

    corpus(9)
    assert C.corpus_fingerprint() != first
    with pytest.raises(SystemExit) as e:
        C.check_stamp(d, "per-residue embeddings")
    assert "rm -rf" in str(e.value), "the refusal must say how to fix it"
    assert C.check_stamp(d, "x", strict=False) is False

    corpus(10)
    assert C.check_stamp(d, "embeddings") is True, \
        "an identical rebuild must REUSE its caches, not force a recompute"
