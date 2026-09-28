"""Torch datasets and collation.

Per-residue embeddings are stored ragged - one contiguous (total_residues, D) fp16
memmap plus an offset table - rather than as a padded (n_proteins, L_max, D) cube.
With 6,176 proteins and a 1,022-residue cap the padded form is 15.7 GB of which
half is padding; the ragged form is ~7 GB and memmaps cleanly off scratch, so the
dataloader never holds the whole thing in RAM.

Ligand graphs are built on demand with an LRU cache. Building one is about a
millisecond, ligands repeat across pairs, and materialising 300,109 dense
(A, A, BOND_DIM) tensors to disk would cost more than recomputing them.

Censored labels (Section 5.2) travel as a `relation` code and are consumed by the
one-sided loss in `train/loop.py`; they are never silently treated as exact.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.embed_esm import POCKET_EXTRA_DIM, pocket_extra_features
from ..features.embed_ligand import ATOM_DIM, BOND_DIM, build_graph, pad_batch
from ..physics.sites import UNK, encode_sequence

REL_CODE = {"=": 0, ">": 1, "<": 2}


class ProteinStore:
    """Ragged per-residue embedding store written by `scripts/phase4_embed.py`."""

    def __init__(self, root):
        self.root = Path(root)
        meta_p = self.root / "meta.json"
        if not meta_p.exists():
            raise SystemExit(
                f"{meta_p} missing - per-residue embeddings have not been extracted.\n"
                "    sbatch --array=0-11 slurm/40_embed_protein.sbatch\n"
                "    python scripts/phase4_embed.py --merge")
        self.meta = json.loads(meta_p.read_text())
        self.dim = int(self.meta["dim"])
        self.offsets = np.load(self.root / "offsets.npy")
        self.lengths = np.load(self.root / "lengths.npy")
        self.emb = np.memmap(self.root / "emb.f16", dtype=np.float16, mode="r",
                             shape=(int(self.offsets[-1]), self.dim))
        extra_p = self.root / "pocket_extra.f16"
        self.extra = (np.memmap(extra_p, dtype=np.float16, mode="r",
                                shape=(int(self.offsets[-1]), POCKET_EXTRA_DIM))
                      if extra_p.exists() else None)

    def __len__(self) -> int:
        return len(self.lengths)

    def get(self, i: int):
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return (np.asarray(self.emb[a:b], np.float32),
                np.asarray(self.extra[a:b], np.float32) if self.extra is not None else None)


class LigandDistStore:
    """Ragged per-ligand site-distance matrices written by phase3_ligand_phys --merge.

    Row `ligand_idx` is that ligand's (S, S) ensemble-max distance between the
    graph's sites, in `build_graph` order. The triangle penalty's `lig_dist`.
    """

    def __init__(self, root):
        root = Path(root)
        self.offsets = np.load(root / "offsets.npy")
        self.n_sites = np.load(root / "n_sites.npy")
        self.dist = np.memmap(root / "dist.f16", dtype=np.float16, mode="r")

    def get(self, ligand_idx: int) -> np.ndarray | None:
        if not 0 <= ligand_idx < len(self.n_sites):
            return None
        s = int(self.n_sites[ligand_idx])
        if not s:
            return None
        a = int(self.offsets[ligand_idx])
        return np.asarray(self.dist[a:a + s * s], np.float32).reshape(s, s)


@functools.lru_cache(maxsize=1)
def default_lig_dist_store() -> LigandDistStore | None:
    from ..paths import WORK
    root = WORK / "ligand_dist"
    return LigandDistStore(root) if (root / "meta.json").exists() else None


@functools.lru_cache(maxsize=200_000)
def _graph(smiles: str, max_atoms: int):
    return build_graph(smiles, max_atoms=max_atoms)


class PairDataset:
    """One (protein, ligand, pK) row. Emits numpy; `collate` makes the batch."""

    def __init__(self, pairs: pd.DataFrame, proteins: pd.DataFrame,
                 store: ProteinStore | None = None, max_res: int = 1022,
                 max_atoms: int = 64, geometry: dict | None = None,
                 need_plm: bool = True):
        self.pairs = pairs.reset_index(drop=True)
        self.proteins = proteins.reset_index(drop=True)
        self.store = store
        self.max_res, self.max_atoms = max_res, max_atoms
        self.need_plm = need_plm
        ## Optional per-pair crystal geometry, keyed by pair index; present only for
        ## the structure-bearing rows that Stage 1 and oracle mode use.
        self.geometry = geometry or {}
        self._seq_cache: dict[int, np.ndarray] = {}
        if need_plm and store is None:
            raise SystemExit("this plm.mode needs cached embeddings; pass a ProteinStore "
                             "or set plm.mode=none")

    def __len__(self) -> int:
        return len(self.pairs)

    def res_idx(self, pi: int) -> np.ndarray:
        if pi not in self._seq_cache:
            s = str(self.proteins.sequence.iat[pi])[:self.max_res]
            self._seq_cache[pi] = encode_sequence(s)
        return self._seq_cache[pi]

    def __getitem__(self, i: int) -> dict | None:
        r = self.pairs.iloc[i]
        pi = int(r.protein_idx)
        g = _graph(str(r.ligand_smiles_std), self.max_atoms)
        if g is None:
            return None
        ridx = self.res_idx(pi)
        item = {
            "res_idx": ridx, "graph": g,
            "y": np.float32(r.pK), "weight": np.float32(r.get("weight", 1.0)),
            "rel": np.int64(REL_CODE.get(str(r.get("relation", "=")), 0)),
            "protein_idx": pi, "pair_idx": i,
            "n_rot": np.float32(r.get("n_rot", 0.0)),
            "dG_solv": np.float32(r.get("dG_solv", np.nan)),
        }
        ld = default_lig_dist_store()
        if ld is not None and "ligand_idx" in r.index and pd.notna(r.ligand_idx):
            m = ld.get(int(r.ligand_idx))
            if m is not None:
                item["lig_dist"] = m
        if self.store is not None:
            emb, extra = self.store.get(pi)
            item["plm"] = emb[:self.max_res]
            item["pocket_extra"] = (extra[:self.max_res] if extra is not None
                                    else pocket_extra_features(
                                        None, str(self.proteins.sequence.iat[pi])[:self.max_res]))
        else:
            item["pocket_extra"] = pocket_extra_features(
                None, str(self.proteins.sequence.iat[pi])[:self.max_res])
        if i in self.geometry:
            item["geom"] = self.geometry[i]
        return item


def collate(items: list[dict | None], max_res: int = 1022, max_atoms: int = 64) -> dict:
    items = [x for x in items if x is not None]
    if not items:
        return {}
    b = len(items)
    L = min(max(len(x["res_idx"]) for x in items), max_res)
    A = min(max(x["graph"]["n_sites"] for x in items), max_atoms)

    out = {
        "res_idx": np.full((b, L), UNK, np.int64),
        "prot_mask": np.zeros((b, L), np.float32),
        "pocket_extra": np.zeros((b, L, POCKET_EXTRA_DIM), np.float32),
        "y": np.zeros(b, np.float32), "weight": np.ones(b, np.float32),
        "rel": np.zeros(b, np.int64),
        "protein_idx": np.zeros(b, np.int64), "pair_idx": np.zeros(b, np.int64),
        "n_rot": np.zeros(b, np.float32), "dG_solv": np.full(b, np.nan, np.float32),
    }
    has_plm = "plm" in items[0]
    if has_plm:
        out["plm"] = np.zeros((b, L, items[0]["plm"].shape[1]), np.float32)

    for i, x in enumerate(items):
        n = min(len(x["res_idx"]), L)
        out["res_idx"][i, :n] = x["res_idx"][:n]
        out["prot_mask"][i, :n] = 1.0
        out["pocket_extra"][i, :n] = x["pocket_extra"][:n]
        if has_plm:
            out["plm"][i, :n] = x["plm"][:n]
        for k in ("y", "weight", "rel", "protein_idx", "pair_idx", "n_rot", "dG_solv"):
            out[k][i] = x[k]

    out |= pad_batch([x["graph"] for x in items], max_atoms=A)

    ## Ligand internal distances for the triangle penalty, from the conformer
    ## ensemble (phase3_ligand_phys -> work/ligand_dist). The store is untruncated
    ## and in build_graph's site order, so [:n, :n] matches a truncated graph.
    ## Where a ligand has no entry the bound is D_MAX, which leaves the penalty
    ## inactive for that ligand rather than wrong.
    from ..physics.terms import D_MAX
    out["lig_dist"] = np.full((b, A, A), D_MAX, np.float32)
    out["has_lig_dist"] = np.zeros(b, np.float32)
    for i, x in enumerate(items):
        m = x.get("lig_dist")
        if m is None:
            continue
        n = min(x["graph"]["n_sites"], A, m.shape[0])
        out["lig_dist"][i, :n, :n] = m[:n, :n]
        out["has_lig_dist"][i] = 1.0

    geoms = [x.get("geom") for x in items]
    if any(g is not None for g in geoms):
        out |= _collate_geometry(geoms, L, A)
    return out


def _collate_geometry(geoms, L: int, A: int) -> dict:
    from ..physics import terms as T
    b = len(geoms)
    n_sites = L * 4
    d = {
        "pocket_target": np.zeros((b, L), np.float32),
        "has_pocket": np.zeros(b, np.float32),
        "dist_bins": np.full((b, n_sites, A), T.FAR_BIN, np.int64),
        "dist_valid": np.zeros((b, n_sites, A), np.float32),
        "true_dist": np.full((b, n_sites, A), 99.0, np.float32),
        "has_geometry": np.zeros(b, np.float32),
    }
    for i, g in enumerate(geoms):
        if g is None:
            continue
        n = min(len(g["pocket"]), L)
        d["pocket_target"][i, :n] = g["pocket"][:n]
        d["has_pocket"][i] = 1.0
        ns = min(g["dist_bins"].shape[0], n_sites)
        na = min(g["dist_bins"].shape[1], A)
        d["dist_bins"][i, :ns, :na] = g["dist_bins"][:ns, :na]
        d["dist_valid"][i, :ns, :na] = g["dist_valid"][:ns, :na]
        d["true_dist"][i, :ns, :na] = np.minimum(g["dist"][:ns, :na], 99.0)
        d["has_geometry"][i] = 1.0
    return d


def to_torch(batch: dict, device: str = "cpu") -> dict:
    import torch
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.as_tensor(v).to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def make_loader(ds: PairDataset, batch_size: int, shuffle: bool, workers: int = 4,
                max_res: int = 1022, max_atoms: int = 64, seed: int = 0):
    import torch
    from torch.utils.data import DataLoader

    class _Wrap(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            return ds[i]

    return DataLoader(_Wrap(), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, drop_last=False,
                      persistent_workers=workers > 0,
                      collate_fn=lambda xs: collate(xs, max_res, max_atoms),
                      generator=torch.Generator().manual_seed(seed))


ATOM_FEATURE_DIM = ATOM_DIM
BOND_FEATURE_DIM = BOND_DIM
