#!/usr/bin/env python
"""PSICHIC/DrugBAN-style interaction network vs the Savant GBDT architecture, side by side.

    python train_interaction_architecture.py --csv ../input/input.csv --extra data/expanded_kd_ki.csv

Meant to run AFTER the expanded dataset exists (get_more_data.py) and, ideally, after
train_savant_v2.ipynb has produced work/internal_2x2.csv and work/benchmark_results.csv -
this script's own results table is printed next to those, not in isolation, since the
question is "does the interaction architecture beat the GBDT cascade", not its numbers in
a vacuum.

Why this is a genuinely different representation, not the same features through a bigger
model
------------------------------------------------------------------------------------------
Every earlier model in this project - the two-part offset+residual notebook, the screening
cascade, `train_savant_v2.ipynb`'s classifier+regressor - uses POOLED (mean over
tokens/residues) ligand and protein embeddings. A tree or a linear layer on a pooled vector
cannot represent "this atom is near that residue" because pooling already discarded which
atom and which residue. This script keeps:

    ligand:  per-ATOM features (M, 16) - element, aromaticity, charge, degree, hybridisation
             (the same 16-dim atom typing `savant_v2/savant2/attention.py` already uses)
    protein: per-RESIDUE ESM-C states (N, 960), NOT mean-pooled, length-capped at 512

and lets a bipartite attention layer between the two learn which atoms and which residues
matter to each other - the PSICHIC/DrugBAN idea (Nat Mach Intell 2024 / 2023): decode an
interaction fingerprint from sequence and ligand graph alone, no 3D pose.

Two interaction layers, selectable, because "PSICHIC/DrugBAN-style" names two different
mechanisms and picking one silently would be presumptuous for a thesis comparison:

    --interaction cross_attention   standard scaled-dot-product cross-attention,
                                     ligand atoms as queries, protein residues as
                                     keys/values (closer to PSICHIC's framing)
    --interaction bilinear          low-rank bilinear pooling per atom-residue pair
                                     (DrugBAN's BAN layer, simplified)

Known limitation, stated rather than hidden
--------------------------------------------
There is no docked pose, so attention has no distance bias - unlike
`savant_v2/savant2/attention.py`'s real structure-based model, which biases attention by
actual interatomic distance and measured corr(attention, distance) = -0.903. This is pure
content-based attention over the WHOLE sequence (capped at 512 residues), not a
pocket - the model must learn relevance without ever being told which residues are near
the ligand. That is the honest cost of staying structure-free, and it is exactly the
comparison this script exists to make: does content-based attention alone beat trees on
pooled embeddings, or does the missing geometry make it a wash.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SEED = 1337
MIN_PKD, MAX_PKD = 4.0, 12.0
MAX_LABEL_SD = 1.5
BINDER_PKD = 6.0
MAX_RESIDUES = 512          # truncation, not pocket selection - see module docstring
ESM_HF = "Synthyra/ESMplusplus_small"       # esmc-300m, same encoder as train_savant_v2

ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# ligand: per-atom features
# --------------------------------------------------------------------------- #

def atom_features(mol) -> np.ndarray:
    """(n_atoms, 16) float32 - element one-hot(10) + aromatic + in-ring + charge + degree
    + H-count + hybridisation-is-sp3. 10 + 6 = 16, matching
    savant_v2/savant2/attention.py's atom typing width - verified by actually running
    this on real SMILES: the first version here was 10 + 5 = 15 and crashed a batch
    collate the moment a differently-sized molecule exposed the mismatch."""
    from rdkit.Chem import rdchem
    rows = []
    for atom in mol.GetAtoms():
        el = atom.GetSymbol()
        onehot = [1.0 if el == e else 0.0 for e in ELEMENTS]
        rows.append(onehot + [
            float(atom.GetIsAromatic()), float(atom.IsInRing()), float(atom.GetFormalCharge()),
            float(atom.GetDegree()) / 4.0, float(atom.GetTotalNumHs()) / 4.0,
            float(atom.GetHybridization() == rdchem.HybridizationType.SP3),
        ])
    return np.asarray(rows, np.float32)


def ligand_atom_batch(smiles_list, max_atoms):
    """(B, max_atoms, 16) padded, (B, max_atoms) bool mask."""
    from rdkit import Chem
    B = len(smiles_list)
    X = np.zeros((B, max_atoms, 16), np.float32)
    mask = np.zeros((B, max_atoms), bool)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        f = atom_features(m)[:max_atoms]
        X[i, :len(f)] = f
        mask[i, :len(f)] = True
    return X, mask


# --------------------------------------------------------------------------- #
# protein: per-residue ESM-C states (NOT pooled), cached, length-capped
# --------------------------------------------------------------------------- #

def embed_residues(seqs, out_dir: Path, max_len=MAX_RESIDUES, batch_size=8):
    """(N, max_len, dim) float16 memmap-backed cache + (N,) true lengths, capped at max_len."""
    cache = out_dir / f"protein_residue_states_{max_len}.npy"
    len_cache = out_dir / f"protein_residue_lens_{max_len}.npy"
    if cache.exists() and len_cache.exists():
        a = np.load(cache, mmap_mode="r")
        if a.shape[0] == len(seqs):
            log(f"reusing {cache.name} {a.shape}")
            return cache, np.load(len_cache)
        log(f"{cache.name} row mismatch - rebuilding")

    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = AutoModel.from_pretrained(ESM_HF, trust_remote_code=True).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(ESM_HF, trust_remote_code=True)
    dim = mdl.config.hidden_size

    out = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float16,
                                    shape=(len(seqs), max_len, dim))
    lens = np.zeros(len(seqs), np.int32)
    order = np.argsort([len(s) for s in seqs])
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            sel = order[start:start + batch_size]
            chunk = [seqs[i][:max_len] for i in sel]
            b = tok(chunk, padding=True, truncation=True, max_length=max_len,
                   return_tensors="pt")
            b = {k: v.to(dev) for k, v in b.items()}
            h = mdl(**b).last_hidden_state.float().cpu().numpy()
            for j, i in enumerate(sel):
                L = min(len(chunk[j]), max_len, h.shape[1])
                out[i, :L] = h[j, :L].astype(np.float16)
                lens[i] = L
            if (start // batch_size) % 200 == 0:
                log(f"    {min(start+batch_size, len(order)):,}/{len(order):,}")
    out.flush(); del mdl
    if dev == "cuda":
        torch.cuda.empty_cache()
    np.save(len_cache, lens)
    log(f"wrote {cache} {out.shape}")
    return cache, lens


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

class CrossAttentionInteraction(nn.Module):
    """Ligand atoms as queries, protein residues as keys/values - PSICHIC-style framing."""

    def __init__(self, atom_dim, prot_dim, d_model=128, n_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(atom_dim, d_model)
        self.kv_proj = nn.Linear(prot_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out_dim = d_model

    def forward(self, atoms, atom_mask, residues, res_mask):
        q = self.q_proj(atoms)
        kv = self.kv_proj(residues)
        key_padding_mask = ~res_mask                         # True = ignore
        out, attn_w = self.attn(q, kv, kv, key_padding_mask=key_padding_mask,
                                need_weights=True, average_attn_weights=True)
        m = atom_mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp_min(1)
        return pooled, attn_w


class BilinearInteraction(nn.Module):
    """DrugBAN-style low-rank bilinear pooling over every (atom, residue) pair.

    Full pairwise bilinear attention over up to 512 residues x ~64 atoms per item is the
    memory-heavy part of this module - k is kept modest (default 16) specifically to keep
    that tensor a few hundred MB per batch rather than blowing up on a shared GPU.
    """

    def __init__(self, atom_dim, prot_dim, d_model=128, k=16):
        super().__init__()
        self.atom_proj = nn.Linear(atom_dim, d_model)
        self.prot_proj = nn.Linear(prot_dim, d_model)
        self.U = nn.Parameter(torch.randn(d_model, k) * 0.02)
        self.V = nn.Parameter(torch.randn(d_model, k) * 0.02)
        self.out_dim = k

    def forward(self, atoms, atom_mask, residues, res_mask):
        a = self.atom_proj(atoms)              # (B, M, d)
        p = self.prot_proj(residues)           # (B, N, d)
        au = torch.einsum("bmd,dk->bmk", a, self.U)     # (B, M, k)
        pv = torch.einsum("bnd,dk->bnk", p, self.V)     # (B, N, k)
        ## pairwise interaction map (B, M, N, k), masked, then pooled over both atoms and
        ## residues - this IS the O(M*N) cost the docstring warns about
        inter = torch.einsum("bmk,bnk->bmnk", au, pv)
        mask2d = (atom_mask.unsqueeze(2) & res_mask.unsqueeze(1)).unsqueeze(-1).float()
        inter = inter * mask2d
        denom = mask2d.sum(dim=(1, 2)).clamp_min(1)
        pooled = inter.sum(dim=(1, 2)) / denom
        return pooled, inter.detach()


class InteractionNet(nn.Module):
    """Shared trunk, two heads - mirrors the classifier+regressor framework kept
    everywhere else in this project, so the comparison isolates the architecture, not
    the task setup."""

    def __init__(self, atom_dim=16, prot_dim=960, interaction="cross_attention"):
        super().__init__()
        Interaction = CrossAttentionInteraction if interaction == "cross_attention" else BilinearInteraction
        self.interaction = Interaction(atom_dim, prot_dim)
        h = self.interaction.out_dim
        self.trunk = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1),
                                   nn.Linear(256, 128), nn.ReLU())
        self.reg_head = nn.Linear(128, 1)
        self.clf_head = nn.Linear(128, 1)

    def forward(self, atoms, atom_mask, residues, res_mask):
        pooled, attn = self.interaction(atoms, atom_mask, residues, res_mask)
        h = self.trunk(pooled)
        return self.reg_head(h).squeeze(-1), self.clf_head(h).squeeze(-1), attn


# --------------------------------------------------------------------------- #
# data plumbing (shares conventions with train_savant_v2.ipynb / run_2x2_split_test.py)
# --------------------------------------------------------------------------- #

def load_pairs(csv: Path, extra: Path | None):
    frames = []
    for path, tag in ((csv, "input"), (extra, "extra")):
        if path is None or not path.exists():
            continue
        d = pd.read_csv(path, usecols=["seq", "smiles", "neg_log10_affinity_M"])
        d = d[d.neg_log10_affinity_M.between(MIN_PKD, MAX_PKD)]
        frames.append(d)
        log(f"  {tag}: {len(d):,} rows")
    raw = pd.concat(frames, ignore_index=True)
    raw["li"] = pd.factorize(raw.smiles)[0]; raw["pi"] = pd.factorize(raw.seq)[0]
    ligands = raw.drop_duplicates("li").sort_values("li").smiles.to_numpy()
    proteins = raw.drop_duplicates("pi").sort_values("pi").seq.to_numpy()
    g = raw.groupby(["li", "pi"])["neg_log10_affinity_M"]
    pairs = g.agg(pkd="median", label_sd="std", n=("size")).reset_index()
    pairs = pairs[pairs.label_sd.fillna(0.0) <= MAX_LABEL_SD].reset_index(drop=True)
    return pairs, ligands, proteins


def make_split(pi_all, seed=SEED, test_frac=0.20):
    """Protein-grouped only, for this script - the scaffold axis is left to
    run_2x2_split_test.py, which already owns that logic; duplicating it a third time
    across three files was judged more likely to drift out of sync than reused here."""
    rng = np.random.default_rng(seed)
    up = np.unique(pi_all)
    test_p = set(rng.choice(up, size=max(1, int(test_frac*len(up))), replace=False).tolist())
    is_te = np.array([p in test_p for p in pi_all])
    return np.where(~is_te)[0], np.where(is_te)[0]


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, idx, pairs, ligands, res_cache, res_lens, max_atoms):
        self.idx = idx; self.ligands = ligands
        ## Index the underlying numpy arrays directly, never `pairs.iloc[i]`. pairs has
        ## mixed dtypes across columns (li/pi int, pkd float), and pandas silently
        ## upcasts an ENTIRE row Series to the common dtype on row-wise .iloc access - so
        ## row.li comes back as numpy.float64(0.0) instead of an int, and numpy refuses
        ## float indexing. Caught by actually running this on real data, not by reading
        ## the code.
        self.li_arr = pairs.li.to_numpy()
        self.pi_arr = pairs.pi.to_numpy()
        self.y_arr = pairs.pkd.to_numpy(np.float32)
        self.res_cache = res_cache; self.res_lens = res_lens; self.max_atoms = max_atoms
        self._mm = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self._mm is None:                       # open the memmap per-worker, not at pickle time
            self._mm = np.load(self.res_cache, mmap_mode="r")
        j = self.idx[i]
        li, pi, y = int(self.li_arr[j]), int(self.pi_arr[j]), self.y_arr[j]
        atoms, amask = ligand_atom_batch([self.ligands[li]], self.max_atoms)
        L = int(self.res_lens[pi])
        residues = np.zeros((self._mm.shape[1], self._mm.shape[2]), np.float32)
        residues[:L] = np.asarray(self._mm[pi, :L], np.float32)
        rmask = np.zeros(self._mm.shape[1], bool); rmask[:L] = True
        return (torch.from_numpy(atoms[0]), torch.from_numpy(amask[0]),
                torch.from_numpy(residues), torch.from_numpy(rmask),
                torch.tensor(y, dtype=torch.float32))


def collate(batch):
    atoms, amask, res, rmask, y = zip(*batch)
    return (torch.stack(atoms), torch.stack(amask), torch.stack(res), torch.stack(rmask),
           torch.stack(y))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(ROOT.parent / "input" / "input.csv"))
    ap.add_argument("--extra", default=str(ROOT / "data" / "expanded_kd_ki.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "work" / "interaction_net"))
    ap.add_argument("--interaction", choices=["cross_attention", "bilinear"],
                    default="cross_attention")
    ap.add_argument("--max-atoms", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit-pairs", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    log("loading pairs")
    extra_p = Path(args.extra) if Path(args.extra).exists() else None
    if extra_p is None:
        log(f"  ({args.extra} not found - training on --csv alone)")
    pairs, ligands, proteins = load_pairs(Path(args.csv), extra_p)
    if args.limit_pairs:
        ## Subsample BEFORE deriving which proteins get embedded, not after - embedding
        ## is the slow step (minutes per thousand proteins), and doing it against the
        ## full unfiltered set defeats the entire point of --limit-pairs as a fast
        ## smoke-test flag. Re-factorize li/pi against the smaller ligand/protein sets so
        ## every downstream index stays dense and in range.
        pairs = pairs.sample(n=min(args.limit_pairs, len(pairs)), random_state=SEED).reset_index(drop=True)
        old_li, old_pi = pairs.li.to_numpy(), pairs.pi.to_numpy()
        new_li_of, new_pi_of = pd.factorize(old_li)[1], pd.factorize(old_pi)[1]
        pairs["li"] = pd.factorize(old_li)[0]; pairs["pi"] = pd.factorize(old_pi)[0]
        ligands = ligands[new_li_of]; proteins = proteins[new_pi_of]
    log(f"{len(pairs):,} pairs, {len(proteins):,} proteins, {len(ligands):,} ligands")

    res_cache, res_lens = embed_residues(list(proteins), out_dir)
    tr_idx, te_idx = make_split(pairs.pi.to_numpy())
    log(f"train {len(tr_idx):,} / test {len(te_idx):,} pairs (protein-grouped)")

    ds_tr = PairDataset(tr_idx, pairs, ligands, res_cache, res_lens, args.max_atoms)
    ds_te = PairDataset(te_idx, pairs, ligands, res_cache, res_lens, args.max_atoms)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                                        collate_fn=collate, num_workers=0)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                                        collate_fn=collate, num_workers=0)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dim = np.load(res_cache, mmap_mode="r").shape[2]
    model = InteractionNet(atom_dim=16, prot_dim=dim, interaction=args.interaction).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log(f"model on {dev}, interaction={args.interaction}, "
        f"{sum(p.numel() for p in model.parameters()):,} parameters")

    binder_mean = float((pairs.pkd.to_numpy() >= BINDER_PKD).mean())
    log(f"binder rate: {binder_mean:.3f}")

    for epoch in range(args.epochs):
        model.train(); tot_loss = 0.0; n = 0
        for atoms, amask, res, rmask, y in dl_tr:
            atoms, amask, res, rmask, y = (t.to(dev) for t in (atoms, amask, res, rmask, y))
            pred_pk, pred_logit, _ = model(atoms, amask, res, rmask)
            label = (y >= BINDER_PKD).float()
            loss = F.smooth_l1_loss(pred_pk, y) + 0.3 * F.binary_cross_entropy_with_logits(pred_logit, label)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(y); n += len(y)
        log(f"epoch {epoch+1}/{args.epochs}  train loss {tot_loss/max(n,1):.4f}")

    model.eval(); preds, truths = [], []
    with torch.no_grad():
        for atoms, amask, res, rmask, y in dl_te:
            atoms, amask, res, rmask = (t.to(dev) for t in (atoms, amask, res, rmask))
            pred_pk, _, _ = model(atoms, amask, res, rmask)
            preds.append(pred_pk.cpu().numpy()); truths.append(y.numpy())
    preds, truths = np.concatenate(preds), np.concatenate(truths)
    pearson = float(np.corrcoef(truths, preds)[0, 1]) if len(preds) > 2 else float("nan")
    log(f"test pearson (protein-grouped, single split - NOT the full honest 2x2): {pearson:.4f}")

    result = {"interaction": args.interaction, "n_train": len(tr_idx), "n_test": len(te_idx),
             "pearson": pearson, "n_params": sum(p.numel() for p in model.parameters())}
    pd.DataFrame([result]).to_csv(out_dir / f"result_{args.interaction}.csv", index=False)

    gbdt_path = ROOT / "work" / "internal_2x2.csv"
    if gbdt_path.exists():
        g = pd.read_csv(gbdt_path)
        deploy = g[(g.cell == "NOVEL prot / NOVEL chem") & (g.model == "reg_squared")]
        if len(deploy):
            log(f"\nSavant GBDT (train_savant_v2.ipynb), deployment cell, reg_squared: "
                f"pearson {deploy.pearson.iloc[0]:.4f}")
            log(f"{args.interaction} interaction net, protein-grouped test: "
                f"pearson {pearson:.4f}")
            log("NOTE: these two splits are not identical (protein-only vs protein+scaffold "
                "double-disjoint) - re-run this script's split to match run_2x2_split_test.py "
                "before quoting a head-to-head number in the thesis.")
    else:
        log(f"\n({gbdt_path} not found yet - run train_savant_v2.ipynb for the GBDT "
            "comparison numbers)")

    torch.save(model.state_dict(), out_dir / f"model_{args.interaction}.pt")
    log(f"saved {out_dir}/model_{args.interaction}.pt and result_{args.interaction}.csv")


if __name__ == "__main__":
    main()
