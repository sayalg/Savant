"""Pocket head: which residues contact the ligand? (PLAN.md Section 6.1)

Two jobs:

1. **Supervised prediction.** Residues with any heavy atom within 8 A of the ligand
   in a crystal structure are positives. Section 9 scores this with AUROC and
   precision@K, and Section 6.7's `esm2_pocket_window` mode uses the prediction to
   mask the PLM features.
2. **Hierarchical pruning.** Only the top-K residues enter the pair module, which
   turns an O(L x A) problem into O(K x A) and makes a 1,022-residue protein
   affordable. The selection is *hard and detached* - gradients reach the pair
   module through the selected embeddings, not through the ranking - because a soft
   top-K would let the affinity loss quietly re-weight which residues are "the
   pocket" to fit a label.

The head is ligand-conditioned: the same protein has different pockets for
different ligands, and a ligand-blind pocket head would be one more way for a model
to look protein-aware while ranking nothing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PocketHead(nn.Module):
    def __init__(self, prot_dim: int, lig_dim: int, extra_dim: int = 0,
                 hidden: int = 128, dropout: float = 0.1, topk: int = 64):
        super().__init__()
        self.topk = topk
        self.lig_proj = nn.Linear(lig_dim, hidden)
        self.net = nn.Sequential(
            nn.Linear(prot_dim + hidden + extra_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, prot: torch.Tensor, lig: torch.Tensor, lig_mask: torch.Tensor,
                prot_mask: torch.Tensor, extra: torch.Tensor | None = None):
        """prot (B,L,prot_dim), lig (B,A,lig_dim) -> logits (B,L).

        `prot` is deliberately the PLM-FREE residue representation: the pocket
        selection then cannot be driven by protein-family identity, which is the
        shortcut Section 6.7 warns about, and it also breaks the circular dependency
        in `esm2_pocket_window` mode (the window is needed before the trunk runs).
        """
        pooled = (lig * lig_mask.unsqueeze(-1)).sum(1) / lig_mask.sum(1, keepdim=True).clamp(min=1)
        ctx = self.lig_proj(pooled).unsqueeze(1).expand(-1, prot.shape[1], -1)
        parts = [prot, ctx] + ([extra] if extra is not None else [])
        logits = self.net(torch.cat(parts, -1)).squeeze(-1)
        return logits.masked_fill(~prot_mask.bool(), -1e4)

    def select(self, logits: torch.Tensor, prot_mask: torch.Tensor, k: int | None = None):
        """Hard, detached top-K. Returns (idx (B,K), mask (B,K), window (B,L))."""
        k = min(k or self.topk, logits.shape[1])
        idx = logits.detach().topk(k, dim=-1).indices.sort(dim=-1).values
        mask = torch.gather(prot_mask, 1, idx)
        window = torch.zeros_like(prot_mask, dtype=logits.dtype)
        window.scatter_(1, idx, 1.0)
        return idx, mask, window


def pocket_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                pos_weight: float = 8.0) -> torch.Tensor:
    """BCE over real residues only.

    Pockets are ~5% of a sequence, so an unweighted BCE is minimised by predicting
    "no residue is in the pocket" and the head never learns anything.
    """
    pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    ls = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype),
                                            reduction="none", pos_weight=pw)
    m = mask.to(ls.dtype)
    return (ls * m).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def pocket_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                   k: int = 32) -> dict:
    """AUROC and precision@K against crystal contacts (Section 9)."""
    m = mask.bool()
    s, t = logits[m].float(), target[m].float()
    if t.sum() == 0 or t.sum() == t.numel():
        return {}
    order = torch.argsort(s, descending=True)
    ts = t[order]
    ## AUROC via the rank-sum identity; no sklearn dependency inside the train loop.
    ranks = torch.arange(1, len(ts) + 1, device=ts.device, dtype=torch.float32)
    npos, nneg = ts.sum(), (1 - ts).sum()
    auroc = ((len(ts) - ranks + 1) * ts).sum() / (npos * nneg) - (npos + 1) / (2 * nneg)
    kk = min(k, len(ts))
    return {"pocket/auroc": float(auroc.clamp(0, 1)),
            "pocket/prec_at_k": float(ts[:kk].mean()),
            "pocket/n_pos": int(npos)}
