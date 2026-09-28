"""Frozen per-residue protein embeddings (PLAN.md Section 6.7, Phase 4).

Per-residue, in fp16, sharded - NOT mean-pooled. Pooling was the previous project's
protein representation and it is exactly what the pocket head and the physics need
to not have: a 7,680-dimensional pooled vector tells a model which protein family
it is looking at and nothing about which residues line the pocket.

Two implementation details that cost time before:

* `attn_implementation="eager"` is REQUIRED if contact maps are wanted. transformers
  5.x defaults to SDPA, which returns no attention weights, and ESM's contact head
  then dies with "stack expects a non-empty TensorList" - an error that names
  neither the cause nor the fix.
* sequences are truncated at 1022 tokens. Section 5.2 asks for the policy to be
  documented and metrics reported by length bin; the truncation count is written
  into the cache manifest so the report can state it rather than imply it.

Contact-map centrality is computed here as well. It is the cheap, structure-free
pocket prior that feeds the pocket head's `extra` channel, and it is the fallback
when no structural pocket labels exist for a protein.
"""
from __future__ import annotations

import numpy as np

from ..preflight import pick_device, require_torch

MODELS = {
    "esm2_650m_full": "facebook/esm2_t33_650M_UR50D",
    "esm2_bottleneck": "facebook/esm2_t33_650M_UR50D",
    "esm2_pocket_window": "facebook/esm2_t33_650M_UR50D",
    "esm2_3b": "facebook/esm2_t36_3B_UR50D",
    "esmc_300m": "EvolutionaryScale/esmc-300m-2024-12",
}
MAX_LEN = 1022


def model_id(plm_mode: str) -> str:
    if plm_mode not in MODELS:
        raise SystemExit(f"plm.mode={plm_mode!r} has no cached-embedding model; "
                         f"have {sorted(MODELS)} (or plm.mode=none)")
    return MODELS[plm_mode]


class Embedder:
    def __init__(self, plm_mode: str = "esm2_650m_full", device: str | None = None,
                 want_contacts: bool = True, dtype: str = "float16"):
        require_torch("protein embedding extraction")
        import torch
        from transformers import AutoTokenizer, EsmModel
        self.torch = torch
        self.name = model_id(plm_mode)
        self.device = device or pick_device()
        self.want_contacts = want_contacts
        ## eager attention or no attentions at all - see the module docstring.
        kw = {"attn_implementation": "eager"} if want_contacts else {}
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = EsmModel.from_pretrained(self.name, **kw).to(self.device).eval()
        self.dtype = getattr(torch, dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    ## GPU memory the returned attention tuple may occupy per forward pass. With
    ## contacts on, ESM returns EVERY layer's (B, H, T, T) matrix - 33 x 20 x B x T^2
    ## in fp32, ~22 GB for eight 1,022-residue chains - so batches are capped by T^2,
    ## not just by token count. Override with PFPL_ATTN_BUDGET_GB.
    ATTN_BUDGET_GB = 6.0

    def _attn_bytes(self, batch: int, t: int) -> int:
        c = self.model.config
        return (c.num_hidden_layers * c.num_attention_heads * batch * t * t
                * self.model.dtype.itemsize)

    def _plan(self, seqs: list[str]) -> list[list[int]]:
        """Split into sub-batches whose attention tuple fits the budget."""
        import os
        if not self.want_contacts:
            return [list(range(len(seqs)))]
        budget = float(os.environ.get("PFPL_ATTN_BUDGET_GB", self.ATTN_BUDGET_GB)) * 2**30
        out, cur, tmax = [], [], 0
        for i, s in enumerate(seqs):
            t = max(tmax, len(s) + 2)
            if cur and self._attn_bytes(len(cur) + 1, t) > budget:
                out.append(cur)
                cur, t = [], len(s) + 2
            cur.append(i)
            tmax = t
        return out + ([cur] if cur else [])

    def encode(self, sequences: list[str]) -> list[dict]:
        seqs = [s[:MAX_LEN] for s in sequences]
        res: list[dict | None] = [None] * len(seqs)
        for idx in self._plan(seqs):
            for i, d in zip(idx, self._encode_safe([seqs[i] for i in idx])):
                d["truncated"] = len(sequences[i]) > MAX_LEN
                res[i] = d
        return res

    def _encode_safe(self, seqs: list[str]) -> list[dict]:
        """Encode; on CUDA OOM, halve the batch and retry instead of dying.

        The budget above makes this rare, but other tenants and fragmentation make
        a hard guarantee impossible, and one OOM used to cost the whole shard.
        """
        torch = self.torch
        try:
            return self._encode(seqs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(seqs) == 1:
                raise
            h = len(seqs) // 2
            print(f"    OOM on a batch of {len(seqs)} (max {max(map(len, seqs))} aa) "
                  "- retrying as two halves", flush=True)
            return self._encode_safe(seqs[:h]) + self._encode_safe(seqs[h:])

    def _encode(self, seqs: list[str]) -> list[dict]:
        torch = self.torch
        enc = self.tok(seqs, return_tensors="pt", padding=True, truncation=True,
                       max_length=MAX_LEN + 2)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc, output_attentions=self.want_contacts)
            h = out.last_hidden_state
            res = []
            for i, s in enumerate(seqs):
                n = len(s)
                ## Strip <cls> and <eos>: position 0 is not a residue, and including
                ## it shifts every downstream residue index by one.
                e = h[i, 1:n + 1].to(self.dtype).cpu().numpy()
                d = {"emb": e, "length": n}
                if self.want_contacts:
                    d["contact_centrality"] = self._contacts(out.attentions, i, n)
                res.append(d)
        del out, h, enc
        return res

    def _contacts(self, attentions, i: int, n: int) -> np.ndarray:
        """Per-residue contact centrality from the attention stack.

        Averaging the symmetrised attention over heads and layers, then summing the
        long-range part per residue, is a serviceable structure-free proxy for
        "how buried is this residue" - which is what the pocket head wants as a
        prior. It is not a contact map and is not claimed to be one.

        Accumulated layer by layer for ONE sample: `torch.stack(attentions)` copied
        the whole (L, B, H, T, T) tuple - the 18 GiB allocation behind the OOM.
        """
        torch = self.torch
        a = None
        for layer in attentions:
            x = layer[i, :, 1:n + 1, 1:n + 1].float().mean(0)          # (n, n)
            a = x if a is None else a.add_(x)
        a = a / len(attentions)
        a = (a + a.T) / 2
        sep = (torch.arange(n, device=a.device)[:, None]
               - torch.arange(n, device=a.device)[None, :]).abs()
        a = a * (sep >= 6)                                    # long-range contacts only
        c = a.sum(-1)
        c = (c - c.mean()) / (c.std() + 1e-6)
        return c.to(self.dtype).cpu().numpy()


def pocket_extra_features(centrality: np.ndarray, sequence: str) -> np.ndarray:
    """(L, 4) cheap per-residue prior for the pocket head's `extra` channel.

    Deliberately structure-free and PLM-free: contact centrality, relative position,
    a local hydropathy window, and a local flexibility proxy.
    """
    from ..physics.sites import _KD_HYDRO
    n = len(sequence)
    hyd = np.array([_KD_HYDRO.get(c, 0.0) for c in sequence.upper()], np.float32) / 4.5
    win = np.convolve(hyd, np.ones(9) / 9.0, mode="same")
    pos = np.linspace(0, 1, n, dtype=np.float32)
    flex = np.convolve(np.isin(list(sequence.upper()), list("GSPDNK")).astype(np.float32),
                       np.ones(7) / 7.0, mode="same")
    c = centrality[:n] if centrality is not None and len(centrality) >= n \
        else np.zeros(n, np.float32)
    return np.stack([c, pos, win, flex], -1).astype(np.float32)


POCKET_EXTRA_DIM = 4
