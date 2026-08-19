"""ESM-2 embeddings of the binding pocket.

The geometric features say *what* the interface does - how many hydrophobic
contacts, which residue types line the site. They do not say what the protein
*is*. A protein language model does: ESM-2 per-residue embeddings carry
evolutionary and structural context that residue-type counts cannot.

Scoped to the pocket, not the whole chain, for two reasons. Earlier on this
benchmark, whole-protein amino-acid composition added +0.012 Pearson r - global
identity is nearly useless for affinity, because binding happens in one small
site. And a pocket of ~40 residues keeps the cross-attention model's sequence
length small enough to train quickly and to interpret residue by residue.

Compute is amortised over unique sequences: 47,619 complexes cover only 11,965
proteins, so per-residue embeddings are cached by sequence hash and every
complex sharing that protein slices its own pocket out of the same matrix.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from savant2.structure import Complex, residue_keys

# facebook/esm2_t30_150M_UR50D is the default: 640 dimensions, ~150M params,
# a few ms per sequence on a GPU and still tractable on CPU. The 650M model
# (t33, 1280-d) is better but doubles storage and inference cost; set
# ESM_MODEL to switch.
DEFAULT_ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
MAX_POCKET_RESIDUES = 48
POCKET_CUTOFF = 8.0


def sequence_id(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


def pocket_residues(
    cx: Complex, cutoff: float = POCKET_CUTOFF, max_residues: int = MAX_POCKET_RESIDUES
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residues lining the pocket, nearest first.

    Returns ``(sequence_positions, residue_names, min_distances)``.

    ``sequence_positions`` are 0-based indices into the protein sequence. AF3 and
    Boltz both number residues 1..N against the sequence they were given, so
    ``res_id - 1`` is the sequence index; a structure numbered otherwise (a PDB
    entry with author numbering, say) would need remapping first.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(cx.l_xyz)
    dist, _ = tree.query(cx.p_xyz, k=1)
    keys = residue_keys(cx)

    best: Dict[str, Tuple[float, int, str]] = {}
    for d, key, rid, rname in zip(dist, keys, cx.p_res_id, cx.p_res_name):
        if d > cutoff:
            continue
        if key not in best or d < best[key][0]:
            best[key] = (float(d), int(rid), str(rname))

    if not best:
        return np.zeros(0, int), np.zeros(0, "<U3"), np.zeros(0, np.float32)

    ordered = sorted(best.values(), key=lambda t: t[0])[:max_residues]
    positions = np.array([rid - 1 for _, rid, _ in ordered], dtype=int)
    names = np.array([name for _, _, name in ordered], dtype="<U3")
    dists = np.array([d for d, _, _ in ordered], dtype=np.float32)
    return positions, names, dists


class EsmEmbedder:
    """Lazily loaded ESM-2, with an on-disk per-sequence cache.

    torch and transformers are imported only when this is constructed, so the
    geometric feature path and the CLI keep working in an environment without
    them.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ESM_MODEL,
        cache_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
        max_length: int = 1024,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)

    def _cache_path(self, seq: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{sequence_id(seq)}.npy"

    def embed(self, seq: str) -> np.ndarray:
        """Per-residue embeddings, shape ``(len(seq), dim)``, float16.

        Sequences longer than ``max_length`` are embedded in overlapping windows
        and stitched, so a 1951-residue target does not silently get truncated
        to its first 1024 residues - which would drop the pocket entirely for
        any C-terminal site.
        """
        path = self._cache_path(seq)
        if path is not None and path.exists():
            return np.load(path)

        chunks: List[np.ndarray] = []
        stride = self.max_length - 128           # overlap, trimmed on stitch
        starts = list(range(0, max(len(seq), 1), stride)) or [0]
        for start in starts:
            window = seq[start:start + self.max_length]
            if not window:
                continue
            tokens = self.tokenizer(window, return_tensors="pt", add_special_tokens=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with self.torch.no_grad():
                out = self.model(**tokens).last_hidden_state[0]
            # Drop the BOS/EOS tokens ESM adds, leaving one row per residue.
            residue_states = out[1:1 + len(window)].float().cpu().numpy()
            keep_from = 0 if start == 0 else 64
            chunks.append(residue_states[keep_from:])
            if start + self.max_length >= len(seq):
                break

        embedding = np.concatenate(chunks, axis=0)[:len(seq)].astype(np.float16)
        if path is not None:
            np.save(path, embedding)
        return embedding

    def pocket_embedding(
        self, seq: str, positions: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(per_residue, pooled)`` for the pocket.

        ``per_residue`` is ``(len(positions), dim)`` for the cross-attention
        model; ``pooled`` concatenates mean and max over pocket residues, which
        is what the gradient-boosted models consume.
        """
        full = self.embed(seq)
        positions = np.asarray([p for p in positions if 0 <= p < len(full)], dtype=int)
        if len(positions) == 0:
            return (np.zeros((0, self.dim), np.float16),
                    np.zeros(2 * self.dim, np.float32))
        per_residue = full[positions]
        pooled = np.concatenate(
            [per_residue.astype(np.float32).mean(axis=0),
             per_residue.astype(np.float32).max(axis=0)]
        )
        return per_residue, pooled


def pooled_feature_names(dim: int) -> List[str]:
    return ([f"esm_mean_{i}" for i in range(dim)] +
            [f"esm_max_{i}" for i in range(dim)])
