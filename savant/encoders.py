"""Mean-pooled language-model embeddings for proteins and ligands.

Both encoders are mean-pooled over the last hidden state with padding masked
out, which is how the training matrix was built. Anything else here -- the fp16
cast, the 1024-token cap -- is also a training detail, not a preference, and
changing it silently shifts every prediction.

`torch` and `transformers` are imported lazily inside the functions so that
importing :mod:`savant` stays cheap: reading a SMILES file or inspecting a
bundle should not pull in a deep-learning stack.
"""
from __future__ import annotations

import functools

import numpy as np

## The ligand encoder is fixed by the trained bundle. The protein encoder is
## not: it is recorded per bundle as `esm_model`, because the two shipped
## variants differ in exactly that.
CHEMBERTA = "DeepChem/ChemBERTa-77M-MLM"

## Training encoded proteins in fp16 on GPU and truncated at 1024 tokens. A
## longer sequence is truncated here too, exactly as it was then.
PROTEIN_MAX_TOKENS = 1024


@functools.lru_cache(maxsize=None)
def load_encoder(name: str, half: bool = False):
    """(tokenizer, model, device), cached so a batch loads the weights once.

    ESM2-650M is ~2.5 GB, so the cache is what makes scoring many ligands
    against one target cheap: the protein encoder is paid for once per process.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
    if half and device == "cuda":
        mdl = mdl.half()          # training encoded in fp16 on GPU
    return tok, mdl, device


def encode(texts, name, batch_size=16, half=False, max_length=None):
    """(n, hidden) mean-pooled last hidden state, ignoring padding.

    The mask matters: mean-pooling over padded positions would make an
    embedding depend on what else happened to be in its batch.
    """
    import torch

    tok, mdl, device = load_encoder(name, half)
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
            hidden = mdl(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            out.append(((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def encode_ligands(smiles, batch_size=64):
    """(n, 384) ChemBERTa-77M-MLM embedding."""
    return encode(list(smiles), CHEMBERTA, batch_size)


def encode_proteins(seqs, esm_model, batch_size=8):
    """(n, 1280) or (n, 320) ESM2 embedding, per the bundle's `esm_model`."""
    return encode(list(seqs), esm_model, batch_size,
                  half=True, max_length=PROTEIN_MAX_TOKENS)
