"""Shared helpers for the benchmark builder pipeline."""
import os
import sys
import yaml

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg):
    for key in ("work_dir", "output_dir"):
        os.makedirs(cfg["paths"][key], exist_ok=True)


def sanitize_seq(seq):
    """Uppercase and map any non-standard residue to X (mmseqs/BLOSUM safe)."""
    return "".join(ch if ch in STD_AA else "X" for ch in str(seq).upper())


def split_chains(seq, min_len):
    """A target sequence may contain multiple chains separated by ':'."""
    return [c.strip() for c in str(seq).split(":") if len(c.strip()) >= min_len]


def write_fasta(id_seq_pairs, path):
    with open(path, "w") as f:
        for sid, seq in id_seq_pairs:
            f.write(f">{sid}\n{seq}\n")


def log(msg):
    print(msg, flush=True)
