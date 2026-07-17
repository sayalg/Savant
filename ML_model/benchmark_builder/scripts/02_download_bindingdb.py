#!/usr/bin/env python
"""
Step 02 — Download the BindingDB Kd table.

Pulls the TDC-curated BindingDB "Kd" table (Harvard Dataverse) to
<bindingdb_tsv>. Columns: ID1(drug id), X1(SMILES), ID2(UniProt),
X2(target sequence), Y(Kd in nM).

RUN THIS ON A LOGIN NODE (compute nodes usually have no internet). It is a
one-time ~55 MB download; skipped if the file already exists (use --force to
re-download). If your cluster blocks outbound HTTP entirely, download the file
elsewhere and place it at <bindingdb_tsv> manually.
"""
import argparse
import os
import requests

from common import load_config, ensure_dirs, log


def main(config_path, force):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    dest = cfg["paths"]["bindingdb_tsv"]
    url = cfg["download"]["url"]

    if os.path.exists(dest) and not force:
        log(f"[02] {dest} already exists ({os.path.getsize(dest):,} bytes); skipping. Use --force to re-download.")
        return

    log(f"[02] downloading BindingDB Kd table from {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest + ".part"
        n = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
                n += len(block)
        os.replace(tmp, dest)
    log(f"[02] wrote {dest} ({n:,} bytes)")

    # sanity check header
    with open(dest) as f:
        header = f.readline().strip().split("\t")
    expected = ["ID1", "X1", "ID2", "X2", "Y"]
    if header[:5] != expected:
        log(f"[02] WARNING: unexpected header {header[:8]} (expected {expected})")
    else:
        log("[02] header OK: ID1, X1, ID2, X2, Y")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.config, a.force)
