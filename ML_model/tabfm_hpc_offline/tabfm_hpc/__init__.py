"""tabfm_hpc: run Google's TabFM offline on an HPC cluster.

Split into two phases so it survives an air-gapped compute node:
  1. download_weights.py  (online, login/transfer node, once)
  2. run_tabfm.py         (offline, compute node, per dataset)
"""
from .offline import go_offline, resolve_device

__all__ = ["go_offline", "resolve_device"]
__version__ = "0.1.0"
