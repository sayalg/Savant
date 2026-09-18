#!/bin/bash
# One-time environment setup. Run on a node WITH internet (login/transfer).
# Creates a venv and installs TabFM + deps. Adjust module names + CUDA to site.
set -euo pipefail

VENV="${VENV:-$HOME/tabfm-venv}"

module load python/3.11 || true   # or: conda / spack / your site's loader

python -m venv "${VENV}"
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel

# 1) Install torch matching your cluster's CUDA. Examples:
#    CUDA 12.4:  pip install torch --index-url https://download.pytorch.org/whl/cu124
#    CPU only:   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2) TabFM + the rest.
pip install -r requirements.txt

echo "Env ready at ${VENV}. Activate with: source ${VENV}/bin/activate"
