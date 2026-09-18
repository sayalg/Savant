#!/bin/bash
#SBATCH --job-name=bench-plapt
#SBATCH --output=logs/plapt_%A_%a.out
#SBATCH --error=logs/plapt_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU
#SBATCH --array=0
#
# PLAPT over the benchmark, from a plain checkout in the savant2 conda env.
#
#   sbatch sbatch/run_plapt.sbatch
#
# ONE array task is the default and is enough. PLAPT caches embeddings by string
# in a diskcache, and the benchmark has ~20 unique proteins across 500 pairs, so
# each protein is encoded once and the rest is a ChemBERTa pass plus a small ONNX
# head. Sharding would re-encode every protein in every task and race on the
# shared cache. If you do raise --array, give each task its own --cache-dir.
#
# PLAPT needs no structure, so this does not depend on the Boltz stage and can
# run at any point before 06_merge_scores.py.
#
# ---------------------------------------------------------------------------
# ONE-TIME SETUP (login node, needs outbound network)
#
#   git clone https://github.com/Bindwell/PLAPT.git ~/sw/PLAPT
#   conda activate savant2
#   pip install onnxruntime diskcache
#
# If you clone somewhere else, EXPORT the location (a bare assignment is not
# visible to the preflight's child process):
#
#   PLAPT_DIR=$HOME/somewhere/PLAPT sbatch sbatch/run_plapt.sbatch
#
# That is the whole install. plapt.py imports only torch, transformers,
# onnxruntime, numpy, diskcache and tqdm - savant2 already has all but the middle
# two. PLAPT's requirements.txt additionally lists datasets, evaluate, accelerate,
# biopython and pillow, which are for their training and notebook code, not for
# inference, and are deliberately not installed here.
#
# The 5.9 MB ONNX head is committed to the repo directly (not Git LFS), so the
# clone is self-contained.
#
# Then warm the HuggingFace cache while you still have network, because the
# compute nodes usually do not:
#
#   python -c "
#   from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
#   BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
#   BertModel.from_pretrained('Rostlab/prot_bert')
#   RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
#   RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
#   print('weights cached')"
# ---------------------------------------------------------------------------

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# --- environment -----------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-savant2}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# --- the benchmark being evaluated ----------------------------------------
export BENCHMARK_CSV="${BENCHMARK_CSV:-$(pwd)/../output/savant_benchmark.csv}"
export EVAL_WORK="${EVAL_WORK:-$(pwd)/work}"

# --- where plapt.py and models/affinity_predictor.onnx live ----------------
# `export`, not a bare assignment: the preflight below runs in a child python
# process, which inherits only exported variables. A plain shell variable leaves
# os.environ["PLAPT_DIR"] empty there, and the check then reports "missing
# plapt.py" with no directory in front of it.
#
# The ${VAR/#\~/$HOME} substitution expands a leading tilde. Bash does NOT expand
# `~` inside "${VAR:-~/sw/PLAPT}" or in a quoted value, so a tilde set that way
# would otherwise be taken literally and resolve to <cwd>/~/sw/PLAPT.
PLAPT_DIR="${PLAPT_DIR:-$HOME/sw/PLAPT}"
export PLAPT_DIR="${PLAPT_DIR/#\~/$HOME}"

# Compute nodes are usually offline; fail loudly on a cache miss rather than
# hanging on a download that cannot succeed.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "env       ${CONDA_ENV}"
echo "plapt     ${PLAPT_DIR}"
echo "benchmark ${BENCHMARK_CSV}"
echo "work      ${EVAL_WORK}"
echo "hf cache  ${HF_HOME} (offline=${HF_HUB_OFFLINE})"

# --- preflight: fail here, with a fixable message, not mid-run -------------
python - <<'PY' || { echo "preflight failed - see above" >&2; exit 1; }
import importlib, os, sys

missing = []
for mod, hint in [("torch", "already in savant2"),
                  ("transformers", "already in savant2"),
                  ("numpy", "already in savant2"),
                  ("tqdm", "already in savant2"),
                  ("onnxruntime", "pip install onnxruntime"),
                  ("diskcache", "pip install diskcache")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append((mod, hint))
if missing:
    print("missing python packages:", file=sys.stderr)
    for m, h in missing:
        print(f"  {m:14s} -> {h}", file=sys.stderr)
    sys.exit(1)

raw = os.environ.get("PLAPT_DIR", "")
if not raw:
    print("PLAPT_DIR is empty in this process - it must be exported, not just "
          "assigned, for the preflight to see it.", file=sys.stderr)
    sys.exit(1)
d = os.path.abspath(os.path.expanduser(raw))
for p in (os.path.join(d, "plapt.py"),
          os.path.join(d, "models", "affinity_predictor.onnx")):
    if not os.path.isfile(p):
        print(f"missing {p}   (PLAPT_DIR={raw!r})", file=sys.stderr)
        if os.path.isdir(d):
            print(f"  {d} exists and contains: {sorted(os.listdir(d))[:20]}",
                  file=sys.stderr)
        else:
            print(f"  {d} does not exist", file=sys.stderr)
        print(f"  git clone https://github.com/Bindwell/PLAPT.git {d}",
              file=sys.stderr)
        sys.exit(1)
print("plapt dir", d, "ok")

import numpy, onnxruntime, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("onnxruntime", onnxruntime.__version__, "numpy", numpy.__version__)
PY

DEVICE="cuda"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "no usable GPU; falling back to CPU"; DEVICE="cpu"; }

python scripts/09_run_plapt.py \
    --plapt-dir "${PLAPT_DIR}" \
    --device "${DEVICE}" \
    --num-shards "${NUM_SHARDS:-1}"

echo "done. Next: python scripts/06_merge_scores.py"
