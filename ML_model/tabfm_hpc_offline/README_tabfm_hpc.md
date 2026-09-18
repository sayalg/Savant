# TabFM on HPC (offline-first)

A thin, HPC-friendly wrapper around Google's
[`google/tabfm-1.0.0-pytorch`](https://huggingface.co/google/tabfm-1.0.0-pytorch)
zero-shot tabular foundation model. It solves the thing that actually bites you
on a cluster: **the model wants to download ~13 GB of weights from Hugging Face
at runtime, but compute nodes have no internet.**

> Named `README_tabfm_hpc.md` so it doesn't collide with anything else in
> `ML_model/`. The wrapper lives alongside your existing TabFM notebooks
> (`train_tabfm_cla_model.ipynb`, `train_tabfm_reg_model.ipynb`, …) and reuses
> the same prepped CSVs (`combined_50k_filtered_prepped_*.csv`).

The fix is to split the workflow into two phases:

| Phase | Where | Network | Script |
|-------|-------|---------|--------|
| 1. Download weights | login / data-transfer node | **online**, once | `download_weights.py` |
| 2. Train + infer | compute node | **fully offline** | `run_tabfm.py` |

Everything in phase 2 sets `HF_HUB_OFFLINE=1` and reads weights from a local
directory, so a stray `from_pretrained` can never hang your job trying to reach
the internet.

## What TabFM is (the 30-second dissection)

- Zero-shot, in-context tabular model. `fit()` just loads your training rows as
  *context*; there's no gradient training. The prediction is a single forward
  pass. This is why memory scales with the number of training rows.
- Two independent checkpoints, each `config.json` + `model.safetensors`
  (~6.56 GB): `classification/` (≤10 classes) and `regression/`.
- Architecture: per-cell Fourier features → column/set-transformer attention →
  row compression (RoPE, CLS tokens) → a 24-block causal "ICL" transformer
  (256-dim, 8 heads, SwiGLU).
- Loaded via `huggingface_hub`'s `PyTorchModelHubMixin`; wrapped by
  `TabFMClassifier` / `TabFMRegressor` (sklearn-style `fit`/`predict`/`predict_proba`).
- **License:** weights are non-commercial (and the repo may be **gated** — you
  may need to accept the license and authenticate). Code is Apache-2.0.

## Layout (added under `ML_model/`)

```
download_weights.py     phase 1: snapshot_download into a local dir
run_tabfm.py            phase 2: offline train + infer + save
env_setup.sh            create venv, install torch + tabfm
requirements.txt
tabfm_hpc/
  offline.py            forces HF offline + keeps caches off home quota
  model.py              loads TabFM from LOCAL weights (network disabled)
  data.py               CSV/Parquet -> X, y
  save.py               writes logits / proba / predictions / model / metadata
slurm/
  download.slurm        phase 1 as a batch job
  run.slurm             phase 2 as a batch job
examples/
  make_sample_data.py   tiny synthetic dataset to smoke-test the pipeline
```

## Quickstart

### 0. Environment (online node, once)
```bash
bash env_setup.sh                       # edit module/CUDA lines for your site
source ~/tabfm-venv/bin/activate
```

### 1. Download weights (online node, once)
Put weights on **scratch/project storage, never your home dir** (13 GB will
blow a home quota, and the default `~/.cache/huggingface` lives there):
```bash
python download_weights.py --task classification --dest /scratch/$USER/tabfm
```
If the repo is gated (401/403), accept the license on the model page while
logged in, then `hf auth login` (or `export HF_TOKEN=hf_...`) and retry.

### 2. Train + infer (compute node, offline)
Point it straight at your existing prepped data:
```bash
export HF_HUB_OFFLINE=1
python run_tabfm.py \
    --task classification \
    --weights /scratch/$USER/tabfm \
    --train combined_50k_filtered_prepped_binary_classification.csv \
    --test  combined_50k_filtered_prepped_binary_classification.csv \
    --target <your_label_column> \
    --out results/run1 \
    --device auto
```

### Or via SLURM
```bash
WEIGHTS_DIR=/scratch/$USER/tabfm TASK=classification sbatch slurm/download.slurm
WEIGHTS_DIR=/scratch/$USER/tabfm \
    TRAIN=combined_50k_filtered_prepped_binary_classification.csv \
    TEST=combined_50k_filtered_prepped_binary_classification.csv \
    TARGET=<your_label_column> sbatch slurm/run.slurm
```

### Smoke test (no real data needed)
```bash
python examples/make_sample_data.py --out data
python run_tabfm.py --task classification --weights /scratch/$USER/tabfm \
    --train data/train.csv --test data/test.csv --target label --out results/smoke
```

## Outputs — saved for other frameworks

`run_tabfm.py --out <dir>` writes framework-neutral artifacts:

| File | What it is |
|------|-----------|
| `logits.npy` / `logits.csv` | raw pre-softmax class logits (classification) |
| `proba.npy` / `proba.csv`   | class probabilities |
| `predictions.csv`           | final class label or regressed value |
| `model.joblib`              | the fitted estimator (reload in Python) |
| `backbone_state_dict.pt`    | raw torch weights of the backbone |
| `metadata.json`             | shapes, class order, dtype, library versions |

**Logits** are the key export for downstream use (stacking into XGBoost,
temperature/Platt calibration, ensembling, or feeding a head written in another
library). If the installed TabFM build doesn't expose raw logits, we fall back
to `log(proba)`, which is a mathematically valid softmax pre-image (equal up to
an additive constant) — noted in `metadata.json`.

## HPC gotchas this handles

- **No internet on compute nodes** → all weights local; `HF_HUB_OFFLINE=1` set
  before any HF import so nothing tries (and fails) to phone home mid-job.
- **Home quota** → `HF_HOME` redirected to your weights dir on scratch.
- **Gated / non-commercial weights** → download step supports `HF_TOKEN`.
- **Parallel filesystems (Lustre/GPFS)** → symlink probes/warnings disabled.
- **Only need one task** → download just `classification` *or* `regression`
  (~6.6 GB) instead of the full 13 GB.
- **Memory scales with training rows** → `run.slurm` requests generous `--mem`;
  bump it for large training tables.

## Notes / caveats

- Pin a specific commit for reproducibility: `download_weights.py --revision <sha>`.
- The loader in `tabfm_hpc/model.py` tries the documented entrypoints in order
  (`tabfm_v1_0_0_pytorch.load(checkpoint_path=...)` then
  `TabFM_HF.from_pretrained(local, local_files_only=True)`) because the public
  API has shifted slightly across releases; if a future release renames these,
  that one file is the only place to adjust.
- Non-commercial license applies to the weights and to anything you produce
  with them — check it before using results in a commercial setting.
