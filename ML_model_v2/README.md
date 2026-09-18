# ML_model_v2 — running it all on the HPC

Five pieces, in the order they actually depend on each other. Steps 1 and 3 need network;
everything else is offline. Every script/notebook here is resumable/cacheable — re-running
a step that already has output reuses it rather than redoing the work.

## 0. Environment

One known landmine, already hit once: `transformers>=4.40` (no upper bound) can resolve
against a `huggingface_hub` it's incompatible with, failing on import with
`cannot import name 'is_offline_mode' from 'huggingface_hub'`. If any script below dies on
that import, fix it first:

```bash
pip install -U transformers huggingface_hub
# if that alone doesn't resolve it:
pip install 'transformers==4.46.*' 'huggingface_hub>=0.25,<0.26'
```

`train_interaction_architecture.py` also needs plain PyTorch (already in the `savant2` env).

## 1. Get more Kd/Ki data — login node, needs network

```bash
cd ML_model_v2
python scripts/get_more_data.py chembl          # ~636,684 activities, resumable, ~10-20 min
python scripts/get_more_data.py chembl-targets  # target_chembl_id -> UniProt accession
python scripts/get_more_data.py uniprot         # fetch sequences
python scripts/get_more_data.py merge           # combine, dedupe, exclude eval sets
```

**`data/` currently holds a 300-row smoke test I ran to verify the pipeline** (207 pairs
out of 636,684 possible) — the checkpoint will resume from there automatically, so just run
`chembl` again with no `--limit-rows` and it picks up where the test left off. Nothing to
delete first.

Optionally add BindingDB's Ki data too (Ki was never in `input.csv` — only Kd was):

```bash
pip install PyTDC        # once
python scripts/get_more_data.py bindingdb
python scripts/get_more_data.py merge           # re-run merge to pick it up
```

Output: `data/expanded_kd_ki.csv`. `input.csv` is never modified.

## 2. Train the classifier + regressor — GPU node

```bash
jupyter nbconvert --to notebook --execute train_savant_v2.ipynb --ExecutePreprocessor.timeout=-1
```

Uses `input.csv` + `data/expanded_kd_ki.csv` (falls back to `input.csv` alone with a
warning if step 1 hasn't been run). Encoders are fixed (esmc-300m, ChemBERTa-77M-MLM — the
sweep is done, see `work/part_A_protein_sweep.csv` / `part_B_ligand_ecfp_sweep.csv` from
the prior project for why). Trains a binder classifier and two regressors
(`reg_squared` for absolute pKd, `reg_ranking` for within-target order) under the honest
protein+scaffold double-disjoint 2x2 split, and scores the v3 benchmark.

Output: `work/savant_v2_bundle.pkl`.

## 3. Structure-informed protein embeddings — SLURM array, up to 12 GPUs, needs network once

**Always self-check before submitting the array:**

```bash
python scripts/embed_structure_encoders.py --mode esmfold2 --selfcheck
```

Read its output. It reports a warm-call GPU-hour projection for 21,140 proteins — sanity
check it against ~29 GPU-hours (measured earlier this project at batch 4) before trusting
the array's runtime budget; a batch-size-1 selfcheck runs several times slower per-protein
than a real batched job.

**The manifest is not generated for you — build it yourself, and the order must exactly
match the protein index everything else uses**, since embeddings are assigned to manifest
lines by position (line 1 -> row 0, etc.). A mismatched manifest produces embeddings that
silently attach to the wrong protein everywhere downstream.

If you've run `run_2x2_split_test.py`, its `static.npz` already stores the protein list in
the right order:

```bash
python -c "
import numpy as np
a = np.load('work/static.npz', allow_pickle=True)['proteins']
open('work/protein_manifest.txt','w').write('\n'.join(str(s) for s in a))
"
```

**Do not use `ndarray.tofile(sep='\n')` for this** (an earlier version of this README did).
On object-dtype string arrays — which is what pandas `.to_numpy()` produces — some numpy
versions write `repr()` per element, wrapping every sequence in quotes. The script now
validates the manifest up front and rejects that with an explanatory message, but the
correct command above avoids it entirely.

If you've only run `train_savant_v2.ipynb`, its `static.npz` now saves `proteins` too (a
fix made after the first version of this notebook didn't) — same command, same `work/`
directory. If you're re-using a `static.npz` written before this fix, the notebook
detects the missing key and patches it back in on the next run, no RDKit recompute needed
(watch for `static.npz predates saving proteins/ligands - adding them now` in the log).

`input.csv` contains exactly one contaminated "sequence"
(`biotin-C6linker-VRRLRRLTAREAA`, a modified-peptide annotation in a sequence field,
affecting 34 of 1,836,729 pairs). The manifest validator rejects it by default. Pass
`--on-invalid skip` to proceed: the sequence stays in the manifest (preserving index
alignment), gets a zero embedding, and the merge writes a `<mode>_valid.npy` mask so
downstream code can exclude it rather than mistake a zero vector for a real embedding.

To use it with the sbatch wrapper, set it in the environment:

```bash
ON_INVALID=skip sbatch --array=0-11 sbatch/embed_structure_encoders.sbatch esmfold2 work/protein_manifest.txt
```

Then submit the array:

```bash
sbatch --array=0-11 sbatch/embed_structure_encoders.sbatch esmfold2 work/protein_manifest.txt

# after all 12 tasks finish:
python scripts/embed_structure_encoders.py --mode esmfold2 --manifest work/protein_manifest.txt --merge
```

`saprot` mode exists (structure-aware sequence tokens via Foldseek 3Di) but is
**best-effort, not verified end-to-end** — no Foldseek binary or SaProt weights were
available to test it against in advance. Run its `--selfcheck` first; if it fails, the
error names which step to fix (Foldseek not on `$PATH`, or ESMFold2's fold output not
matching the PDB-export attempts in `_fold_to_pdb`).

Shard count comes from `--array`, never a hardcoded value — that mismatch has already
cost this project one full benchmark run.

## 4. Fair encoder comparison — CPU/single GPU, after step 3 (or standalone)

```bash
python scripts/run_2x2_split_test.py --emb-dir work/emb
```

Auto-discovers every `.npy` in `--emb-dir` that matches the current protein count and
sweeps all of them — including `esmfold2.npy`/`saprot.npy` from step 3 once merged,
alongside anything already cached from earlier work (point `--emb-dir` at
`../ML_model/work_two_part/emb` to include the old esm2/esmc runs in the same table).
Reports both a cheap ridge diagnostic and a real regressor, on all four 2x2 cells.

## 5. Interaction architecture vs the GBDT cascade — GPU, after steps 1 and 2

```bash
python scripts/train_interaction_architecture.py --interaction cross_attention
python scripts/train_interaction_architecture.py --interaction bilinear
```

Per-atom ligand features + per-residue (unpooled) protein embeddings, two selectable
interaction mechanisms (PSICHIC-style cross-attention, DrugBAN-style bilinear). Prints a
side-by-side comparison against step 2's `work/internal_2x2.csv` if it exists — note the
splits aren't identical yet (protein-only here vs protein+scaffold in step 2), so treat
that comparison as indicative until unified.

## 6. Real screening-deck evaluation (LIT-PCBA)

Once step 2 has produced `work/savant_v2_bundle.pkl`:

```bash
cd ../ML_model/litpcba
python prep_litpcba.py score --bundle ../../ML_model_v2/work/savant_v2_bundle.pkl --exclude-functional
```

Reports `all` vs `clean` (contamination-excluded) enrichment tables — quote `clean`.

## Order summary

```
1 (data) ──► 2 (train) ──► 5 (architecture comparison) ──► 6 (LIT-PCBA)
                │
3 (structure embeddings, independent) ──► 4 (encoder comparison)
```

Steps 1 and 3 are the only ones that need network/SLURM scheduling; everything else runs
immediately on whatever node you're on.
