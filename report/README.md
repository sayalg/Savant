# The manuscript

`Article.tex` is the paper: *Savant: Fast, Structure-Free Protein-Ligand Affinity
Prediction Enabled Ligand Screening.*

Every number in it is measured, and each table carries a `% source:` comment
naming the file it came from, so a table can be re-checked against a re-run
rather than trusted. The repository root README maps each `\label` to the script
that produces its source file.

## Build it

```bash
cd report
pdflatex Article
bibtex   Article
pdflatex Article
pdflatex Article
```

**You need `zHenriquesLab-StyleBioRxiv.cls` in this directory.** It is not
vendored here. Fetch it from the
[HenriquesLab bioRxiv template](https://github.com/HenriquesLab/Journal-style-bioRxiv)
and drop the `.cls` beside `Article.tex`, or swap the `\documentclass` line for
`article` and remove the `\leadauthor`, `\affil`, `\shorttitle`, `\keywords` and
`\corrauthor` commands it provides.

## Regenerate the figures

```bash
python report/make_figures.py      # runs from anywhere
```

That reads `ML_model/work_curve/learning_curve_summary.csv` and the `bench_A_*`
/ `bench_B_*` CSVs in `ML_model/benchmark/eval/output/`, and overwrites the four
PNGs in `figures/`. It needs only `pandas`, `numpy` and `matplotlib`.

Every figure is restricted to the four compared methods and to the model of
record, `Savant esm650`, so nothing plots a development variant the text leaves
out.

| figure | shows | reads |
|---|---|---|
| `fig_learning_curve.png` | held-out *R²* and *r* against training size, with and without the HADDOCK3 block | `learning_curve_summary.csv` |
| `fig_decomposition.png` | between-target vs. within-target correlation, per method | `bench_A_decomposition.csv` |
| `fig_spread.png` | predicted / true standard deviation, per method | `bench_A_pooled.csv` |
| `fig_within_target.png` | within-target Spearman ρ for each of the 20 targets, Savant vs. Boltz-2 | `bench_B_per_target.csv` |

## Files

| file | what |
|---|---|
| `Article.tex` | the manuscript |
| `references.bib` | its bibliography |
| `make_figures.py` | the four figures |
| `figures/` | their output |
| `intro_discussion_lit_review.md` | the literature review the Introduction and Discussion follow, with the reasoning behind each citation |

## Two things the header comment of `Article.tex` records, worth repeating

- **The screening-directed variant is not reported.** An ESM C 300M encoder with
  target-centered residual labels and scaffold-held-out inner folds is described
  in the Conclusions as future work, because that run was never completed. It
  lives on the `dev` branch. Do not move it into Methods or Results until it has
  numbers.
- **The per-target QSAR and Tanimoto k-NN baselines were removed** on 2026-09-17,
  along with the data-volume crossover analysis built on them.
  `make_figures.py` was updated to match, so no figure still plots them. The
  columns survive in `bench_scored.csv` because the scoring notebook computes
  them; nothing in the paper reads them.
