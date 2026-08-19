# Friday report

`savant_benchmark_report.qmd` — the benchmark write-up, aimed at a reader who is
new to binding-affinity prediction. All results are expressed in Kd (nanomolar)
so the four tools can be compared on the same scale.

## Render it

```bash
conda activate savant2
quarto render savant_benchmark_report.qmd
```

That produces a single self-contained `savant_benchmark_report.html` — figures
and styling are embedded, so it can be emailed or dropped in a shared folder
with no accompanying files.

For a PDF you also need a LaTeX toolchain:

```bash
quarto install tinytex
quarto render savant_benchmark_report.qmd --to pdf
```

## Requirements

Quarto itself (https://quarto.org/docs/get-started/), plus `jupyter`,
`matplotlib`, `pandas` and `numpy` in the active environment. Everything else
comes from `../environment.yml`.

```bash
conda install -c conda-forge jupyter matplotlib
```

## Data

The report reads `../output/benchmark_v2_scored.csv` and computes every number
in it at render time — nothing is hardcoded, so re-running the benchmark and
re-rendering keeps the text and figures in agreement.

Code chunks are folded by default (`code-fold: true`); a reader can expand any
of them to see exactly how a figure or table was produced.
