"""Figures for report/Article.tex.

Run from anywhere:  python report/make_figures.py

Every figure is restricted to the compared tools and to the model of record
(`Savant esm650`), so nothing here shows the development variants that the
write-up deliberately leaves out.

The per-target QSAR and Tanimoto k-NN baselines were dropped from the paper,
so they are dropped here too: a figure that still plotted them would disagree
with every table beside it. The crossover figure went with them, since its
entire content was Savant against those 2 baselines.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "ML_model" / "benchmark" / "eval" / "output"
CURVE = REPO / "ML_model" / "work_curve"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL = "Savant esm650"
DISPLAY = {"Savant esm650": "Savant", "Boltz-2": "Boltz-2",
           "PRODIGY-LIG": "PRODIGY-LIG", "PLAPT": "PLAPT"}
ORDER = list(DISPLAY)
COLOR = {"Savant esm650": "#2a78d6", "Boltz-2": "#e2603a",
         "PRODIGY-LIG": "#18a06b", "PLAPT": "#5b4bb0"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})


def bare(ax):
    ax.spines[["top", "right"]].set_visible(False)


# --- 1. learning curve -------------------------------------------------------
curve = pd.read_csv(CURVE / "learning_curve_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for feat, col, lab in [("structure-based", "#2a78d6", "structure-based (+HADDOCK3)"),
                       ("structure-free", "#8a857c", "structure-free")]:
    s = curve[curve.features == feat].sort_values("n_train")
    axes[0].errorbar(s.n_train, s.R2, yerr=s.R2_sd, marker="o", ms=4,
                     color=col, label=lab, capsize=2, lw=1.4)
    axes[1].plot(s.n_train, s.Pearson, marker="o", ms=4, color=col, label=lab, lw=1.4)
for ax, lab in zip(axes, ["held-out $R^2$", "held-out Pearson $r$"]):
    ax.set_xlabel("training pairs")
    ax.set_ylabel(lab)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    bare(ax)
fig.tight_layout()
fig.savefig(FIG / "fig_learning_curve.png", bbox_inches="tight")
plt.close(fig)

# --- 2. between vs within decomposition -------------------------------------
dec = pd.read_csv(EVAL / "bench_A_decomposition.csv", index_col=0)
d = dec.loc[[t for t in ORDER if t in dec.index]]
x = np.arange(len(d))
fig, ax = plt.subplots(figsize=(7.2, 3.1))
ax.bar(x - 0.19, d.between_r, 0.36, label="between-target (which protein binds tighter)",
       color="#c9c4bb", edgecolor="none")
ax.bar(x + 0.19, d.within_r, 0.36, label="within-target (which ligand binds tighter)",
       color="#2a78d6", edgecolor="none")
ax.axhline(0, color="#444", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([DISPLAY[t] for t in d.index], fontsize=8)
ax.set_ylabel("Pearson $r$")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
bare(ax)
fig.tight_layout()
fig.savefig(FIG / "fig_decomposition.png", bbox_inches="tight")
plt.close(fig)

# --- 3. per-target within-target Spearman, Savant vs Boltz-2 -----------------
pt = pd.read_csv(EVAL / "bench_B_per_target.csv")
key = pt.columns[0]
pt = pt.set_index(key)
cols = [c for c in pt.columns if MODEL in c or "Boltz-2" in c]
sav_col = next(c for c in pt.columns if c.startswith(MODEL))
bz_col = next(c for c in pt.columns if c.startswith("Boltz-2"))
p = pt[[sav_col, bz_col]].dropna().sort_values(sav_col)
y = np.arange(len(p))
fig, ax = plt.subplots(figsize=(6.4, 5.2))
for yi, (a, b) in zip(y, p.values):
    ax.plot([a, b], [yi, yi], color="#ccc7bd", lw=1.2, zorder=1)
ax.scatter(p[sav_col], y, s=26, color="#2a78d6", label="Savant", zorder=2)
ax.scatter(p[bz_col], y, s=26, color="#e2603a", marker="s", label="Boltz-2", zorder=2)
ax.axvline(0, color="#444", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(p.index, fontsize=7.5)
ax.set_xlabel("within-target Spearman $\\rho$")
ax.legend(fontsize=8, frameon=False, loc="lower right")
bare(ax)
fig.tight_layout()
fig.savefig(FIG / "fig_within_target.png", bbox_inches="tight")
plt.close(fig)

# --- 4. benchmark summary: between / within / RMSE ---------------------------
# One figure carrying the paper's central contrast (between vs within) and the
# calibration result (RMSE against the constant baseline), for the 4 compared
# methods. This replaces the separate decomposition and spread figures so the
# paper stays inside a 10-float budget.
dec = pd.read_csv(EVAL / "bench_A_decomposition.csv", index_col=0)
pool = pd.read_csv(EVAL / "bench_A_pooled.csv", index_col=0)
order = [t for t in ORDER if t in dec.index and t in pool.index]
x = np.arange(len(order))
const = float(pool.loc["Constant mean", "rmse"])

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
panels = [(dec.loc[order, "between_r"], "between-target $r$", "which protein binds tighter"),
          (dec.loc[order, "within_r"], "within-target $r$", "which ligand binds tighter"),
          (pool.loc[order, "rmse"], "RMSE (p$K_d$)", "lower is better")]
for ax, (vals, ylab, sub) in zip(axes, panels):
    ax.bar(x, vals, 0.62, color=[COLOR[t] for t in order], edgecolor="none")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[t] for t in order], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(sub, fontsize=7.5, color="#555")
    bare(ax)
axes[2].axhline(const, color="#444", lw=0.9, ls="--")
axes[2].text(len(order) - 0.45, const, "constant baseline", fontsize=6.5,
             va="bottom", ha="right", color="#444")
axes[2].set_ylim(0, max(pool.loc[order, "rmse"].max(), const) * 1.18)
fig.tight_layout()
fig.savefig(FIG / "fig_benchmark.png", bbox_inches="tight")
plt.close(fig)

# --- 5. prediction spread ----------------------------------------------------
pool = pd.read_csv(EVAL / "bench_A_pooled.csv", index_col=0)
d = pool.loc[[t for t in ORDER if t in pool.index]]
fig, ax = plt.subplots(figsize=(6.6, 2.9))
x = np.arange(len(d))
ax.bar(x, d.pred_std_ratio, 0.55, color=[COLOR[t] for t in d.index], edgecolor="none")
ax.axhline(1.0, color="#444", lw=0.9, ls="--")
ax.text(len(d) - 0.4, 1.02, "true spread", fontsize=7.5, ha="right", color="#444")
ax.set_xticks(x)
ax.set_xticklabels([DISPLAY[t] for t in d.index], fontsize=8)
ax.set_ylabel("predicted sd / true sd")
bare(ax)
fig.tight_layout()
fig.savefig(FIG / "fig_spread.png", bbox_inches="tight")
plt.close(fig)

print("wrote:", *[p.name for p in sorted(FIG.glob('*.png'))], sep="\n  ")

# --- 6. calibration, Savant against Boltz-2 ----------------------------------
# Predicted against measured, with the identity line and a binned calibration
# curve. The contrast is the point: both methods order compounds, but their
# predictions occupy very different fractions of the measured range.
from scipy.stats import spearmanr

bench = pd.read_csv(EVAL / "bench_scored.csv")
TRUTH = "neg_log10_affinity_M"
SPEC = [("Savant", "savant_esm650_pk", "#2a78d6"),
        ("Boltz-2", "boltz_affinity_pic50", "#e2603a")]

fin = bench[TRUTH].notna()
lo = float(bench.loc[fin, TRUTH].min()) - 0.4
hi = float(bench.loc[fin, TRUTH].max()) + 0.4

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharex=True, sharey=True)
for ax, (name, col, colr) in zip(axes, SPEC):
    m = bench[TRUTH].notna() & bench[col].notna()
    y = bench.loc[m, TRUTH].to_numpy(float)
    p = bench.loc[m, col].to_numpy(float)

    ax.plot([lo, hi], [lo, hi], color="#888", lw=1.0, ls="--", zorder=1)
    ax.scatter(y, p, s=13, color=colr, alpha=0.45, edgecolor="none", zorder=2)

    # binned calibration curve: mean prediction within each measured-pKd bin
    edges = np.arange(np.floor(y.min()), np.ceil(y.max()) + 1, 1.0)
    idx = np.digitize(y, edges) - 1
    cx, cy, ce = [], [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        if sel.sum() >= 5:
            cx.append((edges[k] + edges[k + 1]) / 2)
            cy.append(p[sel].mean())
            ce.append(p[sel].std() / np.sqrt(sel.sum()))
    ax.errorbar(cx, cy, yerr=ce, marker="o", ms=4, lw=1.6, color="#222",
                capsize=2, zorder=3, label="binned mean")

    slope = np.polyfit(y, p, 1)[0]
    ax.set_title(f"{name}\nslope {slope:.2f}   spread {p.std()/y.std():.2f}   "
                 r"$\rho$ " f"{spearmanr(y, p).correlation:.2f}", fontsize=8)
    ax.set_xlabel("measured p$K_d$")
    bare(ax)
axes[0].set_ylabel("predicted p$K_d$")
axes[0].legend(fontsize=7, frameon=False, loc="upper left")
axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)
fig.tight_layout()
fig.savefig(FIG / "fig_calibration.png", bbox_inches="tight")
plt.close(fig)

# --- 7. training label distribution ------------------------------------------
# dataset_rows.csv is the collapsed training set itself: its row count equals
# n_fit + n_val + n_test from variant_comparison.csv, so this is the label the
# model actually saw rather than the raw corpus before replicate collapsing.
TRAIN = REPO / "ML_model" / "models_full" / "dataset_rows.csv"
y = pd.read_csv(TRAIN, usecols=["neg_log10_affinity_M"]).neg_log10_affinity_M.to_numpy(np.float32)

CUT = 6.0
fig, ax = plt.subplots(figsize=(7.0, 3.0))
bins = np.arange(0, 15.25, 0.25)
ax.hist(y[y < CUT], bins=bins, color="#b9b3a8", edgecolor="none")
ax.hist(y[y >= CUT], bins=bins, color="#2a78d6", edgecolor="none")
ax.axvline(CUT, color="#222", lw=1.4)
top = ax.get_ylim()[1]
ax.set_ylim(0, top * 1.22)
## Annotations sit above the tallest bar rather than over it: both classes peak
## close to the cut, so anything placed at the axis top overlaps the histogram.
ax.text(CUT - 0.3, top * 1.10, f"non-binder\n{100*(y<CUT).mean():.1f}%",
        ha="right", va="top", fontsize=8.5, color="#6b6459")
ax.text(CUT + 0.3, top * 1.10, f"binder\n{100*(y>=CUT).mean():.1f}%",
        ha="left", va="top", fontsize=8.5, color="#1b5fa8")
ax.annotate(r"p$K_d$ = 6", xy=(CUT, top * 0.55), xytext=(CUT + 1.9, top * 0.72),
            fontsize=8, color="#222",
            arrowprops=dict(arrowstyle="->", color="#222", lw=0.9))
ax.set_xlabel("measured p$K_d$")
ax.set_ylabel("training pairs")
ax.set_xlim(0, 13)
bare(ax)
fig.tight_layout()
fig.savefig(FIG / "fig_label_distribution.png", bbox_inches="tight")
plt.close(fig)

# --- 8. carbonic anhydrase 2 case study --------------------------------------
# Structure panels rendered by benchmark_structures/render_panels.pml, composed
# with the measured and predicted values on a shared axis underneath. Draws
# labelled placeholders when the panels are absent, so the layout can be checked
# before the renders exist.
STRUCT = REPO / "report" / "benchmark_structures"

CASE = [  # ordered weakest -> strongest measured affinity
    ("CHEMBL6174863", 4.02, 4.37, 4.85),
    ("CHEMBL6176691", 5.06, 5.10, 5.22),
    ("CHEMBL6148278", 6.18, 7.04, 7.96),
    ("CHEMBL6151935", 7.24, 7.20, 6.27),
    ("CHEMBL6134150", 8.25, 7.22, 7.16),
]
ids = [c[0] for c in CASE]
true = np.array([c[1] for c in CASE])
sav = np.array([c[2] for c in CASE])
bz = np.array([c[3] for c in CASE])
x = np.arange(len(CASE))

fig = plt.figure(figsize=(7.4, 4.6))
gs = fig.add_gridspec(2, len(CASE), height_ratios=[1.45, 1.0], hspace=0.10, wspace=0.04)

for i, cid in enumerate(ids):
    ax = fig.add_subplot(gs[0, i])
    panel = STRUCT / f"panel_{i+1}.png"
    if panel.exists():
        ax.imshow(plt.imread(panel))
    else:
        ax.add_patch(plt.Rectangle((0.04, 0.04), 0.92, 0.92, fill=False,
                                   ls="--", lw=1.0, color="#bbb"))
        ax.text(0.5, 0.5, f"panel_{i+1}.png", ha="center", va="center",
                fontsize=7, color="#999")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(cid.replace("CHEMBL", "CHEMBL "), fontsize=6.5, color="#444", pad=3)

ax = fig.add_subplot(gs[1, :])
ax.plot(x, true, "-o", color="#222", lw=1.8, ms=7, label="measured", zorder=3)
ax.plot(x, sav, "-o", color="#2a78d6", lw=1.6, ms=6, label="Savant", zorder=2)
ax.plot(x, bz, "-s", color="#e2603a", lw=1.6, ms=5.5, label="Boltz-2", zorder=2)
for xi, t, s in zip(x, true, sav):
    ax.plot([xi, xi], [min(t, s), max(t, s)], color="#2a78d6", lw=0.8, alpha=0.35, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([f"{t:.2f}" for t in true], fontsize=8)
ax.set_xlabel("compound, ordered by measured p$K_d$", fontsize=9)
ax.set_ylabel("p$K_d$", fontsize=9)
ax.set_xlim(-0.45, len(CASE) - 0.55)
ax.set_ylim(3.6, 8.9)
## lower right: the only quadrant none of the 3 series passes through.
ax.legend(fontsize=8, frameon=False, ncol=1, loc="lower right")
bare(ax)

fig.savefig(FIG / "fig_case_study.png", bbox_inches="tight", dpi=300)
plt.close(fig)
