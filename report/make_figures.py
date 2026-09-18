"""Figures for report/savant_methods_results.tex.

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

# --- 4. prediction spread ----------------------------------------------------
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
