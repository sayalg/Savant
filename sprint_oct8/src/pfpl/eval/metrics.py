"""One metric definition, used by every model (PLAN.md Section 9).

Two headline numbers, kept side by side on purpose:

* **Pearson on the protein-cluster split.** This is the number Section 9's
  comparison rule is written in: physics must beat the matched no-physics control
  by dPearson >= 0.02 with a bootstrap CI excluding zero. It is what the literature
  in Section 0.2 reports, so it is what makes this work comparable to anything else.
* **Median per-target Spearman.** This is the number that survives contact with
  reality. The previous project's headline pooled Pearson of 0.564 on a 315,792-row
  held-out fold was indistinguishable from an oracle that predicts each target's
  mean and never looks at the ligand (0.570) - a pooled metric over many proteins is
  saturated by between-protein variance and says nothing about whether a model can
  rank ligands, which is the only thing a screen does.

Three baselines are reported in EVERY table, per Section 0.3.2:

    const        predict the global training mean. Sets the RMSE floor. Its
                 per-target rho is UNDEFINED (a constant has no within-target
                 variance) and is reported as such, not as 0.
    target_mean  predict each test target's own true mean. The ligand-blind POOLED
                 ceiling. Not achievable at inference; it exists to show how much of
                 a pooled score needs no ligand information at all.
    ligand_only  a real model given ligand features and no protein. The memorisation
                 floor, and the only baseline with a meaningful per-target rho.

Section 0.3.2's rule is enforced by `shortcut_flag`: if ligand-only is within 0.02
Pearson of the fusion model, the split has a ligand shortcut and needs a harder
variant. That is not advice here; it comes back in the report as a flag.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

MIN_TARGET_N = 8          # a per-target rho on fewer pairs than this is noise
SHORTCUT_MARGIN = 0.02    # Section 0.3.2 / Section 9
UNDEF = float("nan")


# ---------------------------------------------------------------- statistics

def _ranks(v: np.ndarray) -> np.ndarray:
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v), float)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return UNDEF
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return UNDEF if len(x) < 3 else pearson(_ranks(x), _ranks(y))


def rmse(x, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(x, float) - np.asarray(y, float)) ** 2)))


def mae(x, y) -> float:
    return float(np.mean(np.abs(np.asarray(x, float) - np.asarray(y, float))))


def pairwise_accuracy(y, p, min_gap: float = 0.0) -> float:
    """Fraction of ligand pairs ordered correctly - Section 9's within-target metric."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 2:
        return UNDEF
    dy = y[:, None] - y[None, :]
    dp = p[:, None] - p[None, :]
    iu = np.triu_indices(len(y), 1)
    dy, dp = dy[iu], dp[iu]
    m = np.abs(dy) > min_gap
    return float((np.sign(dy[m]) == np.sign(dp[m])).mean()) if m.any() else UNDEF


def _nanmedian(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else UNDEF


def _nanmean(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else UNDEF


def bootstrap_ci(v, stat=np.median, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return (UNDEF, UNDEF)
    rng = np.random.default_rng(seed)
    s = np.sort(stat(rng.choice(v, size=(n, len(v)), replace=True), axis=1))
    return float(s[int(alpha / 2 * n)]), float(s[int((1 - alpha / 2) * n)])


def paired_bootstrap_delta(y, pa, pb, metric=pearson, n: int = 2000, seed: int = 0):
    """CI on metric(a) - metric(b) on the SAME rows. Section 9's comparison rule.

    Paired, because the two models see identical test items and an unpaired interval
    on a small difference is far too wide to decide anything.
    """
    y, pa, pb = (np.asarray(v, float) for v in (y, pa, pb))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d = np.array([metric(y[i], pa[i]) - metric(y[i], pb[i]) for i in idx])
    d = d[np.isfinite(d)]
    if len(d) < 10:
        return dict(delta=UNDEF, lo=UNDEF, hi=UNDEF, excludes_zero=False)
    d.sort()
    lo, hi = float(d[int(0.025 * len(d))]), float(d[int(0.975 * len(d))])
    return dict(delta=float(metric(y, pa) - metric(y, pb)), lo=lo, hi=hi,
                excludes_zero=bool(lo > 0 or hi < 0))


def quantile_calibrator(train_scores, train_y):
    """Map an arbitrary score onto the training label distribution.

    A ranking or margin loss emits a score with no units, so RMSE against pK is
    meaningless and would fail a gate for the wrong reason (one such model reported
    an offset slope of +27.3). Calibrating by quantile makes error comparable again
    without touching the ordering - monotone by construction, so it cannot change rho.
    """
    s = np.sort(np.asarray(train_scores, float))
    y = np.sort(np.asarray(train_y, float))
    if len(s) < 2:
        return lambda p: np.asarray(p, float)

    def cal(p):
        q = np.searchsorted(s, np.asarray(p, float), side="left") / max(len(s) - 1, 1)
        return np.quantile(y, np.clip(q, 0.0, 1.0))
    return cal


def enrichment(y_true, y_pred, k: int = 10, q: float = 0.9) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    if n < k + 1:
        return dict(ef=UNDEF, recall=UNDEF, n_active=0)
    thr = np.quantile(y_true, q)
    act = y_true >= thr
    if act.sum() == 0 or act.all():
        return dict(ef=UNDEF, recall=UNDEF, n_active=int(act.sum()))
    top = np.argsort(-y_pred)[:k]
    hits = int(act[top].sum())
    return dict(ef=float((hits / k) / (act.sum() / n)), recall=float(hits / act.sum()),
                n_active=int(act.sum()), threshold=float(thr))


# ------------------------------------------------------------------ the report

@dataclass
class Report:
    name: str
    n_pairs: int = 0
    n_targets: int = 0
    pearson: float = UNDEF
    pearson_ci: tuple = (UNDEF, UNDEF)
    spearman: float = UNDEF
    rmse: float = UNDEF
    mae: float = UNDEF
    rmse_const: float = UNDEF
    per_target_spearman_median: float = UNDEF
    per_target_spearman_mean: float = UNDEF
    per_target_spearman_ci: tuple = (UNDEF, UNDEF)
    per_target_pairwise_acc: float = UNDEF
    frac_targets_positive: float = UNDEF
    pred_std_ratio: float = UNDEF
    offset_slope: float = UNDEF
    ef_at_10_median: float = UNDEF
    coverage_80: float = UNDEF
    unc_error_corr: float = UNDEF
    scale_free: bool = False
    pooled_only: bool = False
    note: str = ""
    per_target: list = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("per_target")
        return d

    def line(self) -> str:
        def f(v, s="+.3f"):
            return "   -  " if not np.isfinite(v) else format(v, s)
        lo, hi = self.per_target_spearman_ci
        ci = "" if not np.isfinite(lo) else f"[{lo:+.3f},{hi:+.3f}]"
        pos = "  - " if not np.isfinite(self.frac_targets_positive) \
            else f"{self.frac_targets_positive:4.0%}"
        return (f"{self.name:<34} r {f(self.pearson):>6}  "
                f"rho {f(self.per_target_spearman_median):>6} {ci:<17} "
                f"tgt {self.n_targets:>4} pos {pos}  "
                f"RMSE {f(self.rmse, '.3f')}/{f(self.rmse_const, '.3f')}  "
                f"slope {f(self.offset_slope):>6}  EF@10 {f(self.ef_at_10_median, '.2f'):>5}")


def evaluate(df: pd.DataFrame, name: str = "model", y: str = "y", pred: str = "pred",
             target: str = "protein_idx", logvar: str | None = None,
             min_n: int = MIN_TARGET_N, train_mean: float | None = None,
             scale_free: bool = False, calibrator=None, pooled_only: bool = False,
             note: str = "") -> Report:
    for c in (y, pred, target):
        if c not in df.columns:
            raise KeyError(f"evaluate() needs column {c!r}; have {list(df.columns)}")
    cols = [target, y, pred] + ([logvar] if logvar and logvar in df.columns else [])
    d = df[cols].dropna(subset=[target, y, pred])
    gm = float(train_mean if train_mean is not None else d[y].mean())
    yv, pv = d[y].to_numpy(float), d[pred].to_numpy(float)

    ## RMSE only means something on the label's own scale. A ranking score has no
    ## scale, so either calibrate it or decline to report an error.
    cal_pv = calibrator(pv) if (scale_free and calibrator is not None) else pv
    rm = rmse(yv, cal_pv) if (not scale_free or calibrator is not None) else UNDEF

    rep = Report(name=name, n_pairs=len(d), scale_free=scale_free,
                 pooled_only=pooled_only, note=note,
                 pearson=pearson(yv, pv), spearman=spearman(yv, pv),
                 rmse=rm, mae=mae(yv, cal_pv) if np.isfinite(rm) else UNDEF,
                 rmse_const=rmse(yv, np.full(len(yv), gm)))
    rep.pearson_ci = bootstrap_ci(
        np.array([pearson(yv[i], pv[i]) for i in
                  np.random.default_rng(0).integers(0, len(yv), (200, len(yv)))])
        if len(yv) > 20 else np.array([]), stat=np.median)

    if logvar and logvar in d.columns:
        sd = np.exp(0.5 * d[logvar].to_numpy(float))
        err = np.abs(yv - cal_pv)
        rep.coverage_80 = float((err <= 1.2816 * sd).mean())
        rep.unc_error_corr = spearman(sd, err)

    if pooled_only:
        rep.n_targets = int(d[target].nunique())
        return rep

    d = d.assign(_scaled=cal_pv if not (scale_free and calibrator is None) else np.nan)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t, g in d.groupby(target, sort=False):
            if len(g) < min_n or g[y].std() < 1e-9:
                continue
            gy, gp = g[y].to_numpy(float), g[pred].to_numpy(float)
            gs = g["_scaled"].to_numpy(float)
            rows.append(dict(target=t, n=len(g), rho=spearman(gy, gp), r=pearson(gy, gp),
                             pair_acc=pairwise_accuracy(gy, gp),
                             y_mean=float(gy.mean()), pred_mean=float(np.mean(gs)),
                             y_std=float(gy.std()), pred_std=float(np.std(gs, ddof=1)),
                             ef10=enrichment(gy, gp, 10)["ef"]))
    pt = pd.DataFrame(rows)
    if pt.empty:
        rep.note = (note + f" | no target had >={min_n} pairs with label variance").strip(" |")
        return rep

    rho = pt.rho.to_numpy(float)
    rep.n_targets = len(pt)
    rep.per_target_spearman_median = _nanmedian(rho)
    rep.per_target_spearman_mean = _nanmean(rho)
    rep.per_target_spearman_ci = bootstrap_ci(rho)
    rep.per_target_pairwise_acc = _nanmedian(pt.pair_acc.to_numpy(float))
    fin = rho[np.isfinite(rho)]
    rep.frac_targets_positive = float((fin > 0).mean()) if len(fin) else UNDEF
    rep.ef_at_10_median = _nanmedian(pt.ef10.to_numpy(float))
    ## Does the protein move the prediction the right way and by the right amount?
    ## slope 1.0 = correctly scaled; the shipped v1 model measured 0.089.
    if (len(pt) >= 3 and pt.y_mean.std() > 1e-9
            and np.isfinite(pt.pred_mean).all() and pt.pred_mean.std() > 1e-12):
        rep.offset_slope = float(np.polyfit(pt.y_mean, pt.pred_mean, 1)[0])
    if pt.y_std.mean() > 1e-9 and np.isfinite(pt.pred_std).all():
        rep.pred_std_ratio = float(pt.pred_std.mean() / pt.y_std.mean())
    rep.per_target = pt.to_dict("records")
    return rep


def blind_baselines(train: pd.DataFrame, test: pd.DataFrame, y: str = "y",
                    target: str = "protein_idx") -> dict[str, Report]:
    """const and target_mean. Both are POOLED-ONLY by construction: a prediction
    that is constant within a target has no within-target ordering, so reporting a
    per-target rho for them would be reporting a division by zero."""
    gm = float(train[y].mean())
    out = {}
    t = test.copy(); t["pred"] = gm
    out["const"] = evaluate(t, "baseline: global constant", y=y, target=target,
                            train_mean=gm, pooled_only=True,
                            note="RMSE floor; per-target rho undefined")
    t = test.copy(); t["pred"] = t.groupby(target)[y].transform("mean")
    out["target_mean"] = evaluate(t, "ORACLE: target mean (ligand-blind)", y=y,
                                  target=target, train_mean=gm, pooled_only=True,
                                  note="pooled ceiling without ligand information; "
                                       "not achievable at inference")
    return out


def shortcut_flag(model: Report, ligand_only: Report,
                  margin: float = SHORTCUT_MARGIN) -> dict:
    """Section 0.3.2: ligand-only within `margin` Pearson means the split is soft."""
    a, b = model.pearson, ligand_only.pearson
    if not (np.isfinite(a) and np.isfinite(b)):
        return {"shortcut": None, "note": "a Pearson is undefined"}
    return {
        "shortcut": bool(a - b < margin),
        "delta_pearson_vs_ligand_only": float(a - b),
        "note": (f"ligand-only is within {margin} Pearson of the model: this split has a "
                 "ligand shortcut and needs a harder variant (Section 5.3.5)"
                 if a - b < margin else "model is meaningfully above the ligand-only floor"),
    }


def tier1_gate(model: Report, control: Report, ligand_only: Report,
               target_mean: Report, y, pred_model, pred_control,
               margin: float = SHORTCUT_MARGIN) -> dict:
    """Gate G2 / Tier 1 exactly as Section 9 defines it.

    "beats the matched no-physics control on the protein-cluster split by
    dPearson >= 0.02 with a bootstrap CI excluding zero, and beats ligand-only and
    target-mean."
    """
    d = paired_bootstrap_delta(y, pred_model, pred_control, pearson)
    beats_control = bool(np.isfinite(d["delta"]) and d["delta"] >= margin
                         and d["excludes_zero"])
    beats_ligand = bool(model.per_target_spearman_median
                        > ligand_only.per_target_spearman_median)
    beats_ligand_r = bool(model.pearson > ligand_only.pearson)
    beats_oracle = bool(model.pearson > target_mean.pearson)
    passed = beats_control and beats_ligand and beats_ligand_r and beats_oracle
    return {
        "tier1_pass": passed,
        "delta_pearson_vs_control": d,
        "beats_control_by_margin": beats_control,
        "beats_ligand_only_per_target_rho": beats_ligand,
        "beats_ligand_only_pearson": beats_ligand_r,
        "beats_target_mean_pearson": beats_oracle,
        "margin": margin,
    }


def print_table(reports: dict[str, Report], title: str = "") -> None:
    print("\n" + "=" * 140)
    if title:
        print(title)
        print("=" * 140)
    for r in reports.values():
        print("  " + r.line())
        if r.note:
            print(f"      ({r.note})")
    print("=" * 140)


def write(reports: dict[str, Report], path, extra: dict | None = None) -> None:
    payload = {k: v.to_dict() for k, v in reports.items()}
    if extra:
        payload["_extra"] = extra
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print(f"  wrote {path}")
