"""The shared trainer: losses, checkpoint/resume, and the calibration monitor.

PLAN.md Section 8's curriculum is four stages over one model. They differ only in
which data they see and which loss terms are active, so they share this loop rather
than each owning a copy that can drift:

    Stage 0  physics pretraining on QM dimers        (physics/qm_pretrain.py)
    Stage 1  geometry: distogram CE + pocket BCE + triangle + contact prior
    Stage 2  affinity: heteroscedastic Huber + censored hinge + the ANCHOR
    Stage 3  crystal fine-tune: all of the above, lower LR, higher weight

The anchor is the load-bearing part. Section 0.3.4 requires the distogram
cross-entropy to stay on through every later stage, in mixed batches, and Section
2.1 explains why: with only an affinity loss, gradient descent discovers that
pushing probability mass toward each pair's energy minimum lowers the loss, and the
model converges on an over-attractive geometry that no arrangement of atoms could
produce. `LossWeights.disto` defaults to a non-zero value in every stage, and
`fit()` refuses to run Stage 2 with the anchor off unless `allow_unanchored=True`
is passed explicitly and recorded in the run config.

Every job is checkpointed each epoch and resumes from `last.pt`, because Section
7.1 requires wall-time kills and preemption to be survivable.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..eval import metrics as M
from ..physics.energy import (contact_prior_loss, distogram_calibration,
                              distogram_ce, triangle_penalty)
from ..physics.params import drift_report, prior_penalty
from ..models.coord_head import wta_coord_loss
from ..models.pocket import pocket_loss, pocket_metrics


@dataclass
class LossWeights:
    affinity: float = 1.0
    disto: float = 1.0          # the anchor - see the module docstring
    pocket: float = 1.0
    triangle: float = 0.1
    contact: float = 0.05
    prior: float = 0.01         # annealed in `anneal_prior`
    coord: float = 0.5          # Variant B only
    censored: float = 0.5


def huber_hetero(pred, target, logvar, weight, delta: float = 1.0,
                 use_uncertainty: bool = True):
    """Heteroscedastic Huber (Section 8, Stage 2).

    Section 4.4 puts inter-lab scatter at ~0.5 log units with 6-8% of curated pairs
    disagreeing by more than 1. A plain MSE spends its capacity on those outliers;
    Huber does not, and the learned variance lets the model say which pairs it
    cannot fit rather than distorting itself to fit them.
    """
    err = pred - target
    a = err.abs()
    h = torch.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))
    if not use_uncertainty:
        return (h * weight).sum() / weight.sum().clamp(min=1e-6)
    ## The +0.5*logvar term is what stops the model from declaring infinite variance
    ## everywhere to zero out the loss.
    l = h * torch.exp(-logvar) + 0.5 * logvar
    return (l * weight).sum() / weight.sum().clamp(min=1e-6)


def censored_hinge(pred, target, rel, weight):
    """One-sided loss for censored measurements (Section 5.2).

    `relation` is in **pK space** - see schema.ALLOWED_RELATIONS, which is where
    every loader flips its concentration-space relation on the way in. So:

        rel ">"  the true pK is ABOVE target  -> penalise pred BELOW target
        rel "<"  the true pK is BELOW target  -> penalise pred ABOVE target

    Penalising the other side would train on information the assay does not
    contain: "Kd > 10 uM" says only that binding is weaker than 10 uM, and any
    prediction weaker still is consistent with it.
    """
    gt = rel == 1      # pK > target
    lt = rel == 2      # pK < target
    loss = torch.zeros_like(pred)
    loss = torch.where(gt, F.relu(target - pred), loss)
    loss = torch.where(lt, F.relu(pred - target), loss)
    m = (gt | lt).to(pred.dtype) * weight
    return (loss * m).sum() / m.sum().clamp(min=1e-6)


def anneal_prior(base: float, step: int, total: int, floor: float = 0.1) -> float:
    """Section 6.4: strong parameter priors early, relaxed later, never zero.

    Zero would let the parameter heads drift arbitrarily far from tabulated force
    field values, at which point calling them force-field parameters is a claim the
    model no longer supports.
    """
    frac = min(step / max(total, 1), 1.0)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * frac)))


def compute_losses(out, batch, w: LossWeights, stage: str, step: int = 0,
                   total_steps: int = 1, use_uncertainty: bool = True) -> tuple:
    parts: dict[str, torch.Tensor] = {}
    dev = out.pK.device
    zero = torch.zeros((), device=dev)

    y = batch["y"]
    weight = batch["weight"]
    rel = batch["rel"]
    exact = (rel == 0).to(y.dtype)

    if w.affinity > 0 and stage in ("affinity", "finetune"):
        parts["affinity"] = huber_hetero(out.pK, y, out.logvar, weight * exact,
                                         use_uncertainty=use_uncertainty)
        if w.censored > 0 and (rel != 0).any():
            parts["censored"] = w.censored * censored_hinge(out.pK, y, rel, weight)

    hg = batch.get("has_geometry")
    if hg is not None and hg.sum() > 0:
        m = hg.bool()
        if w.pocket > 0 and batch.get("has_pocket") is not None:
            pm = batch["has_pocket"].bool()
            if pm.any():
                parts["pocket"] = w.pocket * pocket_loss(
                    out.pocket_logits[pm], batch["pocket_target"][pm],
                    batch["prot_mask"][pm])
        if w.disto > 0 and out.disto_logits is not None:
            ## Targets are stored at (L*4, A); the model emits (K*4, A) for the
            ## selected pocket residues, so the targets are gathered to match.
            tb, tv = _gather_sites(batch["dist_bins"], batch["dist_valid"],
                                   out.pocket_idx, out.disto_logits.shape[1])
            parts["disto"] = w.disto * distogram_ce(out.disto_logits[m], tb[m], tv[m])
        if w.coord > 0 and out.coord_d is not None:
            td, tv = _gather_true_dist(batch["true_dist"], batch["dist_valid"],
                                       out.pocket_idx, out.coord_d.shape[2])
            parts["coord"] = w.coord * wta_coord_loss(out.coord_d[m], td[m], tv[m],
                                                      torch.ones(int(m.sum()),
                                                                 out.coord_d.shape[1],
                                                                 device=dev)
                                                      / out.coord_d.shape[1])

    if out.P is not None:
        if w.triangle > 0 and "lig_dist" in batch:
            parts["triangle"] = w.triangle * triangle_penalty(
                out.P, batch["lig_dist"], out.site_mask, batch["lig_mask"])
        if w.contact > 0 and out.energy is not None:
            parts["contact"] = w.contact * contact_prior_loss(
                out.energy.burial, batch["lig_mask"])

    if w.prior > 0 and out.theta_i is not None:
        pw = anneal_prior(w.prior, step, total_steps)
        parts["prior"] = pw * (prior_penalty(out.theta_i, out.prior_i, out.site_mask)
                               + prior_penalty(out.theta_j, out.prior_j,
                                               batch["lig_mask"]))

    total = sum(parts.values()) if parts else zero
    return total, {k: float(v.detach()) for k, v in parts.items()}


def _gather_sites(dist_bins, dist_valid, pocket_idx, n_sites_model):
    """(B, L*4, A) crystal targets -> (B, K*4, A) for the selected pocket residues."""
    b, _, a = dist_bins.shape
    k = pocket_idx.shape[1]
    site = (pocket_idx.unsqueeze(-1) * 4 + torch.arange(4, device=pocket_idx.device)
            ).reshape(b, k * 4)
    site = site.clamp(max=dist_bins.shape[1] - 1)
    g = site.unsqueeze(-1).expand(-1, -1, a)
    tb = torch.gather(dist_bins, 1, g)
    tv = torch.gather(dist_valid, 1, g)
    if tb.shape[1] != n_sites_model:      # A padded to a different width
        n = min(tb.shape[1], n_sites_model)
        tb, tv = tb[:, :n], tv[:, :n]
    return tb, tv


def _gather_true_dist(true_dist, dist_valid, pocket_idx, n_sites_model):
    return _gather_sites(true_dist, dist_valid, pocket_idx, n_sites_model)


class Trainer:
    def __init__(self, model, cfg, out_dir: Path, stage: str = "affinity",
                 device: str | None = None):
        from ..preflight import pick_device
        self.model = model
        self.cfg = cfg
        self.stage = stage
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device or pick_device()
        self.model.to(self.device)
        t = cfg.train
        self.w = LossWeights(**(cfg.get_path(f"train.loss_weights.{stage}", None)
                                or cfg.get_path("train.loss_weights.default", {}) or {}))
        self.opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(t.get("lr", 3e-4)), weight_decay=float(t.get("weight_decay", 1e-2)))
        self.clip = float(t.get("grad_clip", 1.0))
        self.epochs = int(t.get("epochs", 20))
        self.patience = int(t.get("patience", 5))
        self.use_unc = bool(t.get("uncertainty", True))
        self.amp = bool(t.get("amp", True)) and self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.history: list[dict] = []
        self.best = -np.inf
        self.start_epoch = 0
        self.total_steps = 1

    # ------------------------------------------------------------- checkpointing
    def save(self, name: str, epoch: int, extra: dict | None = None) -> None:
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "epoch": epoch, "best": self.best, "history": self.history,
                    "cfg": dict(self.cfg), **(extra or {})}, self.out / name)

    def maybe_resume(self) -> None:
        p = self.out / "last.pt"
        if not p.exists():
            return
        ck = torch.load(p, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.start_epoch = int(ck["epoch"]) + 1
        self.best = float(ck.get("best", -np.inf))
        self.history = ck.get("history", [])
        print(f"  resumed from {p} at epoch {self.start_epoch}")

    # -------------------------------------------------------------------- epochs
    def run_epoch(self, loader, train: bool, epoch: int) -> dict:
        self.model.train(train)
        agg: dict[str, float] = {}
        n = 0
        preds, ys, tgts, lvs = [], [], [], []
        t0 = time.time()
        for step, batch in enumerate(loader):
            if not batch:
                continue
            from ..data.datasets import to_torch
            b = to_torch(batch, self.device)
            with torch.amp.autocast("cuda", enabled=self.amp):
                out = self.model(b)
                loss, parts = compute_losses(
                    out, b, self.w, self.stage,
                    step=epoch * max(len(loader), 1) + step,
                    total_steps=self.total_steps, use_uncertainty=self.use_unc)
            if train:
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            bs = len(b["y"])
            n += bs
            agg["loss"] = agg.get("loss", 0.0) + float(loss.detach()) * bs
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v * bs
            if not train:
                preds.append(out.pK.detach().float().cpu().numpy())
                ys.append(b["y"].detach().float().cpu().numpy())
                tgts.append(b["protein_idx"].detach().cpu().numpy())
                lvs.append(out.logvar.detach().float().cpu().numpy())
                ## Calibration monitor (Section 0.3.4): if this degrades while
                ## affinity improves, the physics is being gamed.
                if out.P is not None and b.get("has_geometry", torch.zeros(1)).sum() > 0:
                    m = b["has_geometry"].bool()
                    tb, tv = _gather_sites(b["dist_bins"], b["dist_valid"],
                                           out.pocket_idx, out.P.shape[1])
                    for k, v in distogram_calibration(out.P[m], tb[m], tv[m]).items():
                        agg[k] = agg.get(k, 0.0) + v * bs
                    for k, v in pocket_metrics(out.pocket_logits[m],
                                               b["pocket_target"][m],
                                               b["prot_mask"][m]).items():
                        agg[k] = agg.get(k, 0.0) + v * bs
        res = {k: v / max(n, 1) for k, v in agg.items()}
        res["n"] = n
        res["sec"] = round(time.time() - t0, 1)
        ## fit() MAXIMISES its selection metric, so a loss-shaped quantity needs a
        ## negated alias. Stage 1 has no affinity signal to select on and must select
        ## on distogram quality instead.
        if "disto/nll" in res:
            res["val_neg_disto_nll"] = -res["disto/nll"]
        if "disto/ece" in res:
            res["val_neg_disto_ece"] = -res["disto/ece"]
        if preds:
            df = __import__("pandas").DataFrame(dict(
                y=np.concatenate(ys), pred=np.concatenate(preds),
                protein_idx=np.concatenate(tgts), logvar=np.concatenate(lvs)))
            r = M.evaluate(df, "val", logvar="logvar")
            res |= {"val_pearson": r.pearson, "val_rho": r.per_target_spearman_median,
                    "val_rmse": r.rmse, "val_coverage80": r.coverage_80}
            self._last_val_df = df
        return res

    def fit(self, train_loader, val_loader, select_on: str = "val_pearson",
            allow_unanchored: bool = False) -> dict:
        if self.stage in ("affinity", "finetune") and self.w.disto <= 0 \
                and not allow_unanchored:
            raise SystemExit(
                "The distogram anchor is OFF for an affinity stage.\n"
                "  PLAN.md Section 0.3.4 requires it through every later stage, and "
                "Section 2.1\n  explains why: with only an affinity loss the model "
                "learns to move probability\n  mass toward each pair's energy minimum "
                "and converges on an over-attractive,\n  geometrically impossible "
                "solution that scores well on pK and means nothing.\n\n"
                "  If this is a deliberate ablation, pass allow_unanchored=True so it "
                "is recorded\n  in the run config and reported as such.")
        self.maybe_resume()
        self.total_steps = self.epochs * max(len(train_loader), 1)
        bad, warned = 0, False
        for ep in range(self.start_epoch, self.epochs):
            tr = self.run_epoch(train_loader, True, ep)
            va = self.run_epoch(val_loader, False, ep)
            ## A `select_on` that never appears scores -inf forever, so the run
            ## early-stops after `patience` epochs and saves no best checkpoint -
            ## which looks exactly like a model that failed to learn. Say so instead.
            if not warned and select_on not in va:
                warned = True
                avail = sorted(k for k in va if k.startswith(("val_", "disto/", "pocket/")))
                raise SystemExit(
                    f"select_on={select_on!r} is not produced by this stage.\n"
                    f"  available: {avail}\n"
                    "  Left alone this would early-stop after "
                    f"{self.patience} epochs with no best checkpoint, which is\n"
                    "  indistinguishable from a model that failed to learn.\n"
                    "  Stage 1 (geometry) has no affinity signal: select on "
                    "val_neg_disto_nll.")
            rec = {"epoch": ep, **{f"train_{k}": v for k, v in tr.items()},
                   **{f"val_{k}": v for k, v in va.items()}}
            if hasattr(self.model, "energy") and self.model.energy is not None:
                rec |= self.model.energy.weight_report()
            self.history.append(rec)
            score = va.get(select_on, -np.inf)
            print(f"  ep {ep:>3}  loss {tr.get('loss', 0):.4f}  "
                  f"val r {va.get('val_pearson', float('nan')):+.3f}  "
                  f"rho {va.get('val_rho', float('nan')):+.3f}  "
                  f"rmse {va.get('val_rmse', float('nan')):.3f}  "
                  f"ece {va.get('disto/ece', float('nan')):.3f}  "
                  f"[{tr['sec']:.0f}+{va['sec']:.0f}s]", flush=True)
            self.save("last.pt", ep)
            if np.isfinite(score) and score > self.best:
                self.best, bad = score, 0
                self.save("best.pt", ep)
            else:
                bad += 1
                if bad >= self.patience:
                    print(f"  early stop: no {select_on} improvement in {bad} epochs")
                    break
        (self.out / "history.json").write_text(json.dumps(self.history, indent=1,
                                                          default=float))
        return {"best": self.best, "epochs_run": len(self.history),
                "select_on": select_on, "loss_weights": asdict(self.w)}

    @torch.no_grad()
    def predict(self, loader):
        from ..data.datasets import to_torch
        import pandas as pd
        self.model.eval()
        rows = []
        for batch in loader:
            if not batch:
                continue
            b = to_torch(batch, self.device)
            with torch.amp.autocast("cuda", enabled=self.amp):
                out = self.model(b)
            rows.append(pd.DataFrame(dict(
                pair_idx=b["pair_idx"].cpu().numpy(),
                protein_idx=b["protein_idx"].cpu().numpy(),
                y=b["y"].float().cpu().numpy(),
                pred=out.pK.float().cpu().numpy(),
                logvar=out.logvar.float().cpu().numpy())))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    @torch.no_grad()
    def decompose(self, loader, max_batches: int = 4) -> dict:
        """Per-term energies for the Section 9 physics diagnostics."""
        from ..data.datasets import to_torch
        self.model.eval()
        acc: dict[str, list] = {}
        for i, batch in enumerate(loader):
            if not batch or i >= max_batches:
                break
            out = self.model(to_torch(batch, self.device))
            if out.energy is None:
                return {}
            for k, v in out.energy.per_term.items():
                acc.setdefault(k, []).append(v.float().cpu().numpy())
            acc.setdefault("contact_count", []).append(
                out.energy.contact_count.float().cpu().numpy())
            for k, v in drift_report(out.theta_i, out.prior_i, out.site_mask).items():
                acc.setdefault(k, []).append(np.array([v]))
        return {k: float(np.concatenate(v).mean()) for k, v in acc.items()}
