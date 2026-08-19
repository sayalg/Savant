#!/usr/bin/env python
"""
Step 05 - Train the cross-attention classifier and regressor.

Same two-stage flow as everything else in v2 and as v1 before it: one model
decides binder vs nonbinder, a second predicts pKd for what passes. They are
separate models with separate weights, written as attention_classifier.pkl and
attention_regressor.pkl.

The split is protein-grouped. With a model this expressive over 47,619
complexes on 11,965 proteins, a random split would let it memorise targets and
report a number that means nothing on a new one.

    python 05_train_attention.py --examples work/attention_training
    python 05_train_attention.py --examples work/attention_training --epochs 30 --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from savant2.attention import AttentionModel, build_module, collate   # noqa: E402
from savant2.metrics import (METRIC_ORDER, classification_metrics,    # noqa: E402
                             format_table, regression_metrics)

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
MODELS = ROOT / "models"


def load_examples(path: Path):
    import joblib
    shards = sorted(path.glob("*.joblib")) if path.is_dir() else [path]
    if not shards:
        raise SystemExit(f"no example shards in {path} - run 04_build_attention_examples.py")
    examples, esm_dim = [], None
    for shard in shards:
        blob = joblib.load(shard)
        examples.extend(blob["examples"])
        esm_dim = blob["esm_dim"]
    return examples, esm_dim


def build_split(examples, mode: str, manifest: str, protein_frac: float,
                scaffold_frac: float, seed: int):
    """Held-out fold, grouped by protein alone or by protein AND scaffold.

    The protein-only mode is what produced this model's r 0.598 held-out score
    against 0.269 on the external benchmark. The double-disjoint mode reproduces
    the benchmark's conditions - its ligands are gated at Tanimoto <= 0.50, so
    holding out proteins alone leaves the ligand axis free to leak.
    """
    import pandas as pd

    from savant2.splits import (describe, double_disjoint_split,
                                murcko_scaffolds, protein_split)

    proteins = np.array([e["protein_group"] for e in examples])
    if mode == "protein":
        train_idx, test_idx, stats = protein_split(proteins, test_size=0.2, seed=seed)
    else:
        table = pd.read_csv(manifest, usecols=["id", "smiles"])
        smiles_by_id = dict(zip(table.id, table.smiles))
        missing = sum(1 for e in examples if e["id"] not in smiles_by_id)
        if missing:
            print(f"  WARNING: {missing} examples have no SMILES in {manifest}; "
                  "they group as ACYCLIC", flush=True)
        scaffolds = murcko_scaffolds([smiles_by_id.get(e["id"], "") for e in examples])
        print(f"  {len(np.unique(scaffolds))} distinct scaffolds", flush=True)
        train_idx, test_idx, stats = double_disjoint_split(
            proteins, scaffolds, protein_test_frac=protein_frac,
            scaffold_test_frac=scaffold_frac, seed=seed,
        )
    print(describe(stats), flush=True)
    return ([examples[i] for i in train_idx],
            [examples[i] for i in test_idx], stats)


def run_epoch(module, examples, esm_dim, task, device, optimiser=None,
              batch_size=32, shuffle=True):
    import torch
    import torch.nn.functional as F

    training = optimiser is not None
    module.train(training)
    order = np.random.permutation(len(examples)) if shuffle else np.arange(len(examples))
    total_loss, preds, targets = 0.0, [], []

    for start in range(0, len(order), batch_size):
        chunk = [examples[i] for i in order[start:start + batch_size]]
        batch = collate(chunk, esm_dim, device)
        if task == "classification":
            y = torch.tensor([float(e["is_binder"]) for e in chunk], device=device)
        else:
            y = torch.tensor([float(e["pkd"]) for e in chunk], device=device)

        with torch.set_grad_enabled(training):
            out, _ = module(batch)
            out = out.squeeze(-1)
            if task == "classification":
                # pos_weight corrects the 94/6 binder split; without it the model
                # learns to answer "binder" and stops.
                pos = max(float(y.sum()), 1.0)
                weight = torch.tensor((len(y) - pos) / pos, device=device).clamp(0.2, 20.0)
                loss = F.binary_cross_entropy_with_logits(out, y, pos_weight=weight)
            else:
                # Huber: the label carries assay noise and a handful of extreme
                # pKd values, and squared error would let those dominate.
                loss = F.huber_loss(out, y, delta=1.0)

        if training:
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimiser.step()

        total_loss += float(loss.detach()) * len(chunk)
        preds.append(out.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    return total_loss / max(len(order), 1), np.concatenate(preds), np.concatenate(targets)


def train_one(task, train, valid, esm_dim, args):
    import torch

    module = build_module(esm_dim, d_model=args.d_model, n_heads=args.n_heads,
                          n_layers=args.n_layers, dropout=args.dropout,
                          n_outputs=1).to(args.device)
    optimiser = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    n_params = sum(p.numel() for p in module.parameters())
    print(f"\n=== {task}: {n_params/1e6:.2f}M parameters, "
          f"{len(train)} train / {len(valid)} held-out complexes ===")

    best_score, best_state, patience = -np.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, _, _ = run_epoch(module, train, esm_dim, task, args.device,
                                     optimiser, args.batch_size)
        _, pred, true = run_epoch(module, valid, esm_dim, task, args.device,
                                  None, args.batch_size, shuffle=False)
        scheduler.step()

        if task == "classification":
            proba = 1 / (1 + np.exp(-pred))
            stats = classification_metrics(true.astype(bool), proba >= 0.5, proba)
            score = stats.get("roc_auc", stats["accuracy"])
            summary = f"auc {score:.3f}  recall {stats['binder_recall']:.3f}"
        else:
            stats = regression_metrics(true, pred)
            score = stats["pearson_r"]
            summary = (f"r {stats['pearson_r']:.3f}  rho {stats['spearman_rho']:.3f}  "
                       f"rmse {stats['rmse']:.3f}  mae {stats['mae']:.3f}")

        marker = ""
        # An epoch whose predictions are constant scores nan, which must never
        # become best_score: every later `score > nan` is False, so the run would
        # silently keep the first epoch's weights forever. nan epochs are skipped
        # for selection, but the first one still leaves a loadable fallback.
        snapshot = lambda: {k: v.detach().cpu().clone()
                            for k, v in module.state_dict().items()}
        if score == score and score > best_score:
            best_score, patience = score, 0
            best_state = snapshot()
            marker = "  *"
        else:
            if best_state is None:
                best_state = snapshot()
            patience += 1
        print(f"  epoch {epoch:3d}  loss {train_loss:.4f}  {summary}"
              f"  ({time.perf_counter()-started:.0f}s){marker}", flush=True)
        if patience >= args.patience:
            print(f"  early stop: no improvement in {args.patience} epochs")
            break

    module.load_state_dict(best_state)
    _, pred, true = run_epoch(module, valid, esm_dim, task, args.device, None,
                             args.batch_size, shuffle=False)
    if task == "classification":
        proba = 1 / (1 + np.exp(-pred))
        final = classification_metrics(true.astype(bool), proba >= 0.5, proba)
        print("  held-out:", {k: round(v, 3) if isinstance(v, float) else v
                              for k, v in final.items()})
    else:
        final = regression_metrics(true, pred)
        # Name the split that actually ran: this line is what gets pasted into a
        # report, and "protein-grouped" under a protein+scaffold run is exactly
        # the kind of mislabel that survives into a figure caption.
        print(f"  held-out ({args.split_mode.replace('_', ' + ')} grouped):")
        print(format_table([final], METRIC_ORDER))

    wrapper = AttentionModel(
        esm_dim=esm_dim, task=task, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
        state_dict=best_state,
        metadata={"task": task, "held_out": final, "n_train": len(train),
                  "n_valid": len(valid), "n_params": int(n_params),
                  "epochs_run": epoch, "esm_dim": esm_dim},
    )
    return wrapper


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", default=str(WORK / "attention_training"))
    ap.add_argument("--out-dir", default=str(MODELS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128, dest="d_model")
    ap.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    ap.add_argument("--n-layers", type=int, default=2, dest="n_layers")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--task", choices=["both", "classification", "regression"], default="both")
    ap.add_argument("--split-mode", choices=["protein", "protein_scaffold"],
                    default="protein", dest="split_mode",
                    help="protein_scaffold holds out proteins AND Bemis-Murcko "
                         "scaffolds, reproducing the benchmark's disjointness")
    ap.add_argument("--manifest", default=str(WORK / "manifest_training.csv"),
                    help="source of SMILES for scaffold grouping")
    ap.add_argument("--protein-frac", type=float, default=0.30, dest="protein_frac")
    ap.add_argument("--scaffold-frac", type=float, default=0.30, dest="scaffold_frac")
    args = ap.parse_args()

    import torch
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    examples, esm_dim = load_examples(Path(args.examples))
    print(f"{len(examples)} examples, ESM dim {esm_dim}, device {args.device}")
    train, valid, split_stats = build_split(
        examples, args.split_mode, args.manifest,
        args.protein_frac, args.scaffold_frac, seed=0,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("both", "classification"):
        model = train_one("classification", train, valid, esm_dim, args)
        model.save(out_dir / "attention_classifier.pkl")
        print(f"  wrote {out_dir / 'attention_classifier.pkl'}")

    if args.task in ("both", "regression"):
        # Stage 2 sees binders only, matching v1's flow.
        train_b = [e for e in train if e["is_binder"]]
        valid_b = [e for e in valid if e["is_binder"]]
        model = train_one("regression", train_b, valid_b, esm_dim, args)
        model.save(out_dir / "attention_regressor.pkl")
        print(f"  wrote {out_dir / 'attention_regressor.pkl'}")


if __name__ == "__main__":
    main()
