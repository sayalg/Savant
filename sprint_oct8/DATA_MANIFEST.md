# DATA_MANIFEST

Phase 0 verification checklist (PLAN.md Appendix A). Every row starts
UNVERIFIED. Fill in the result and the date; do not delete rows.

Section 0.4.10: preprint and vendor numbers are not ground truth, and no
dataset or benchmark fact may be fabricated. An empty cell is an honest
answer; a plausible guess is not.

| # | Claim | Evidence | Phase 0 action | Status | Finding | Checked |
|---|---|---|---|---|---|---|
| 1 | Boltz-2 FEP+ avg Pearson ~0.62 (0.66 on 4 targets) vs FEP+ ~0.72; affinity output is log10 IC50 (uM); binder-probability head; MIT; v2.1.x current | PP/V + repo | Read the paper and repo; confirm units, training cutoffs, version | UNVERIFIED |  |  |
| 2 | Independent Boltz-2 critiques: external mean ~0.42; poor on molecular-glue ternary complexes; a reviewer flagged a missing scaffold split | V, review comments | Re-read sources; treat as indicative only | UNVERIFIED |  |  |
| 3 | PIGNet2: Morse-type vdW, NO explicit Coulomb, CASF-2016 R~0.747 / Spearman ~0.651 / SR1 ~93%, docked+cross-docked augmentation, open repo | PR | Read paper and repo; confirm terms and numbers | UNVERIFIED |  |  |
| 4 | GEMS/CleanSplit: ligand-only RMSE 1.424 on original PDBbind; ~49% of CASF test points leakage-affected; GEMS on CleanSplit -> CASF-2016 r 0.803 | PR | Read paper; reproduce split filtering if feasible | UNVERIFIED |  |  |
| 5 | HonestAffinity strict tiers: pose-free r ~0.43-0.53, RMSE ~1.5-1.67; ESM-2 REDUCES Pearson on every strict tier (-0.024/-0.042/-0.064) | PP | Re-derive with our own baselines; do not rely on the decimals | UNVERIFIED |  |  |
| 6 | PSICHIC 0.478 on Atom3D LBA 30%; IPBind 0.732 | PP | Optional check | UNVERIFIED |  |  |
| 7 | TankBind ~0.6 on a PDBbind time split; E3Bind documents knotted distance maps | PR/PP | Optional check | UNVERIFIED |  |  |
| 8 | BindingDB ~560K Ki, ~100K Kd, ~1.8M IC50; TDC split 375,032 Ki / 52,284 Kd | PR/D | RECOUNT from the current download - phase1 prints the real numbers | UNVERIFIED |  |  |
| 9 | PDBbind v2020 is the last freely available release; later versions restricted; PDBBind-opt is not redistributable | D | Read the licence terms | UNVERIFIED |  |  |
| 10 | LP-PDBBind CC-BY 4.0; ~10,980/2,312/4,651 train/val/test | PR/D | Verify | UNVERIFIED |  |  |
| 11 | PLINDER ~449k systems, CC-BY, structure resource not an affinity source | D | Verify affinity content before relying on it | UNVERIFIED |  |  |
| 12 | BindingNet v1/v2 Kd/Ki fraction | UNVERIFIED | COUNT Ki/Kd rows before use - the plan flags this as unverified | UNVERIFIED |  |  |
| 13 | SAIR: 5.24M structures, ~1.05M systems, IC50-type, ~3% anomalies, CC BY 4.0 | PP/D | Verify licence and label types; mostly excluded by the label policy | UNVERIFIED |  |  |
| 14 | Papyrus/Papyrus++ keeps Ki/Kd; ChEMBL 34; reproducibility threshold corrected to 0.5 log units, which shrinks the size | PR/D | Verify the current release | UNVERIFIED |  |  |
| 15 | DES370K SAPT0 components; DES5M energies only; Splinter ~1.7M configs of >9,000 dimers with SAPT0 | PR/D | Verify the components BEFORE designing the Stage 0 losses | UNVERIFIED |  |  |
| 16 | ESM-C licences: 300M open (Cambrian), 600M NON-COMMERCIAL, 6B API-only; ESM-2 ~1022-token limit; Uni-Mol MIT | D | Verify licences; Section 0.4.8 blocks 600M/6B without confirmation | UNVERIFIED |  |  |
| 17 | MISATO ~20,000 complexes, MD + QM | PR/D | Verify contents | UNVERIFIED |  |  |
| 18 | Noise ceiling: ~0.5 log inter-lab scatter; 6-8% of curated matched pairs differ by >1 unit | PR + estimate | QUANTIFY on our own duplicates - harmonize.noise_ceiling() does this | UNVERIFIED |  |  |

## Dataset inventory

Filled in by `scripts/phase1_harmonize.py` (per-source row counts,
unique proteins and ligands, Kd vs Ki, overlap, drop reasons).
