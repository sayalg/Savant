# Pose-free, physics-informed protein–ligand affinity model: implementation plan (v3, merged with research review, HPC edition)

Hand-off document for Claude Code. Read all of it before starting. Items marked **[USER INPUT]** need an answer from the user; do not guess them. Items marked **(verify)** come from a literature/dataset review whose claims have not been independently re-checked; Phase 0 re-verifies them (Appendix A lists the claims and their evidence type).

---

## 0. Read this first: verdict, expectations, and guardrails

### 0.1 Verdict from the feasibility review

- The plan is technically coherent and almost every component already exists as open tooling (Section 3). The novel part is the *combination*: physics evaluated as an expectation under predicted residue-site × ligand-atom distance distributions, with force-field-like parameters produced from PLM/CLM embeddings. Close precedents exist (TankBind, DeepDock, AP-Net/Splinter, PIGNet) but none is an exact match.
- **The goal (comparable to PIGNet2 and Boltz-2 with no folding at inference) is a hypothesis, not a forecast.** Tier 1 (beat a matched no-physics control) is plausible. Tier 2 (match PIGNet2 with Boltz-2 poses) is a stretch. Tier 3 (match Boltz-2) is very unlikely without a geometry signal. See Section 0.2.
- **The biggest risk is data and evaluation, not architecture.** Leakage-controlled splits collapse most published performance, PLM embeddings can *hurt* on strict splits, and the Kd/Ki-only rule removes most of the affinity data used by Boltz-2 and SAIR.
- **The HPC allocation (up to 12 large GPUs concurrently) removes the compute constraint but not these risks.** Use the extra compute to run more variants, ablations, seeds, and sweeps in parallel. Do not use it to skip the baseline-first gates, and do not let it create untracked forking paths (Section 7.4).

### 0.2 Expected performance (evidence and planning ranges)

| Setting | Reported result | Evidence type |
|---|---|---|
| Pose-free PLM+CLM-style models on leakage-controlled splits (LP-PDBBind strict tiers) | Pearson ≈ 0.43–0.53, RMSE ≈ 1.5–1.67 pK | preprint (HonestAffinity, arXiv 2606.03422) |
| Sequence-based PSICHIC on Atom3D LBA, 30% identity | Pearson 0.478 | reported in a preprint (IPBind, arXiv 2504.16261) |
| ESM-2 features on strict tiers | Help on canonical/CASF splits but **reduce** Pearson on every strict tier (−0.024 / −0.042 / −0.064 for tiers 1/2/3) | preprint |
| Ligand-only model on original PDBbind | RMSE 1.424, better than Vina and several DL scorers; collapses on the leakage-controlled CleanSplit | peer-reviewed (GEMS, Nature Machine Intelligence 2025) |
| Leakage in CASF | ≈49% of CASF test points affected | same |
| GEMS trained on CleanSplit, tested on CASF-2016 | Pearson 0.803, RMSE 1.308 (retrained GenScore 0.780/1.362; Pafnucy 0.746/1.484) | peer-reviewed |
| Structure-based IPBind on LBA 30% | Pearson 0.732 | preprint |
| PIGNet2 | CASF-2016 scoring R ≈ 0.747; ranking Spearman ≈ 0.651; docking SR1 ≈ 93%; derivative benchmarks avg R ≈ 0.64 (2015 set), ≈ 0.43 (2020 set) | peer-reviewed (Digital Discovery); verify numbers |
| TankBind | Pearson ≈ 0.6 on a PDBbind time split; independent work (E3Bind) documents "knotted," geometrically invalid distance maps | peer-reviewed / preprint |
| Boltz-2 | FEP+/OpenFE avg Pearson ≈ 0.62 (0.66 on a 4-target subset) vs FEP+ ≈ 0.72. Independent reports: mean ≈ 0.42 in an external comparison; poor or negative correlations on molecular-glue ternary complexes; a reviewer flagged a missing scaffold split and leakage concerns | vendor/preprint + independent blogs and reviews |
| Experimental noise ceiling | ≈0.5 log unit inter-lab scatter; 6–8% of matched pairs differ by >1 unit even after curation; practical Pearson ceiling ≈ 0.8–0.9 on curated data and lower on pooled data | literature + estimate |

**Planning ranges (judgment, not literature; revise after Phase 6 baselines):**

- No-physics fusion control on protein-cluster/strict splits: Pearson ≈ 0.40–0.55, RMSE ≈ 1.5–1.7 pK.
- Physics model gain over that control: small (roughly +0.00 to +0.05 Pearson) with wide uncertainty, including a real chance of no gain.
- Within-target Spearman: roughly 0.3–0.6, strongly target-dependent, and lower for narrow congeneric series.
- Odds: Tier 1 roughly even; Tier 2 low; Tier 3 very low. If Tiers 2–3 fail, the value case is speed/cost and interpretable decompositions, which must then be measured (Section 9).

### 0.3 DO

1. **Build honest splits and all baselines first** (Wave 1). Every later claim is relative to them.
2. Report the **ligand-only and target-mean baselines in every results table.** If ligand-only is within ≈0.02 Pearson of the fusion model on a split, that split has a ligand shortcut and needs a harder variant.
3. **Treat the PLM as an empirical question.** Run PLM-off, full-embedding, low-rank-bottleneck, and pocket-windowed variants on strict splits early (Section 6.7).
4. Keep the **distogram cross-entropy anchor, triangle penalty, contact prior, and calibration monitoring on from the first affinity step.**
5. Add each physics term through an **ablation gate**; keep only terms that pass.
6. Use **oracle-pose mode** to separate geometry error from physics error (Section 6.6).
7. **Pre-register** the success tiers, the sweep search spaces, and the benchmark protocol (Section 7.4). Log every configuration trained.
8. Quantify **overlap between the benchmark and Boltz-2's training sources** and report the low-overlap subset.
9. **Re-verify every "(verify)" fact** in Phase 0 and record the result in `DATA_MANIFEST.md`.
10. Record **inference cost** for every model so the speed value case can be evaluated.

### 0.4 DON'T

1. **Never use IC50 or EC50** in any form: not as labels, auxiliary heads, features, sample weights, filters, or model-selection criteria. Never use SAIR's or Boltz-2's affinity outputs or labels as training signal, features, or distillation targets, because they are IC50-dominated or IC50-trained. Boltz-2 is a *baseline only* and a possible *geometry-only* teacher.
2. **Never run a co-folding or structure trunk at inference** (Boltz/AF3/Chai/Protenix, including using Boltz-2 trunk embeddings). Do not dock at inference.
3. **Never tune, early-stop, or select models on the sealed benchmark.** It is scored once per model version.
4. **Never headline random-split or original-PDBbind/CASF numbers.** Headline numbers come from protein-cluster, scaffold, time, LP-PDBBind, CleanSplit, and PLINDER splits.
5. **Never add per-target bias terms or target IDs** to improve fit; that makes the model useless on new proteins.
6. **Do not build the full stack before the Tier 1 gate on the MVP.** Stage 0 pretraining can run in parallel as an ablation, but Coulomb/aromatic/halogen/entropy terms, teacher distillation, and PLM fine-tuning are Wave 3 (gated).
7. **Do not add electrostatics or other terms on faith.** PIGNet2 omits explicit Coulomb and still performs well; a directionless screened Coulomb may add noise.
8. **Do not use ESM-C 600M or 6B** unless the user confirms a non-commercial context (Section 3, license). Default to ESM-2 650M and ESM-C 300M.
9. **Do not redistribute PDBbind-derived processed data**, and do not assume PDBbind versions after 2020 are freely usable.
10. **Do not treat preprint or vendor numbers as ground truth**, and do not fabricate any dataset or benchmark fact.
11. **Do not claim physical interpretability** without the diagnostics in Section 9.
12. Do not let compute abundance replace reasoning: more GPUs do not fix leakage, label noise, or missing geometry information.

### 0.5 Working rules for Claude Code

1. Work wave by wave and phase by phase. Each phase ends with acceptance criteria. At the end of a phase, write `reports/phase_N.md` with actual numbers, then stop and check in if any gate failed.
2. Build small first: get the whole pipeline (data → embeddings → model → metrics) running end to end on a ≤5k-complex subset before scaling.
3. Everything is config-driven (Hydra/YAML), seeded, and logged. Cache expensive computations to disk with content hashes.
4. Write unit tests for every physics function (Section 6.5) and for label and leakage policies.
5. Report negative results honestly and keep every ablation table.
6. **Label policy:** the target is pKd, trained on Kd and Ki only. Enforce with `tests/test_labels.py` at ingestion and in CI.
7. Ask the user for anything marked [USER INPUT]. Do not proceed past a gate that failed without reporting.

---

## 1. Goal, constraints, and compute

- **Input at inference:** an amino acid sequence and a ligand SMILES. No 3D structure, no docking, no folding.
- **"No folding" definition (assumed; confirm in Section 13):** no co-folding or structure-trunk computation at inference, including trunk embeddings from a co-folding model, and no docking. A cached or distilled student that runs none of these is allowed. Apo protein structure prediction (ESMFold/AlphaFold DB) is **excluded by default** and appears only as a flagged optional extension.
- **Output:** predicted pKd (ΔG = −1.364·pK kcal/mol at 298.15 K), uncertainty, and an interpretable decomposition: per-term, per-residue, per-ligand-atom contributions and predicted pocket residues.
- **Comparison:** the user's Boltz-2 scores on their benchmark **[USER INPUT: path/format, composition, affinity types, date range]**, PIGNet2 under several pose conditions, and other existing baselines (Section 3).
- **Aim:** performance comparable to PIGNet2 and Boltz-2 without a folding step. Success is defined in tiers (Section 9).
- **Priority:** incorporate as much physics as possible *and show by ablation which physics helps*.
- **Label policy:** target = pKd = −log10(Kd in M). Ki pooled with Kd as pKd-equivalent by default (`label_types: [Kd, Ki]`); `label_types: [Kd]` gives a strict Kd-only model. IC50/EC50 are excluded entirely.
- **Data strategy:** pull from many databases; pretrain on large, noisier, modeled-structure and affinity-only data; fine-tune on crystal-structure complexes.
- **Compute:** HPC with up to 12 large GPUs concurrent. **[USER INPUT: GPU model/VRAM, scheduler (assumed SLURM), CPU-node availability, storage quotas and scratch, wall-time limits, internet access from compute nodes, container support (assumed Apptainer), total GPU-hour allowance.]** Section 7 defines how the compute is used.

---

## 2. Core idea and design variants

The protein LM (PLM) and chemistry LM (CLM) supply *what each site is like*. A fixed, differentiable physical functional form supplies *how sites interact* as a function of distance. Since there is no pose at inference, the model predicts **distance distributions** between protein sites and ligand atoms, and the physics is evaluated as an **expectation under those distributions**.

```
sequence ──► ESM (frozen) ──► residue embeddings ─┐
                                                   ├─► pocket head (which residues contact?)
SMILES ──► RDKit/xTB features ─┐                   │        │ top-K residues
        └─► Uni-Mol (frozen) ──┴► ligand atom emb ─┘        ▼
                                              pair module (cross-attention + pair rep z_ij)
                                                            │
                                    distogram P_ij(bin)  ◄──┴──►  physical parameter heads
                                    (residue site i, ligand atom j)     θ_i, θ_j (charge, ε, r0, donor/acceptor,
                                                            │            hydrophobicity, aromaticity, ...)
                                                            ▼
                     ⟨e_t,ij⟩ = Σ_b P_ij(b) · e_t(d_b; θ_i, θ_j)   for each physical term t
                                                            ▼
        ΔG_pred = Σ_t w_t ⟨E_t⟩ + desolvation + conformational terms + (optional) residual − scale/offset
                                                            ▼
                                 pK_pred, uncertainty, decompositions
```

### 2.1 Core failure mode and required mitigations

Independent per-pair distance distributions are not guaranteed to be geometrically consistent, and the affinity loss can push every pair toward its energy minimum, giving over-attractive, unphysical predictions. Independent work on TankBind-style distance maps documents knotted, invalid geometries. Mitigations (all required for Variant A):

- Keep a **supervised distogram cross-entropy anchor** on structural data throughout training, including the affinity stage (mixed batches).
- Add a **triangle-inequality consistency penalty** using ligand conformer-ensemble distances (for ligand atoms j, k and site i: |E[d_ij] − E[d_ik]| ≤ max ligand distance d_jk).
- Add a **contact-count prior** (expected contacts per ligand atom should match crystal statistics).
- Monitor **distogram calibration on held-out crystal complexes in every later stage.** If calibration degrades, the physics is being gamed.

### 2.2 Geometry variants (run both in Wave 2)

- **Variant A (independent distributions + penalties):** the design above.
- **Variant B (geometry-consistent by construction):** a lightweight coordinate head predicts a small set of K pocket-frame hypotheses (coordinates for the top-K residue sites and the ligand atoms, ligand internal geometry constrained by the conformer ensemble). Distances, and therefore energies, come from these coordinates, and the energy is the mixture-weighted average over hypotheses. Supervised with the crystal-pose distogram/coordinate losses (winner-take-all plus soft mixture weights). Consistent by construction, at the cost of a more complex module. Treat as a research-grade variant; keep the interface identical to A (the energy module only sees per-hypothesis distances or distributions).
- **Variant C (no physics):** matched embedding-fusion control (Section 8, Phase 6).
- **Variant D (oracle):** real coordinates (Section 6.6).

### 2.3 Relationship to PIGNet/PIGNet2, TankBind, and Boltz-2 **(verify)**

| | PIGNet / PIGNet2 | TankBind | This project | Boltz-2 |
|---|---|---|---|---|
| Input | 3D pocket + ligand pose | Protein structure + ligand (pocket blocks) | Sequence + SMILES | Sequence + SMILES |
| Geometry | Real coordinates at atom level | Predicted protein–ligand distance map from a predicted pose | Predicted distance distributions (A) or coordinate hypotheses (B) | Co-folded structure, then affinity module |
| Physics terms | vdW (Morse-type, not Lennard-Jones), H-bond, metal, hydrophobic, rotor entropy proxy; **no explicit Coulomb** | None (learned) | vdW, H-bond, hydrophobic, rotor (MVP) plus gated Coulomb, aromatic, halogen, desolvation, entropy | Learned, not explicitly physical |
| Training signal | PDBbind-scale, with docked/cross-docked decoy augmentation | PDBbind | Multi-source Kd/Ki + geometry + QM dimer physics | Large mixed Ki/Kd/IC50 data, IC50-like output |
| Code | Open (ACE-KAIST/PIGNet2) | Open | To build | MIT (weights and code) |

PIGNet is not a pose-free competitor: the pose carries much of its signal. The genuine differences here are (1) physics evaluated on inferred geometry, which removes the folding step but adds pose-uncertainty failure modes (errors become distribution errors, and averaging over broad distributions smears steep terms such as short-range repulsion), and (2) representation from large-scale language-model pretraining plus QM dimer data. Borrow PIGNet2's Morse-type vdW option and its decoy/pose augmentation for the oracle-mode teacher.

---

## 3. Existing tools: reuse, adapt, or build

License and access statements come from the review and must be verified in Phase 0.

| Tool / resource | What it gives | License / access **(verify)** | Recommendation |
|---|---|---|---|
| ESM-2 (650M; 3B for ablation) | Per-residue embeddings; ~1022-token limit | Open | **Reuse** |
| ESM-C 300M / 600M / 6B | Embeddings (Dec 2024 release) | 300M: Cambrian Open License (commercial OK); 600M: non-commercial; 6B: API only | **Reuse 300M**; 600M only if non-commercial |
| Uni-Mol / Uni-Mol2 | Atom-level molecular embeddings; pocket model | MIT | **Reuse** for ligand atoms |
| RDKit, xTB, AIMNet2, MACE-OFF | Conformers, charges, solvation, energies | Open | **Reuse** (CPU nodes; AIMNet2 on GPU) |
| PIGNet2 | Physics-informed pose-based affinity | Open source | **Reuse as baseline; adapt** Morse vdW and augmentation ideas |
| TankBind | Distance-map-to-affinity | Open | **Adapt ideas; run as a baseline if feasible** |
| DeepDock | Statistical potential over predicted distance distributions | Open | Ideas only |
| AP-Net / Splinter | SAPT0-based physics-aware learning; ~1.7M dimer configs of >9,000 protein–ligand-relevant dimers | Open | **Adapt** as Stage 0 precedent and data |
| AEV-PLIG | Ligand-centric GNN with augmentation, FEP-competitive | BSD-3 | **Baseline** |
| GEMS + CleanSplit | Leakage-controlled PDBbind split and GNN baseline | Open (peer-reviewed) | **Reuse** splits, evaluation, baseline |
| PLINDER | ~449k protein–ligand systems, similarity annotations, leakage-aware splits and eval harness | CC-BY | **Reuse** for geometry and splits. It is a structure/pose resource, not a primary affinity-label source |
| LP-PDBBind | Leakage-controlled PDBbind reorganization | CC-BY 4.0 | **Reuse** splits |
| HiQBind | Reprocessed complexes and workflow | Open workflow; PDBBind-opt not redistributable | **Reuse** workflow for structure cleaning |
| Papyrus / Papyrus++ | Standardized bioactivity; Papyrus++ keeps Ki/Kd intact | Open (2024 release on ChEMBL 34) | **Reuse** as a curated Kd/Ki source |
| TDC / DeepPurpose | Dataset loaders; DTA baselines | Open | **Reuse** loaders; baselines |
| ConPLex, PLAPT, CS-DTA-style models | Pose-free PLM/CLM affinity predictors | Varies | **Reproduce as baselines** where code exists; CS-DTA is close to the no-physics control |
| Boltz-2 (v2.1.x) | Co-folding + affinity module (log10 IC50-like output, binder probability) | MIT | **Baseline; optional geometry-only teacher.** Never as a label/feature/distillation source |
| "Affinity fine-tuning of Boltz-2" framework | Freeze trunk, train affinity head | Open (preprint) | **Optional strong baseline:** fine-tune on this project's Kd/Ki training split and score on the same splits (it needs the folding trunk, so it is a with-folding reference) |
| FlowDock, IPBind, Chai/Protenix/OpenFold3 | Alternative structure/affinity systems | Varies | Optional comparators |
| PoseBusters, MMseqs2, Dimorphite-DL | Pose validity, clustering, protonation | Open | **Reuse** |

Build from scratch: the harmonization pipeline for Kd/Ki, the expected-energy physics module and parameter heads, the pair/distogram/pocket modules, Variant B, the oracle mode, the evaluation and leakage tests.

---

## 4. Data sources

Pull from all of these that are feasible and consistent with the label policy. Everything is harmonized into one schema (Section 5).

### 4.1 Affinity data (Kd and Ki only; IC50/EC50 are never used)

| Source | Provides | Notes |
|---|---|---|
| ChEMBL | Kd/Ki, target UniProt, assay metadata | Keep standard_type in {Kd, Ki}. Filter confidence_score = 9, validity comments null, relation "=" for the main set; keep ">"/"<" as censored labels. |
| BindingDB | Kd/Ki, many PDB links | Keep Kd and Ki columns only. Large overlap with ChEMBL; dedupe by literature source. |
| Papyrus / Papyrus++ | Pre-curated, standardized ChEMBL + other sources | Papyrus++ keeps Ki/Kd and filters IC50/EC50 (ignore that part). Reported reproducibility threshold corrected to 0.5 log units, which shrinks its size **(verify)**. Use as a cross-check and an additional Kd/Ki source. |
| PDBbind (latest freely available: v2020; later versions restricted) | Crystal complexes + Kd/Ki | Drop IC50 entries from labels. Registration and license terms apply **(verify)**. |
| Binding MOAD | Crystal complexes + affinities | Overlaps PDBbind. |
| BioLiP2 | Biologically relevant complexes; affinity annotations | Also a structure source. |
| PubChem BioAssay | Ki/Kd where target-mapped | Ki/Kd only; low yield, noisy target mapping. |
| IUPHAR/BPS Guide to Pharmacology | Curated Ki/Kd | Small, high quality. |
| PDSP Ki Database | Receptor Ki values | GPCR/transporter coverage. |
| Davis (kinase Kd matrix) | Dense Kd, 400+ kinases | Within-target ranking evaluation; **hold out entirely if used as a test set**. |
| Platinum (mutation effects on affinity) | ΔΔG for point mutants | Evaluation only **(verify availability)**. |

### 4.2 Structure + affinity data (geometry supervision)

| Source | Role |
|---|---|
| PDBbind, Binding MOAD, BioLiP2 | Crystal-structure fine-tuning set (highest weight). |
| HiQBind | >18,000 PDB entries, >30,000 complexes (BioLiP2 + Binding MOAD + BindingDB affinities); preferred cleaned structures **(verify)**. |
| PLINDER | Geometry and split source (systems, similarity annotations); not a primary affinity source **(verify)**. |
| LP-PDBBind / PDBbind CleanSplit | Leakage-controlled splits (Section 5.3). |
| BindingNet (v1/v2) | Modeled complexes with ChEMBL-derived activities. **Kd/Ki fraction is unverified; check before use** and use only Ki/Kd rows. |
| SAIR | 5.24M Boltz-1x-folded structures over ~1.05M systems, labeled IC50-type; ~3% show physical anomalies; CC BY 4.0 per its dataset card **(verify)**. Mostly excluded by the label policy (see below). |
| CrossDocked2020 | ~22.5M docked poses; geometry pretraining only. |
| MISATO | MD and QM data for ~20,000 PDBbind complexes. Use residue–ligand distance **histograms over frames** as soft distogram targets. |

For predicted or docked poses: filter with PoseBusters-style validity and confidence scores, drop clashing poses, weight by pose quality. These data are used in Stages 1–2; the crystal set is used for Stage 3.

**IC50-complex policy.** Any complex whose only affinity record is IC50/EC50 has **no affinity label** here. Default (`use_ic50_only_structures: false`): such complexes are excluded entirely, including from geometry pretraining, so no IC50-associated record enters training. This removes most of SAIR and part of BindingNet/PDBbind. Optional (`use_ic50_only_structures: true`): keep the structures for distogram/pocket supervision only, with the affinity loss masked and the IC50 value never read. Report dataset sizes under both settings.

### 4.3 Physics data (pretraining the functional forms)

| Source | Role |
|---|---|
| Splinter | ~1.7M configurations of >9,000 protein–ligand-relevant dimers with SAPT0 components; **preferred Stage 0 source** **(verify)**. |
| DES370K | CCSD(T)/MP2/HF + SAPT0/aVTZ components (electrostatics/exchange first order; induction/dispersion second order); SAPT0 confirmed **(verify)**. |
| DES5M | SNS-MP2 energies without a full SAPT decomposition; energies only. |
| SPICE | Off-equilibrium geometries, energies, forces for drug-like fragments and dimers. |
| S66x8 / S22 / NCIA-type sets | Non-covalent interaction curves; validation of distance dependence. |
| PLA15 / PLF547-type sets | Protein–ligand fragment interaction energies (few complexes); sanity checks on crystal poses **(verify)**. |
| OMol25 | Optional large-scale atom-level charges/energies **(verify)**. |
| FEP benchmark sets (Wang et al. 2015; Merck; OpenFE) | Within-series ranking evaluation. Test only. |

### 4.4 Expected Kd/Ki data volume **(verify; order of magnitude)**

BindingDB (2024) reportedly holds ≈2.9M measurements: ≈1.8M IC50, ≈560K Ki, ≈100K Kd, ≈220K EC50. A therapeutic-data-commons split lists 375,032 Ki and 52,284 Kd pairs. After the Kd/Ki-only filter and harmonization/dedup against ChEMBL, expect **several hundred thousand** protein–ligand pairs, Ki-dominated. Curated ChEMBL matched pairs still show a 6–8% rate of >1-unit disagreement. Report actual counts per source in Phase 1.

---

## 5. Harmonization and splits

### 5.1 Unified schema (one row per measurement)

```
measurement_id, source_db, source_record_id, doc_id (PMID/DOI),
uniprot, pdb_id (nullable), sequence, sequence_hash, is_mutant, mutation_string,
ligand_smiles_raw, ligand_smiles_std, inchikey_full, inchikey_flat,
type (Ki|Kd only), relation (=|>|<), value_nM, pK, temperature, pH,
assay_id, structure_path (nullable), structure_source (crystal|modeled|cofolded|none)
```

### 5.2 Rules

- **Type filter at ingestion:** rows whose type is not Kd or Ki (IC50, EC50, potency, % inhibition, etc.) are dropped before any other processing. Only counts go to `reports/dropped_types.csv`.
- **Ligands:** RDKit standardization (largest fragment, salts/solvents removed, stereo kept). One fixed protonation policy at pH 7.4 (e.g., Dimorphite-DL major state). Store raw and standardized SMILES. Drop ligands outside a sane range (<5 or >80 heavy atoms), organometallics, and long peptides; log counts.
- **Proteins:** sequence from PDB SEQRES when a structure exists, otherwise UniProt canonical. Mutants: apply the mutation if it parses, otherwise flag `is_mutant` and exclude from the main training set. Handle the ESM-2 window (1022 tokens) by cropping to the domain/pocket region; document the policy and report metrics by length bin.
- **Deduplicate across sources at the measurement level** (doc_id, uniprot, inchikey, value). ChEMBL, BindingDB, PDBbind, and Papyrus overlap heavily, and double counting causes leakage and false confidence.
- **Aggregation:** per (protein, ligand, type) take the median pK. Flag pairs with spread >1 log unit. Keep a cross-type table (Ki vs Kd on the same pair).
- **Labels:** single target **pKd** (Ki pooled when `label_types` includes Ki), one output head. No IC50 head, no Cheng–Prusoff conversion. The measurement type stays as metadata for stratified evaluation and the Kd-only ablation, and is never a model input. Censored Kd/Ki labels get a one-sided loss.
- **Source weights and noise:** start with crystal-linked Kd/Ki = 1.0, ChEMBL/BindingDB Kd/Ki = 0.8, modeled-structure Kd/Ki = 0.5, and use a heteroscedastic loss (learned or per-source noise scale).

### 5.3 Splits (build once, freeze, version-control)

1. **Benchmark quarantine:** remove from all training data every measurement whose protein–ligand pair is in the user's benchmark. For a "strict" variant also remove proteins with ≥30% sequence identity to any benchmark protein **and** ligands with Tanimoto ≥0.9 to benchmark ligands. Report both.
2. **Internal validation/test splits:** random (sanity only), scaffold-grouped, protein-cluster-grouped (MMseqs2 at 30% **and** 50% identity), time-based where dates exist.
3. **Structural data:** use LP-PDBBind (identity <0.5 and ligand similarity <0.99 between train and test; ≈10,980/2,312/4,651 train/val/test), PDBbind CleanSplit, and PLINDER splits so no crystal complex leaks through a near-duplicate. Mirror the strict LP tiers when reporting.
4. **Boltz-2 overlap:** check benchmark proteins and ligands against ChEMBL, BindingDB, PubChem, and MF-PCBA (Boltz-2's reported training sources; it also reportedly retained assays with average ipTM > 0.75 **(verify)**), and note its training cutoffs. Report results on the low-overlap subset separately.
5. **Shortcut check:** compute ligand-only, protein-only, and target-mean baselines on every split (Phase 6). If ligand-only is within ≈0.02 Pearson of the fusion model, tighten the split.

---

## 6. Physics specification

### 6.1 Sites and geometry

- **Ligand sites:** every heavy atom plus ring centroids (for π terms).
- **Protein sites (per residue):** three pseudo-sites: Cα (backbone), side-chain centroid, side-chain terminal functional atom (Gly/Ala masked appropriately); ring centroids for F/Y/W/H. Define exactly in `physics/sites.py`.
- **Distogram:** 2.0–12.0 Å in 0.5 Å bins (20 bins; <2.0 Å falls in the first bin) plus a "far/no contact" bin. Targets from crystal structures or MISATO frame histograms.
- **Hierarchical pruning:** a pocket head predicts per-residue binding-site probability (supervised by residues with any heavy atom within 8 Å of the ligand in crystals). Only the top-K residues (K≈64, config) enter the pair module; others are treated as "far." Top-K selection is hard and detached; gradients flow through the pair module.
- **Residue-level coarse-graining trades directionality for tractability.** Distance-only H-bond and electrostatic terms cannot represent angles or polarization; report failure cases by chemistry.

### 6.2 Energy terms (evaluated at bin centers, then averaged under P_ij)

Every term is a registry entry with an `enabled` flag, a wave, and an ablation record.

| Term | Form (distance-only) | Parameters from embeddings | Wave |
|---|---|---|---|
| van der Waals | Soft-core Lennard-Jones ε_ij[(r0/d)^12 − 2(r0/d)^6] (Lorentz–Berthelot) **or** PIGNet2-style Morse-type; `vdw_form: lj \| morse` is an ablation | ε, r0 per site | MVP (Wave 2) |
| H-bond | Gaussian at ~2.6–3.2 Å × donor·acceptor propensities | donor, acceptor ∈ [0, 1] | MVP |
| Hydrophobic contact | Piecewise-linear (Vina/ChemScore-like) between apolar sites | hydrophobicity ∈ [0, 1] | MVP |
| Rotor/conformational penalty | rotatable bonds × burial | learned weight ≥ 0 | MVP |
| Screened Coulomb | 332·q_i·q_j / (ε(d)·d), ε(d) = 4d or Mehler–Solmajer | bounded partial charge | Wave 3, gated |
| Aromatic: π–π, cation–π | ring-centroid distance terms | aromaticity, cation/anion flags | Wave 3, gated |
| Halogen bond | ligand halogens vs acceptor sites | acceptor propensity | Wave 3, gated |
| Desolvation | ligand solvation free energy (6.3) × burial b_j, plus charged-residue desolvation × burial; b_j from expected contacts under P | burial from P | Wave 3, gated |
| Geometric entropy | −T·H[P] | weight in [0, 1] | Wave 3, gated; high risk of gaming, always ablate |

Assembly: `ΔG = Σ_t softplus(w_t)·E_t + desolv + conf + b`, then `pK_pred = −ΔG/(RT ln10)` with learned scale and offset (the model energy is not literally ΔG°). Attractive-term weights are positive-constrained. An optional **residual head** captures non-pairwise effects, with a clamp and dropout. Always report physics-only vs physics+residual and the variance fraction carried by the residual. The offset is global; never per target.

### 6.3 Ligand-side physics features (no pose needed)

- **Tier A (always):** RDKit Crippen logP/MR, TPSA, polar/apolar SASA, rotor count, formal charge, MMFF/Gasteiger charges, ring/aromatic flags, ETKDG conformer ensemble (N≈10–20) with MMFF energies, and conformer-ensemble internal-distance histograms (geometric prior in the pair module and for the triangle penalty).
- **Tier B:** AIMNet2 or GFN2-xTB partial charges, dipole, polarizability, conformer energy spread.
- **Tier C:** xTB ALPB/GBSA solvation free energy. With CPU nodes available, compute for all ligands; otherwise train a surrogate on a computed subset.

Compute Tiers A–C in Wave 1 in the background (cheap on HPC) and treat them as ablation factors (A vs A+B vs A+B+C).

### 6.4 Parameter priors

Initialize and regularize the parameter heads toward tabulated force-field values: AMBER ff14SB-style residue-site charges and LJ types for protein sites; GAFF2/OpenFF or AIMNet2/xTB charges for ligand atoms. Use an annealed L2 penalty and log the drift of parameter distributions per stage.

### 6.5 Physics and policy unit tests (`tests/`)

- Every term is finite and differentiable over the distance range, including d → 0 (soft-core).
- Repulsive at short range, decaying to zero at long range, with the smooth switch verified.
- Coulomb sign law (when enabled): opposite charges attract, like charges repel, magnitude scales with the charge product.
- Invariance to ligand atom permutation and to SMILES randomization.
- Expected-energy sums equal brute-force evaluation on a toy distribution.
- Shifting probability mass toward closer distances lowers attractive energy monotonically for an attractive-only pair.
- **Oracle-mode equivalence:** with one-hot distributions from real coordinates, the expected-energy path reproduces a direct coordinate-based evaluation of every term.
- `test_labels.py`: every training label has type in {Kd, Ki}; no IC50/EC50 column, auxiliary head, or filter exists anywhere in data tables or configs.
- `test_leakage.py`: no benchmark pair or quarantined cluster in training; splits deliver the leakage guarantees they promise.

### 6.6 Oracle-pose mode (`energy.mode: oracle | predicted`)

The energy module accepts predicted distributions P_ij (default) or one-hot distributions from real coordinates (crystal, modeled, or folded). Same parameter heads, terms, and weights. Uses:

1. **Cost-of-pose-free ablation:** on structure-bearing test sets, run the identical model with true and predicted geometry. The gap is the price of dropping the pose and separates geometry error from parameter/physics error.
2. **Teacher model:** train an oracle-mode model on crystal and (optionally) folded poses with PIGNet2-style augmentation (docked/cross-docked/perturbed decoys, coordinate noise). Its pKd and per-term energies are distillation targets (Stage 2b). Distillation is unproven for this task; keep it optional and gated.
3. **PIGNet-comparable scoring:** the fair pose-based counterpart of the pose-free model.

### 6.7 Protein representation policy (PLM is an empirical question)

Evidence suggests frozen PLM features can encode family identity and hurt on strict splits. Every protein-input option is a config variant evaluated on strict tiers before any physics claim:

| `plm.mode` | Description |
|---|---|
| `none` | Residue type (one-hot/BLOSUM) plus learned embedding only |
| `esm2_650m_full` | Frozen ESM-2 650M, full 1280-d features |
| `esm2_bottleneck` | Frozen ESM-2 650M with a low-rank projection (e.g., 1280→64) plus dropout/noise |
| `esm2_pocket_window` | Embeddings restricted to the predicted pocket window |
| `esmc_300m` | Frozen ESM-C 300M |
| `esm2_3b` | Frozen ESM-2 3B (compute ablation, enabled by the HPC budget) |
| `plm_lora` | LoRA fine-tuning (Wave 3, gated; high overfitting risk) |

The default for the MVP is whichever of the first four wins on the protein-cluster validation split; do not assume the full-embedding version.

---

## 7. HPC compute plan

### 7.1 Assumptions and requirements

Assume SLURM job arrays, DDP across GPUs within a node, Apptainer containers, a parallel filesystem plus node-local scratch, and possibly no internet on compute nodes. Confirm each [USER INPUT]. Requirements:

- Pin the environment in a container; prefetch model weights and datasets on a login node.
- **Checkpoint and resume** every job (wall-time limits and preemption). Make every output idempotent and keyed by content hash so restarted job arrays skip finished work.
- Store caches as large shards (HDF5/Zarr/memmap); stage hot shards to node-local scratch to avoid small-file I/O. Estimate disk use before Phase 4 and confirm with the user.
- Use W&B offline mode (or CSV) and sync from the login node.
- Keep `reports/compute_ledger.csv` with GPU-hours per job/experiment, config hash, and split used.

### 7.2 Workload allocation (up to 12 concurrent GPUs)

| Workload | Parallelism |
|---|---|
| ESM embedding extraction; Uni-Mol ligand embeddings | 12 shard workers, sharded by sequence/ligand hash |
| Ligand physics (RDKit conformers, xTB) | CPU-node arrays; AIMNet2 on GPU |
| Offline teacher folding (Phase 2b) | Up to 12 workers; **measure throughput first, then set the budget**; resumable |
| Stage 0 QM pretraining | 1–4 GPUs (DDP), small |
| Stage 1–3 training runs | 1–4 GPUs per run (DDP); several runs concurrent |
| Ablation matrix, seeds (5–10 per configuration), Variant A/B | 1 GPU per run, up to 12 concurrent |
| Hyperparameter sweeps | Validation-only, pre-declared search space and trial cap |
| Baseline retraining (PIGNet2, GEMS, AEV-PLIG, pose-free DTA models, optional Boltz-2 affinity-head fine-tune) | Parallel single-GPU jobs |

### 7.3 What the compute enables beyond the earlier plan

Ligand physics Tiers B/C for all ligands; ESM-2 3B and PLM-variant ablations; Stage 0 pretraining as a routine ablation; Variant B alongside Variant A; large ensembles; offline Boltz-2 folding of Kd/Ki rows without crystal structures (geometry only); the oracle-pose teacher; and the strong baselines retrained on the same splits.

### 7.4 Guardrails against forking paths

More compute makes overfitting the validation set easy. Therefore: pre-declare the ablation matrix and search spaces in `configs/preregistration.yaml` before running them; cap the number of sweep trials; choose models on the protein-cluster validation split only; log every configuration trained (the ledger) and report the count in the final report; use multiple seeds and bootstrap CIs; score the sealed benchmark once per model version; and apply a multiplicity-aware reading of small gains (a ΔPearson of 0.02 found after hundreds of trials is weak evidence).

---

## 8. Phases and gates

Work proceeds in waves. Later waves start only when the earlier gate passes, except for clearly marked background jobs.

### Wave 1: foundations and honest baselines (Phases 0–6)

#### Phase 0: setup, verification, quarantine

- Create the repo (Section 10), container/environment lockfile, config system, logging, compute ledger.
- **Verify every "(verify)" claim** (Appendix A): dataset versions, sizes, activity types (Kd/Ki fractions), licenses, PIGNet2/Boltz-2 details, ESM-C and Uni-Mol licenses. Write `DATA_MANIFEST.md`.
- Ingest the benchmark and Boltz-2 scores; produce quarantine keys; confirm whether the benchmark contains IC50 rows (excluded from scoring). **[USER INPUT: benchmark files]**
- **Gate:** manifest complete; quarantine keys produced; metric code reproduces the user's Boltz-2 numbers on the benchmark.

#### Phase 1: affinity harvesting and harmonization

- One loader per source into the unified schema. Cross-source dedupe, ligand standardization/protonation, sequence resolution.
- Report counts (rows, unique proteins/ligands, per-type counts, per-source Venn overlap, drop reasons) and compare with the volumes in Section 4.4.
- **Gate:** dedupe verified on 20 hand-checked pairs; no benchmark leakage; `tests/test_labels.py` passes.

#### Phase 2: structure harvesting and geometry labels

- Extract sequence, pocket residues, site coordinates, and distogram targets from crystals (PDBbind/HiQBind/BioLiP2/MOAD/PLINDER), then modeled data (BindingNet, CrossDocked; SAIR only if `use_ic50_only_structures: true`), then MISATO frame histograms.
- Compute pose-quality flags (clashes, PoseBusters-style validity, confidence).
- **Gate:** distogram targets re-computed from raw coordinates on a random sample; pocket labels visualized for 10 complexes.

#### Phase 2b (background; use gated): offline teacher folding

Folding happens only at training-data preparation, never at inference.

- **Config:** `teacher_folding.enabled`, `teacher_folding.max_gpu_hours`; folder = Boltz-2 (or Chai-1).
- **What to fold:** Kd/Ki rows (never IC50) with no crystal structure, chosen for coverage (protein clusters with few or no crystals; diverse scaffolds per target). Exclude every benchmark/quarantined pair and cluster first.
- **Use:** geometry only (distogram/pocket targets at low weight; poses for the oracle-mode teacher). Do **not** use any Boltz-2 affinity output. Filter by folding confidence and PoseBusters-style validity; weight by confidence.
- **Independence caveat:** a student trained on Boltz-2 poses inherits some of its geometric biases. Report benchmark results with teacher folding on and off.
- **Gate:** throughput and cost measured on 50 complexes; subset size and expected GPU-hours reported to the user before the full run. Its data enters training only in Wave 3, unless the user opts in earlier.

#### Phase 3: ligand physics featurization

- Compute Tiers A–C for all ligands with content-hash caching and timing logs.
- **Gate:** features cached; failure rate <2% (or explained); timing report.

#### Phase 4: embedding caches

- Protein: frozen embeddings for every `plm.mode` in Section 6.7 (ESM-2 650M, ESM-C 300M, ESM-2 3B) in fp16. Ligand: Uni-Mol atom-level embeddings (config option for another atom-aligned CLM). Sharded HDF5/Zarr with a manifest.
- Estimate disk use first and confirm with the user if >100 GB. **[USER INPUT: storage budget]**
- **Gate:** deterministic re-extraction matches the cache on a random sample; loader throughput measured on the target filesystem.

#### Phase 5: splits

- Build and freeze the splits from Section 5.3 and write `splits/README.md` with counts and overlap statistics.
- **Gate:** automated leakage tests pass where the split promises them.

#### Phase 6: baselines (must exist before physics is judged)

1. **Target-mean**, **ligand-only** (ECFP + gradient boosting; CLM-embedding MLP), and **protein-only** (PLM-embedding MLP).
2. **Embedding fusion, no physics** (PLM/CLM inputs, cross-attention + MLP head, parameters matched to the physics model). This is the control for every physics claim. Also reproduce one or two published pose-free DTA models (ConPLex/PLAPT/CS-DTA-style) where code exists.
3. **Boltz-2** (user's scores; check its output definition and recalibrate as in Section 9).
4. **PIGNet2** under separate pose conditions (published weights, and retrained on this project's split where feasible; verify license and training overlap with the benchmark):
   - crystal poses (oracle upper bound; reported separately);
   - Boltz-2 predicted poses (the "with folding" pipeline) **[USER INPUT: are the predicted structures from the benchmark run saved?]**;
   - docked poses (GNINA/Vina) on the benchmark only.
5. **Structure-based references on the same splits:** GEMS/CleanSplit baseline and AEV-PLIG; optionally the Boltz-2 affinity head fine-tuned on this project's Kd/Ki training split.
6. **Oracle-mode version of this model** on structure-bearing test sets (after Phase 7).
- **Gate G1:** all baselines evaluated on all splits and the benchmark with bootstrap CIs; inference time per complex recorded (including the folding step where it applies); shortcut check (Section 5.3) reported; PLM-mode results on strict splits reported. The physics work may not be judged against numbers that do not exist yet.

### Wave 2: MVP physics model (Phases 7–9)

#### Phase 7: MVP model implementation

- Physics module with the MVP terms only (vdW, H-bond, hydrophobic, rotor penalty), parameter heads, expected-energy computation, priors, and the tests in Section 6.5.
- Pocket head, distogram head, pair module (Variant A), and Variant B coordinate-hypothesis head.
- **Gate:** all physics tests pass; oracle-mode equivalence verified; overfits a 200-complex subset; distogram calibration reasonable on validation crystals.

#### Phase 8: training curriculum

| Stage | Data | Losses | Notes |
|---|---|---|---|
| 0. Physics pretraining (ablation: on/off) | Splinter/DES370K/SPICE dimers | SAPT-component and interaction-energy regression (electrostatics ↔ Coulomb, exchange ↔ repulsion, dispersion ↔ attraction), S66x8 curve checks | Shared site-parameter network embeds both monomers with the CLM at atom level; protein sites use an adapter. Cheap on HPC. |
| 1. Geometry pretraining | crystal + modeled + MISATO structures | distogram CE (weighted), pocket BCE, triangle penalty, contact prior | No affinity loss; pocket and distogram must reach a good state first |
| 2. Affinity pretraining | all harmonized Kd/Ki data (structure-free rows included) plus replayed structural batches | heteroscedastic Huber on pKd, censored hinge (Kd/Ki only), parameter-prior penalty, **distogram CE anchor** | The gaming failure mode appears here; monitor calibration |
| 3. Crystal fine-tune | crystal-structure complexes with Kd/Ki | all of the above; lower LR, higher weight | High-quality stage |
| 2b. Teacher distillation (Wave 3, gated) | oracle-mode teacher (crystal + optionally folded poses, with augmentation) | student matches teacher pKd and per-term energies + Stage 2 losses incl. the anchor | Small weight; report with/without; never replaces the affinity loss |

- Frozen encoders in Wave 2. Ensembles of ≥5 seeds per configuration (cheap with cached embeddings) plus a heteroscedastic variant.
- **Gate:** each stage's validation metrics and calibration logged.

#### Phase 9: ablation matrix and Gate G2 (Tier 1 decision)

Run the pre-registered matrix in parallel (Section 7.2) on validation splits:

1. No physics (Phase 6 fusion control) vs physics-only vs physics + residual.
2. **PLM modes** (Section 6.7): none / full / bottleneck / pocket-window / ESM-C 300M / ESM-2 3B.
3. **Geometry:** Variant A vs Variant B; anchor on/off; triangle penalty on/off; contact prior on/off.
4. **vdW form:** LJ vs Morse.
5. Parameter priors on/off; Stage 0 on/off.
6. Ligand feature tiers A / A+B / A+B+C.
7. CLM: Uni-Mol vs an atom-aligned token/graph CLM.
8. Predicted-structure pretraining on/off; MISATO soft targets on/off; `label_types` [Kd, Ki] vs [Kd]; `use_ic50_only_structures` on/off (geometry only).
9. Ensemble size; top-K pocket size.
10. Oracle-pose vs predicted-distribution mode on structure-bearing test sets.

- **Gate G2 (Tier 1):** the best MVP configuration beats the matched no-physics control on the protein-cluster split by ΔPearson ≥ 0.02 (or an equivalent RMSE gain) with a bootstrap CI excluding zero, and beats ligand-only and target-mean.
- **If G2 passes:** proceed to Wave 3.
- **If G2 fails:** diagnose in this order: (1) PLM mode (a bottleneck or `none` often helps on strict splits), (2) distogram calibration/geometry variant, (3) individual MVP terms and the residual head, (4) label/data quality. If still failing, stop the physics expansion and deliver the best fusion model with an honest report of the physics ablations. Do not proceed to Wave 3 by default.

### Wave 3: full-scope expansion (only if G2 passed)

Each addition is a single-change ablation against the G2 winner, run in parallel, and kept only if it improves the protein-cluster validation split with a CI excluding zero.

- Terms: Coulomb, aromatic, halogen, desolvation, geometric entropy (one at a time, then combinations of survivors).
- Data: Phase 2b folded poses as geometry supervision; SAIR/other structures only under `use_ic50_only_structures: true` if the user opts in.
- Teacher distillation (Stage 2b) from the oracle-mode teacher with augmentation.
- PLM LoRA fine-tuning (`plm_lora`), only if a frozen PLM mode helped in G2.
- Ensemble scaling and uncertainty calibration.
- **Gate G3:** the expanded model beats the G2 winner on validation with CIs, and the diagnostics in Section 9 pass.

### Wave 4: final evaluation (Phase 10)

- Freeze the model version(s). Run the sealed benchmark **once per model version**, and log the version.
- Report all metrics and diagnostics (Section 9), the Boltz-2/PIGNet2 comparisons under each pose condition, the cost table, the oracle-vs-predicted gap, the configuration count from the ledger, and `reports/final.md` with honest limitations.

---

## 9. Evaluation

**Metrics** (all in pK units; bootstrap 95% CIs, ≥1000 resamples):

- Pearson, Spearman, RMSE, MAE (pooled).
- **Within-target** ranking: per-target Spearman and pairwise accuracy (targets with ≥10 ligands and ≥1.5 log-unit spread), reported as mean and median.
- Binder/non-binder AUROC at a chosen pK threshold **[USER INPUT: threshold, if relevant]**.
- Calibration: interval coverage; correlation of predicted uncertainty with absolute error.
- Stratify by seen/unseen protein cluster (30% and 50% identity tiers), seen/unseen scaffold, affinity range, measurement type (Kd vs Ki), protein family, sequence length.

**Comparison rule:** a physics claim requires beating the matched no-physics fusion control on the protein-cluster split by ΔPearson ≥ 0.02 (or equivalent RMSE gain) with a bootstrap CI excluding zero, holding up on the benchmark.

**Boltz-2 comparison:**

- Its affinity output is IC50-like (reported as log10 IC50 in µM), not a Kd-calibrated pK **(verify definition and units)**. Lead with rank-based metrics (Spearman, per-target Spearman, pairwise accuracy). Report Pearson/RMSE for Boltz-2 only after a linear recalibration fit on data disjoint from the test items.
- Paired bootstrap on the benchmark; per-target breakdown; residual correlation between models.
- **No stacked ensemble with Boltz-2 by default,** since its output carries IC50-trained information. Include a stacking analysis only if the user explicitly approves it **[USER INPUT]**, clearly labeled as outside the label policy.
- Report the low-overlap subset and the benchmark's overlap with Boltz-2's training sources.

**PIGNet2 comparison:** under each pose condition from Phase 6, side by side with this model and Boltz-2. Include a cost table (inference seconds per complex, counting the folding step for pose-based pipelines, and hardware). If accuracy does not beat pose-based pipelines, the value case rests on this cost gap.

**Success tiers** (report which are met; do not redefine after seeing results):

1. **Required (Tier 1):** beats the matched no-physics control on the protein-cluster split (rule above) and beats ligand-only and target-mean.
2. **Target (Tier 2):** within the bootstrap CI of PIGNet2-with-Boltz-2-poses on the benchmark for the primary rank metrics. Expected to be hard.
3. **Stretch (Tier 3):** within the CI of Boltz-2 on the benchmark, or better on the low-overlap subset. Very unlikely.

**Physics diagnostics:**

- Per-term energy distributions on crystal complexes; per-term correlation with labels.
- Distogram calibration on held-out crystal complexes (ECE per bin, contact recovery).
- Pocket recovery: AUROC/precision@K against crystal contacts.
- Hotspot check: do residues with the largest predicted contributions coincide with crystal contact residues and known hotspots for well-studied targets (kinases, proteases, nuclear receptors)?
- Point-mutation sensitivity against Platinum-type data (sign accuracy and correlation).
- Sanity perturbations: chirality inversion of activity-cliff pairs; charge-group changes (carboxylate → amide should reduce predicted salt-bridge energy when Coulomb is enabled); scrambled protein sequence (affinity should collapse toward the ligand-only baseline).
- PLA15-type sanity check on crystal poses (report correlation only; the model is coarse-grained).
- Property-bias check on property-matched decoy sets (DUD-E/LIT-PCBA-style) to detect ligand-property shortcuts, if feasible.

---

## 10. Repository layout

```
pfpl/
  configs/                    # hydra: data, features, model, train, eval, preregistration.yaml
  containers/                 # Apptainer definitions, lockfiles
  slurm/                      # sbatch templates, job-array launchers, resume logic
  sweeps/                     # pre-declared search spaces
  DATA_MANIFEST.md
  data/{raw,interim,processed}/     # gitignored
  splits/                     # frozen split files + README
  src/pfpl/
    data/        chembl.py bindingdb.py papyrus.py pdbbind.py moad.py biolip.py
                 pubchem.py gtopdb.py pdsp.py davis.py bindingnet.py sair.py
                 crossdocked.py misato.py plinder.py teacher_folding.py harmonize.py splits.py
    features/    ligand_phys.py embed_esm.py embed_ligand.py geom_labels.py cache.py
    physics/     sites.py terms.py params.py priors.py energy.py oracle.py qm_pretrain.py
    models/      trunk.py pocket.py pair.py distogram.py coord_head.py affinity.py
    train/       stage0_physics.py stage1_geometry.py stage2_affinity.py stage2b_distill.py stage3_finetune.py
    baselines/   pignet2_wrap.py tankbind_wrap.py aevplig_wrap.py gems_wrap.py dta_models.py boltz2_wrap.py
    eval/        metrics.py physics_diagnostics.py boltz_compare.py pignet_compare.py shortcut_checks.py
  tests/         test_physics.py test_harmonize.py test_leakage.py test_cache.py test_labels.py
  scripts/
  reports/       phase_0.md ... final.md compute_ledger.csv dropped_types.csv
```

**Stack:** Python 3.11, PyTorch (DDP), RDKit, transformers/ESM packages, unimol_tools, MMseqs2, xTB/AIMNet2, pandas/polars, HDF5/Zarr, Hydra, W&B (offline), SLURM, Apptainer.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Leakage and shortcut learning inflate all numbers | Leakage-controlled splits (LP-PDBBind, CleanSplit, PLINDER, cluster/scaffold/time); ligand-only/target-mean baselines in every table; automated leakage tests |
| Frozen PLM features hurt on strict splits | PLM-mode ablations (none/bottleneck/pocket-window) before physics claims; PLM default chosen on strict validation |
| Independent pair distributions are inconsistent and exploitable | Anchor, triangle penalty, contact prior, calibration monitoring; Variant B (coordinate hypotheses) as a structural fix |
| Distance-only H-bond/electrostatic terms lack directionality and polarization; Coulomb may add noise | Terms gated by ablation; optional residual head; honest failure-case reporting |
| Sequence-only input hides protonation, water, and conformational states | Learned parameters absorb averages; failure cases by family |
| Kd/Ki-only rule shrinks data (majority of BindingDB/SAIR/Boltz-2 data is IC50) | Lean on Kd/Ki-rich sources (BindingDB, ChEMBL, Papyrus++, PDBbind, PDSP, Davis); report per-source counts; heteroscedastic loss |
| Ki and Kd pooling adds assay noise | Stratified metrics; Kd-only ablation |
| Noise ceiling caps achievable accuracy | Interpret gains relative to the ~0.5 log inter-lab scatter; do not chase sub-noise RMSE |
| Student inherits Boltz-2 geometric biases via teacher folding | Off by default; on/off comparison; low-overlap subset reporting |
| Unfair PIGNet2/Boltz-2 comparisons (pose source, training overlap, IC50-like output) | Separate pose conditions; overlap checks; recalibration and rank metrics |
| Pose-free model cannot beat pose-based models | Oracle-mode ablation isolates the cause; fall back to the measured speed/interpretability value case |
| Forking paths from abundant compute | Preregistration, sweep caps, compute ledger, multiplicity-aware interpretation (Section 7.4) |
| Reliance on preprint/vendor numbers | Appendix A tracks evidence type; re-verify in Phase 0 |
| License constraints (ESM-C 600M non-commercial; PDBbind terms) | Default to ESM-2 650M/ESM-C 300M; confirm commercial context; do not redistribute PDBbind-derived data |
| Long sequences exceed the PLM window | Domain/pocket cropping; report by length bin |
| HPC operational failures (preemption, I/O, quotas) | Idempotent, resumable jobs; sharded caches; storage estimates |

---

## 12. Optional extensions (after Phase 10)

1. **Structure-augmented branch** using apo predicted structures (AlphaFold DB by UniProt, or ESMFold) for SASA, secondary structure, PROPKA pKa, and residue–residue contacts. **Conflicts with the strict "no folding" reading; run only with user approval,** and label results as such.
2. **Equivariant neural potential** conditioned on embeddings and predicted poses, replacing the pairwise terms.
3. **Pretrained pocket/interaction representations** (Uni-Mol pocket model, ProFSA/BindNet-type) to replace or initialize the cross-attention module.
4. **Ensemble/MD features** (BioEmu or MD-derived conformational diversity).
5. **FEP calibration** on a few targets to test tracking of physics-based ΔΔG within series.
6. **Active learning** using ensemble uncertainty.
7. **Not allowed:** distilling Boltz-2's affinity module into the student, or any IC50-derived signal (label policy).

---

## 13. Open questions for the user

1. Benchmark details: number of complexes/targets, affinity types (Kd/Ki only; IC50 rows are excluded from scoring), date range, file format, and how Boltz-2 scores were produced (head output, version, post-processing). Are the Boltz-2 predicted structures saved (needed for PIGNet2 with predicted poses)?
2. HPC specifics: GPU model/VRAM, scheduler, CPU nodes, storage/scratch quotas, wall-time limits, internet from compute nodes, containers, total GPU-hour allowance.
3. Commercial or academic context (affects ESM-C 600M and PDBbind usage; PDBbind registration status).
4. Is the "no folding" definition in Section 1 correct? In particular, is apo structure prediction (ESMFold/AlphaFold DB) excluded or allowed?
5. Ki pooled with Kd (default) or strict Kd-only? May structures of IC50-only complexes be used for geometry (`use_ic50_only_structures`, default off)?
6. Is a Boltz-2 stacking analysis wanted despite the IC50-trained signal (default: no)?
7. Budget for offline teacher folding (Phase 2b) after throughput is measured.
8. Does the ligand set include metals, covalent inhibitors, peptides, or cofactors? Exclude?
9. Are multi-chain targets (heterodimers, interface pockets) in scope, or single chain only?

---

## 14. Definition of done

- Frozen, version-controlled splits with passing leakage and label-policy tests.
- All Phase 6 baselines (including the matched no-physics control and ligand-only/target-mean), Boltz-2, and PIGNet2 under each pose condition, evaluated on the same splits and the sealed benchmark with confidence intervals, plus the inference-cost table.
- The pre-registered ablation matrix (PLM modes, geometry variants, each physics term, Stage 0, ligand feature tiers), with the compute ledger and configuration count.
- A trained ensemble of the best configuration with per-term decompositions and predicted pockets, and the physics diagnostics.
- The oracle-vs-predicted gap on structure-bearing test sets, and the success tier reached (Section 9) stated plainly.
- `reports/final.md` stating where the model wins, where it loses, which physics terms earned their place, and what would be needed to close the gap to structure-based methods.

---

## Appendix A: evidence and verification checklist (for Phase 0)

Re-verify each item and record the result in `DATA_MANIFEST.md`. Evidence types: **PR** peer-reviewed, **PP** preprint, **V** vendor/industry blog, **D** dataset/model card.

| Claim used in this plan | Evidence | Phase 0 action |
|---|---|---|
| Boltz-2 FEP+ avg Pearson ≈ 0.62 (0.66 on 4 targets) vs FEP+ ≈ 0.72; affinity output log10 IC50 (µM); binder-probability head; MIT license; v2.1.x is current | PP/V + repo | Read the paper and repo; confirm units, training-data cutoffs, current version |
| Independent Boltz-2 critiques (external mean ≈ 0.42; glue complexes; missing scaffold split) | V, review comments | Re-read sources; treat as indicative only |
| PIGNet2: Morse-type vdW, no explicit Coulomb, CASF-2016 numbers, docked/cross-docked augmentation, open repo | PR | Read paper and repo; confirm terms and numbers |
| GEMS/CleanSplit numbers; ligand-only RMSE 1.424; 49% CASF leakage | PR | Read paper; reproduce split filtering if feasible |
| HonestAffinity strict-tier numbers and ESM-2 effect | PP (recent, minor internal table inconsistencies) | Re-derive with own baselines; do not rely on decimals |
| PSICHIC 0.478 (Atom3D LBA 30%) and IPBind 0.732 | PP | Optional check |
| TankBind ≈ 0.6; E3Bind "knotted" maps | PR/PP | Optional check |
| BindingDB counts (~560K Ki, ~100K Kd, ~1.8M IC50); TDC 375,032 Ki / 52,284 Kd | PR/D | Recount from current downloads |
| PDBbind v2020 latest free; post-2020 restricted; HiQBind sizes and non-redistribution of PDBBind-opt | D | Read license terms |
| LP-PDBBind (CC-BY 4.0; split sizes) | PR/D | Verify |
| PLINDER (≈449k systems; CC-BY; structure resource) | D | Verify affinity content |
| BindingNet v1/v2 Kd/Ki fraction | Unverified | **Count Ki/Kd rows before use** |
| SAIR (5.24M structures, ~1.05M systems, IC50-type, ~3% anomalies, CC BY 4.0 per card) | PP/D | Verify license and label types |
| Papyrus/Papyrus++ (Ki/Kd kept; ChEMBL 34; threshold 0.5 log, reduced size) | PR/D | Verify current release |
| DES370K SAPT0 components; DES5M energy-only; Splinter (~1.7M configs, >9,000 dimers, SAPT0) | PR/D | Verify components before designing Stage 0 losses |
| ESM-C licenses (300M open; 600M non-commercial; 6B API); ESM-2 ~1022-token limit; Uni-Mol MIT | D | Verify licenses |
| MISATO (~20,000 complexes, MD + QM) | PR/D | Verify contents |
| Noise ceiling: 6–8% of curated matched pairs differ >1 unit; ~0.5 log inter-lab scatter | PR + estimate | Quantify on this project's own duplicate measurements |
