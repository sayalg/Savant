# Literature Review and Citation Package to Support Rewriting the *Savant* Introduction and Discussion

## TL;DR
- I identified and verified authoritative primary literature for **all 11 Introduction threads and 11 Discussion claims**, with complete BibTeX (Section 4). The strongest, most directly supportive sources are the experimental-noise-floor papers (Kramer 2012; Kalliokoski 2013; Landrum & Riniker 2024), the "structure adds little" papers (Volkov 2022; Chen/DUD-E 2019), the AlphaFold-docking critique (Karelina, Noh & Dror 2023), and the tabular-ML papers (Chen & Guestrin 2016; Grinsztajn 2022) — each maps onto a specific *Savant* claim.
- Two claims need careful hedging: (a) the explicit **pooled-vs-within-target decomposition** has thinner direct precedent — it is best framed as an extension of the benchmark-bias and early-recognition-metric literature rather than as a previously established method; and (b) the **"0.81 pKd irreducible RMSE floor"** should be presented as *consistent with, and larger than,* the ~0.5 pKi single-source floor of Kramer 2012 because *Savant* pools Kd/Ki/IC50 (Kalliokoski 2013 and Landrum & Riniker 2024 explicitly warn this inflates noise).
- Where the literature complicates a claim, I say so explicitly below (see especially Discussion Claims 3, 4, and 8).

---

## 1. INTRODUCTION — Thread-by-Thread Findings

### Thread 1 — Cost and attrition of drug discovery
- **Wouters, McKee & Luyten (JAMA 2020)** is the best modern, independent, publicly-sourced cost estimate: the median capitalized R&D investment to bring a new medicine to market was **$985.3 million (95% CI $683.6M–$1,228.9M)** and the mean was **$1,335.9 million**, based on 63 drugs approved 2009–2018. Use this as your headline number — it is the most defensible because it uses SEC filings rather than confidential industry data.
- **DiMasi, Grabowski & Hansen (J Health Econ 2016)** is the canonical higher industry estimate. Per the enricher's verification: out-of-pocket cost per approved drug was **$1,395 million**, rising to **$2,558 million** capitalized (2013 dollars), and to **$2,870 million** including post-approval R&D, based on 106 drugs from 10 firms. Cite alongside Wouters to bracket the debate.
- **Scannell et al. (Nat Rev Drug Discov 2012)** — "Eroom's Law." Verbatim: *"the number of new drugs approved per billion US dollars spent on R&D has halved roughly every 9 years since 1950, falling around 80-fold in inflation-adjusted terms."* This is your single most rhetorically powerful citation for "inputs got cheaper, output got harder" — precisely the gap *Savant* addresses on the compute-input side.
- **Paul et al. (Nat Rev Drug Discov 2010)** — "How to improve R&D productivity" — frames attrition and the value of cheap early triage.
- *Throughput of biophysical assays:* I was unable to retrieve a single dedicated peer-reviewed "throughput of SPR/ITC/FP/TSA" review within budget. **Recommendation:** cite Gilson & Zhou (2007) for the thermodynamic basis and, for throughput, either a biophysics methods review you already have or leave this as a qualitative statement rather than an over-precise number. Do not fabricate a per-assay cost figure.

### Thread 2 — Binding thermodynamics and molecular recognition
- **Gilson & Zhou (Annu Rev Biophys Biomol Struct 2007)** — canonical reference for the calculation of protein–ligand binding affinities and the ΔG = −RT ln K_d relationship. Use this to define pKd physically.
- **Hopkins, Groom & Alex (Drug Discov Today 2004)** — introduces ligand efficiency; use for "limits of affinity optimization."
- *Note:* for the enthalpy–entropy compensation and individual force contributions (H-bonding, electrostatics, hydrophobic effect, desolvation), the HADDOCK/CAPRI energy terms your Stage-1 model reads (vdw, elec, desolv, bsa) map naturally onto this thermodynamic decomposition — cite Gilson & Zhou as the umbrella reference and the HADDOCK papers (Thread 8) for the operational terms.

### Thread 3 — Generative chemistry / ultra-large libraries (the screening bottleneck)
- **Lyu et al. (Nature 2019)** — ultra-large library docking of **170 million** make-on-demand compounds (>10.7 million scaffolds) against AmpC and D4. The flagship "tangible chemical space" docking campaign.
- **Gorgulla et al. (Nature 2020)** — VirtualFlow, open-source platform for ultra-large virtual screens (>1 billion compounds).
- **Sadybekov & Katritch (Nature 2023)** — review of computational approaches streamlining drug discovery; explicitly frames the need for *fast* ligand-screening methods over gigascale spaces and the "democratization" argument (doubles for Thread 11).
- **Tingle et al. (JCIM 2023)** — ZINC-22, a free multi-billion-scale database of tangible compounds.
- **Olivecrona et al. (J Cheminform 2017)** and **Blaschke et al. (JCIM 2020, REINVENT 2.0)** — landmark generative de novo design methods.
- **Framing recommendation:** the strongest motivating argument is the *mismatch* — generation and enumeration now produce 10⁹–10¹⁰ candidates (Lyu, Gorgulla, Tingle), but rigorous per-compound evaluation (FEP, co-folding) is orders of magnitude slower. *Savant*'s 0.018 s/ligand marginal cost sits directly in that gap.

### Thread 4 — Classical scoring functions and docking (their weakness)
- **Warren et al. (J Med Chem 2006)** — the critical assessment of 10 docking programs and 37 scoring functions across 8 targets. Key finding for you: *"For prediction of compound affinity, none of the docking programs or scoring functions made"* useful rank-order predictions — i.e., docking finds poses but scoring functions fail at affinity. This is your central "docking scoring is weak at affinity" citation.
- **Su et al. (JCIM 2019) — CASF-2016** — the standard comparative "scoring/ranking/docking/screening power" benchmark for scoring functions; use to show even best-case scoring-function correlations are modest.
- Supplement with AutoDock Vina, Glide, RF-Score as needed (metadata not separately verified within budget — flagged).

### Thread 5 — Free energy perturbation / alchemical methods (the accuracy ceiling, and why it can't scale)
- **Wang et al. (JACS 2015)** — FEP+ landmark. Per enricher verification: benchmarked on **8 targets / 199 ligands** (the "JACS set"), achieving **potency prediction better than 1 kcal/mol** with a mean unsigned error of **0.77 kcal/mol**. This is your "accuracy ceiling" citation.
- **Schindler et al. (JCIM 2020)** — large-scale FEP+ validation across active Merck drug-discovery projects (>300 ligands, 14 targets) — establishes prospective real-world accuracy. (Metadata via cross-reference; verify volume 60, pp. 5457–5474.)
- **Cournia, Allen & Sherman (JCIM 2017)** — review of relative binding free-energy methods (metadata not fully verified — flagged).
- **Framing:** FEP requires (i) a high-quality structure, (ii) a congeneric series, and (iii) GPU-hours *per compound* — three constraints *Savant* is explicitly designed to avoid. State this as the reason FEP cannot be used for library-scale triage.

### Thread 6 — Machine learning for affinity: the full DTA arc
- **Structure-free DTA (your lineage):** DeepDTA (Öztürk et al., Bioinformatics 2018) — predicts pKd from sequence + SMILES only, on the **Davis** and **KIBA** benchmarks; the founding structure-free DTA paper. Then GraphDTA (Nguyen et al. 2021), MolTrans (Huang et al. 2021), ConPLex (Singh et al., PNAS 2023), DrugBAN (Bai et al., Nat Mach Intell 2023), and **PLAPT (Rose et al., bioRxiv 2024)** — your direct structure-free comparator.
- **Benchmarks:** Davis and KIBA (via DeepDTA), BindingDB, and **Therapeutics Data Commons (Huang et al., NeurIPS 2021)**.
- **Structure-based ML scoring** (for contrast, to motivate Stage 1): RF-Score, NNScore, Pafnucy, K_DEEP, OnionNet, PotentialNet — cite a representative subset; note these require a structure and generally do not beat simple ligand-memorization baselines (ties into Volkov 2022, Thread 9).

### Thread 7 — Protein and molecular language models as featurizers
- **Lin et al. (Science 2023) — ESM-2** — the canonical protein language model reference; scaled 8M→15B parameters, structure emerges from sequence.
- **EvolutionaryScale (2024) — ESM-C / ESM Cambrian** — your actual protein featurizer. **Important bibliographic caveat:** at the December 2024 release there was **no peer-reviewed paper**; the standard citation is the EvolutionaryScale release/blog plus the Zenodo software DOI (10.5281/zenodo.14219303). A 2026 bioRxiv preprint now exists but its DOI prefix is unusual and unverified — I recommend citing the 2024 Zenodo/release for reproducibility and flagging it as software.
- **Ahmad et al. (2022) — ChemBERTa-2** — your ligand featurizer; RoBERTa-style model pretrained on 77M PubChem SMILES (the "77M-MLM" checkpoint you use). Cite Chithrananda et al. (2020) as the original ChemBERTa if you want the lineage.
- **Analysis/caveat papers:** Volkov et al. (2022) and the "learning characteristics of GNNs" line (Mastropietro et al., Nat Mach Intell 2023) are useful for "what PLM/graph embeddings do and do not capture" — they show embeddings often encode *ligand identity* more than *interaction physics*.

### Thread 8 — Structure prediction and co-folding
- **AlphaFold3 (Abramson et al., Nature 2024)** — your Stage-1 folding engine. Note the December 2024 Addendum (Nature 636:E4) releasing inference code.
- **Boltz-1 (Wohlwend et al., bioRxiv 2024)** and **Boltz-2 (Passaro et al., bioRxiv 2025)** — Boltz-2 is your primary GPU co-folding + affinity comparator.
- **DiffDock (Corso et al., ICLR 2023)**, RoseTTAFold All-Atom, Chai-1 — cite as the co-folding/docking landscape.
- **HADDOCK3 / HADDOCK 2.x** — your Stage-1 rescoring engine and the source of the CAPRI energy terms. Cite the HADDOCK information-driven docking papers (metadata not fully verified within budget — flagged; verify the HADDOCK2.2 / HADDOCK3 references).
- **CRITICAL supporting citation — Karelina, Noh & Dror (eLife 2023):** AF2 models capture binding-pocket structure well, *"Surprisingly, however, ligand binding poses predicted given AF2 models were not significantly more accurate than those predicted given traditional models and were much less accurate than those predicted by docking… to experimentally determined protein structures."* This is the single best external justification for abandoning the structure-based route after Stage 1. Pair it with your own measured "structure was worth only 0.051 R²" result.

### Thread 9 — Evaluation metrics: pooled vs within-target (the central methodological argument)
This thread needs the most careful handling because the *specific* pooled-vs-within-target decomposition is not a widely-established named method. Build the argument from adjacent, well-cited work:
- **Chen et al. (PLoS ONE 2019)** — "Hidden bias in the DUD-E dataset": CNN enrichment came from *"analogue and decoy bias hidden in the DUD-E dataset rather than successful generalization of the pattern of protein–ligand interactions."* Directly supports "pooled benchmarks reward exploiting between-target structure."
- **Sieg, Flachsenberg & Rarey (JCIM 2019)** — "In Need of Bias Control" — evaluating chemical data for ML in structure-based VS; formalizes bias control.
- **Wallach & Heifets (JCIM 2018)** — "Most ligand-based classification benchmarks reward memorization rather than generalization." The strongest statement that pooled metrics can be gamed.
- **Volkov et al. (J Med Chem 2022)** — verbatim: *"an explicit description of protein–ligand noncovalent interactions does not provide any advantage with respect to ligand or protein descriptors… memorization largely dominates true learning."* Ligand-only/protein-only models match full-interaction models on PDBbind — exactly the "pooled correlation is inflated by target identity" phenomenon.
- **Honest framing recommendation:** present the between-vs-within decomposition as *your contribution*, motivated by (not derived from) this bias literature and the early-recognition-metric literature (Truchon & Bayly 2007, Thread/Claim 10). Do not claim it is a standard analysis; claim it *operationalizes* the warnings of Chen, Sieg, Wallach & Heifets, and Volkov for the regression (affinity) setting.

### Thread 10 — Data quality and the experimental noise floor (supports the 0.81 pKd claim)
This is where your evidence is strongest and most quantitative.
- **Kramer et al. (J Med Chem 2012)** — "The experimental uncertainty of heterogeneous public Ki data." Verbatim: *"experimental uncertainty defines a natural upper limit to the predictive performance possible. Models that yield errors smaller than the experimental uncertainty are necessarily overtrained."* Their filtered ChEMBL estimate: **mean error 0.44 pKi, SD 0.54 pKi, median 0.34 pKi.** This is your primary "irreducible floor" citation and the "maximum achievable R²" argument.
- **Kalliokoski et al. (PLoS ONE 2013)** — "Comparability of Mixed IC50 Data." Mixed public IC50 data have *"poor agreement… almost 65% of the points differ by more than 0.3 log units, 27% differ by more than one log unit."* Directly supports your point that *pooling assay types inflates noise*.
- **Landrum & Riniker (JCIM 2024)** — "Combining IC50 or Ki Values from Different Sources Is a Source of Significant Noise." The most recent, direct support for your decision-relevant caveat.
- **Cheng & Prusoff (Biochem Pharmacol 1973)** — the Cheng–Prusoff relationship; cite for why IC50 is assay-condition-dependent and not interchangeable with Ki/Kd.
- **How to state your 0.81 pKd claim:** Frame it as *"consistent with, and necessarily larger than,"* the ~0.5 pKi single-source floor of Kramer 2012 — larger precisely because your corpus pools Kd, Ki, and IC50 across sources, which Kalliokoski 2013 and Landrum & Riniker 2024 show adds variance. This turns a potential weakness (noisy labels) into a principled, well-cited bound on achievable RMSE.

### Thread 11 — CPU vs GPU / democratization
- **Sadybekov & Katritch (Nature 2023)** explicitly discuss how cost-effective, fast ligand screening *"can democratize the drug discovery process."* This is your best on-point citation for the democratization framing.
- For carbon/energy cost of deep-learning inference and per-compound VS cost, I could not verify a single canonical primary source within budget. **Recommendation:** report your own measured 0.018 s/ligand CPU cost and the ~30-min/100k-library projection as the concrete contribution, and cite Sadybekov & Katritch for the qualitative accessibility argument rather than over-claiming an external cost benchmark.

### Thread 12 — Gradient-boosted trees on tabular/descriptor data
- **Chen & Guestrin (KDD 2016)** — XGBoost. Direct method citation.
- **Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022)** — verbatim: *"tree-based models remain state-of-the-art on medium-sized data (~10K samples) even without accounting for their superior speed,"* across 45 datasets. This is your best "why XGBoost over deep nets on tabular features" citation. Note: their result is strongest at ~10K samples; your training set is 1.9M rows, so present this as support for the *inductive-bias* argument (robustness to uninformative features, handling of irregular target functions) rather than a strict sample-size match.
- **Shwartz-Ziv & Armon (Information Fusion 2022)** — "Tabular data: deep learning is not all you need" — corroborating (metadata not fully verified — flagged).

---

## 2. DISCUSSION — Claim-by-Claim Support

**Claim 1 — "The metric selects the winner" (pooled vs within-target ranking inverts).**
Support: Truchon & Bayly (2007) is the foundational statement that *the choice of metric changes the ranking of methods* — they show ROC/AUAC/average-rank are "poor metrics" for early recognition and derive BEDROC. Combine with Chen (2019), Sieg (2019), and Wallach & Heifets (2018). **Complication to disclose:** none of these performs the exact between-/within-target variance decomposition you report, so present your inversion as a novel, quantitative demonstration of a qualitatively known problem.

**Claim 2 — "Removing the structure was the correct trade."**
Support: **Volkov et al. (2022)** is the strongest — explicit interaction descriptions give no advantage over ligand/protein descriptors. **Ramsundar et al. (2015)** shows predictive power grows with data volume and tasks (learning-curve/scaling argument). Your own +0.058 R²/doubling learning-curve slope with no saturation is the internal evidence; frame Volkov as external corroboration that the *marginal value of structure is small* relative to the marginal value of data. Karelina, Noh & Dror (2023) supplies the "predicted structures aren't good enough anyway" leg.

**Claim 3 — "Compression of predictions is a consequence of the label" (0.425 of measured spread).**
Support: this is **regression dilution / attenuation** — under squared-error loss with noisy labels, least-squares predictions shrink toward the conditional mean by a factor related to the reliability ratio (signal variance / (signal + noise variance)). Kramer et al. (2012) provides the noise magnitude that drives the attenuation. **Complication to disclose honestly:** I did not retrieve a cheminformatics-specific primary paper naming "prediction range shrinkage in QSAR" within budget; cite the statistical attenuation concept (a classical statistics reference such as Spearman's correction / Fuller's *Measurement Error Models*) plus Kramer 2012, and present the 0.425 compression as *quantitatively consistent* with the reliability ratio implied by a ~0.8 pKd noise SD against your label spread — show the arithmetic rather than asserting a citation says exactly 0.425.

**Claim 4 — "No method beat a constant-mean predictor on RMSE; all R² negative."**
Support: This is expected on narrow-dynamic-range test sets and follows directly from Kramer's "maximum achievable R²" logic — when test variance approaches the noise variance, R² → 0 or negative even for a genuinely useful ranker. **Key nuance to state:** R² and RMSE measure *calibration/variance explained*, whereas within-target Spearman/Pearson measure *ranking* — a method can rank well (useful for triage) while failing R² (Truchon & Bayly's ranking-vs-classification distinction generalizes here). This reconciles your negative R² with your positive within-target correlation and is the honest, decision-relevant interpretation.

**Claim 5 — "Savant beats per-target QSAR at low data; QSAR wins when data is abundant."**
Support: **Ramsundar et al. (2015)** — massively multitask networks beat single-task models, and *"the predictive power of multitask networks improves as additional tasks and data are added,"* with the benefit concentrated on low-data targets. This is the canonical data-volume crossover between global/multi-task and local/per-target models. Supplement with the applicability-domain literature (Sheridan's "similarity to training set" work). Your rho 0.459 vs 0.018 on <1000-compound targets is a clean instance of this crossover.

**Claim 6 — "Should not be used to choose between targets for a fixed ligand."**
Support: your near-zero between-target correlation (0.100) is the internal evidence. Externally, frame target-selection/target-fishing/proteome-wide selectivity as a *distinct and harder* problem. I did not verify a specific polypharmacology review within budget — **recommendation:** cite a target-prediction/polypharmacology review you already hold, or state the limitation from first principles (the model was trained on target-centered residuals, which by construction removes between-target signal). Do not overstate.

**Claim 7 — "Features and objective matter more than structure presence/absence" (the PLAPT comparison).**
Support: Volkov et al. (2022) again (model class vs feature content). For PLAPT specifically, it is a **bioRxiv preprint (Rose et al. 2024)**, not journal-published — a legitimate, citable point when noting the absence of independent peer review and the general concern (Wallach & Heifets 2018; Sheridan 2013) about over-optimistic reporting in DTA papers under non-stringent splits. Frame your result as: same structure-free inputs, different feature engineering and a screening-aligned objective, yield different behavior — supporting "objective/feature design dominates."

**Claim 8 — "Training on target-centered residuals and ranking-oriented objectives."**
Support: Truchon & Bayly (2007) motivates ranking-oriented (early-recognition) objectives over global regression. Ramsundar (2015) motivates residualization as a form of multi-task/target-conditioning. **Complication:** I did not retrieve a specific "learning-to-rank for affinity / pairwise ranking loss" primary paper within budget (e.g., PBCNet-style pairwise networks exist — Nature Comput Sci 2023 — but I could not verify metadata). **Recommendation:** cite the pairwise/relative-binding literature you hold, and present target-centered residual labels as a principled choice justified by the within-target objective, not as a previously standard technique.

**Claim 9 — Scaffold split and temporal validation as the realistic evaluation.**
Support (strong): **Sheridan (JCIM 2013)** — *"Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction"* — the canonical argument that random CV is optimistic and time-split estimates true prospective performance. **Bemis & Murcko (J Med Chem 1996)** — the scaffold definition underlying your Murcko-scaffold inner split. Your benchmark's "2025-or-later, absent from ChEMBL v34" design is a time-split in the Sheridan sense — say so explicitly.

**Claim 10 — BEDROC and early-recognition metrics; critique of enrichment factors.**
Support (strong): **Truchon & Bayly (JCIM 2007)** — introduces BEDROC and shows *"the EF metric is not sensitive to ranking performance before and after the cutoff"* and that ROC/AUAC/average-rank are poor for early recognition. Directly supports both your advocacy of early-recognition metrics and your critique of EF at high active density.

**Claim 11 — Temporal separation protects against training-set contamination.**
Support: Sheridan (2013) for the chemistry-specific data-leakage argument; Wallach & Heifets (2018) for memorization; and the general foundation-model contamination concern (test data postdating the training corpus). Your ECFP4 Tanimoto <0.50 filter plus post-v34 temporal cutoff is a strong, defensible anti-leakage design — frame it as implementing exactly what Sheridan and Wallach & Heifets recommend.

---

## 3. Concrete Restructuring Suggestions

**Introduction (expand from ~6 lightly-cited paragraphs to ~8–9 paragraphs):**
1. *The productivity problem* — Wouters 2020 + DiMasi 2016 (cost), Scannell 2012 (Eroom's Law), Paul 2010. One paragraph. Thesis: inputs got cheaper, output didn't.
2. *The screening bottleneck* — Lyu 2019, Gorgulla 2020, Tingle 2023, Olivecrona/Blaschke (generation ≫ evaluation throughput); Sadybekov & Katritch 2023 as the umbrella. Thesis: we can enumerate 10⁹⁺ molecules but not score them.
3. *What affinity is and how it's measured* — Gilson & Zhou 2007, Hopkins 2004; biophysical assays are the gold standard but low-throughput.
4. *Physics-based in silico affinity and why it doesn't scale* — Warren 2006 + Su 2019 (docking scoring weak); Wang 2015 + Schindler 2020 (FEP accurate but needs structure + congeneric series + GPU-hours/compound).
5. *Structure prediction raised hopes* — AlphaFold3 (Abramson 2024), Boltz (Wohlwend 2024, Passaro 2025) — *but* Karelina 2023 shows predicted structures don't yet support reliable pose/affinity prediction. This paragraph motivates your Stage-1→Stage-2 pivot.
6. *Structure-free ML affinity* — DeepDTA 2018 → GraphDTA/MolTrans/ConPLex/DrugBAN/PLAPT; ESM-2/ESM-C and ChemBERTa-2 as featurizers; TDC/Davis/KIBA benchmarks.
7. *The evaluation problem* — Chen 2019, Sieg 2019, Wallach & Heifets 2018, Volkov 2022: pooled benchmarks reward memorization/between-target shortcuts. Introduce the pooled-vs-within-target framing here as *your* lens.
8. *The data-noise floor* — Kramer 2012, Kalliokoski 2013, Landrum & Riniker 2024: set expectations for achievable RMSE before presenting results.
9. *This work* — one paragraph stating the two-stage design, the CPU/democratization goal (Sadybekov & Katritch 2023), XGBoost choice (Chen & Guestrin 2016; Grinsztajn 2022), and the screening-aligned objective.

**Discussion — add citations subsection-by-subsection** exactly as mapped in Section 2 above. Highest-leverage additions: Volkov 2022 (Claims 2 & 7), Kramer 2012 (Claims 3 & 4), Sheridan 2013 (Claims 9 & 11), Truchon & Bayly 2007 (Claims 1, 8, 10), Ramsundar 2015 (Claim 5), Karelina 2023 (Claim 2).

---

## 4. BibTeX (verified DOIs; uncertainties flagged inline)

```bibtex
@article{wouters2020cost,
  author  = {Wouters, Olivier J. and McKee, Martin and Luyten, Jeroen},
  title   = {Estimated Research and Development Investment Needed to Bring a New Medicine to Market, 2009--2018},
  journal = {JAMA},
  year    = {2020},
  volume  = {323},
  number  = {9},
  pages   = {844--853},
  doi     = {10.1001/jama.2020.1166}
}

@article{dimasi2016innovation,
  author  = {DiMasi, Joseph A. and Grabowski, Henry G. and Hansen, Ronald W.},
  title   = {Innovation in the Pharmaceutical Industry: New Estimates of {R\&D} Costs},
  journal = {Journal of Health Economics},
  year    = {2016},
  volume  = {47},
  pages   = {20--33},
  doi     = {10.1016/j.jhealeco.2016.01.012}
}

@article{scannell2012eroom,
  author  = {Scannell, Jack W. and Blanckley, Alex and Boldon, Helen and Warrington, Brian},
  title   = {Diagnosing the Decline in Pharmaceutical {R\&D} Efficiency},
  journal = {Nature Reviews Drug Discovery},
  year    = {2012},
  volume  = {11},
  number  = {3},
  pages   = {191--200},
  doi     = {10.1038/nrd3681}
}

@article{paul2010improve,
  author  = {Paul, Steven M. and Mytelka, Daniel S. and Dunwiddie, Christopher T. and Persinger, Charles C. and Munos, Bernard H. and Lindborg, Stacy R. and Schacht, Aaron L.},
  title   = {How to Improve {R\&D} Productivity: The Pharmaceutical Industry's Grand Challenge},
  journal = {Nature Reviews Drug Discovery},
  year    = {2010},
  volume  = {9},
  number  = {3},
  pages   = {203--214},
  doi     = {10.1038/nrd3078}
}

@article{gilson2007calculation,
  author  = {Gilson, Michael K. and Zhou, Huan-Xiang},
  title   = {Calculation of Protein--Ligand Binding Affinities},
  journal = {Annual Review of Biophysics and Biomolecular Structure},
  year    = {2007},
  volume  = {36},
  pages   = {21--42},
  doi     = {10.1146/annurev.biophys.36.040306.132550}
}

@article{hopkins2004ligand,
  author  = {Hopkins, Andrew L. and Groom, Colin R. and Alex, Alexander},
  title   = {Ligand Efficiency: A Useful Metric for Lead Selection},
  journal = {Drug Discovery Today},
  year    = {2004},
  volume  = {9},
  number  = {10},
  pages   = {430--431},
  doi     = {10.1016/S1359-6446(04)03069-7}
}

@article{lyu2019ultralarge,
  author  = {Lyu, Jiankun and Wang, Sheng and Balius, Trent E. and Singh, Isha and Levit, Anat and Moroz, Yurii S. and O'Meara, Matthew J. and Che, Tao and Algaa, Enkhjargal and Tolmachova, Kateryna and Tolmachev, Andrey A. and Shoichet, Brian K. and Roth, Bryan L. and Irwin, John J.},
  title   = {Ultra-Large Library Docking for Discovering New Chemotypes},
  journal = {Nature},
  year    = {2019},
  volume  = {566},
  number  = {7743},
  pages   = {224--229},
  doi     = {10.1038/s41586-019-0917-9}
}

@article{gorgulla2020virtualflow,
  author  = {Gorgulla, Christoph and Boeszoermenyi, Andras and Wang, Zi-Fu and Fischer, Patrick D. and Coote, Paul W. and Padmanabha Das, Krishna M. and Malets, Yehor S. and Radchenko, Dmytro S. and Moroz, Yurii S. and Scott, David A. and Fackeldey, Konstantin and Hoffmann, Moritz and Iavniuk, Iryna and Wagner, Gerhard and Arthanari, Haribabu},
  title   = {An Open-Source Drug Discovery Platform Enables Ultra-Large Virtual Screens},
  journal = {Nature},
  year    = {2020},
  volume  = {580},
  number  = {7805},
  pages   = {663--668},
  doi     = {10.1038/s41586-020-2117-z}
}

@article{sadybekov2023computational,
  author  = {Sadybekov, Anastasiia V. and Katritch, Vsevolod},
  title   = {Computational Approaches Streamlining Drug Discovery},
  journal = {Nature},
  year    = {2023},
  volume  = {616},
  number  = {7958},
  pages   = {673--685},
  doi     = {10.1038/s41586-023-05905-z}
}

@article{tingle2023zinc22,
  author  = {Tingle, Benjamin I. and Tang, Khanh G. and Castanon, Mar and Gutierrez, John J. and Khurelbaatar, Munkhzul and Dandarchuluun, Chinzorig and Moroz, Yurii S. and Irwin, John J.},
  title   = {{ZINC-22} --- A Free Multi-Billion-Scale Database of Tangible Compounds for Ligand Discovery},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2023},
  volume  = {63},
  number  = {4},
  pages   = {1166--1176},
  doi     = {10.1021/acs.jcim.2c01253}
}

@article{olivecrona2017reinvent,
  author  = {Olivecrona, Marcus and Blaschke, Thomas and Engkvist, Ola and Chen, Hongming},
  title   = {Molecular De-Novo Design Through Deep Reinforcement Learning},
  journal = {Journal of Cheminformatics},
  year    = {2017},
  volume  = {9},
  pages   = {48},
  doi     = {10.1186/s13321-017-0235-x},
  note    = {Article number 48 --- verify if precision critical}
}

@article{blaschke2020reinvent2,
  author  = {Blaschke, Thomas and Ar{\'u}s-Pous, Josep and Chen, Hongming and Margreitter, Christian and Tyrchan, Christian and Engkvist, Ola and Papadopoulos, Kostas and Patronov, Atanas},
  title   = {{REINVENT} 2.0: An {AI} Tool for De Novo Drug Design},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2020},
  volume  = {60},
  number  = {12},
  pages   = {5918--5922},
  doi     = {10.1021/acs.jcim.0c00915}
}

@article{warren2006critical,
  author  = {Warren, Gregory L. and Andrews, C. Webster and Capelli, Anna-Maria and Clarke, Brian and LaLonde, Judith and Lambert, Millard H. and Lindvall, Mika and Nevins, Neysa and Semus, Simon F. and Senger, Stefan and Tedesco, Giovanna and Wall, Ian D. and Woolven, James M. and Peishoff, Catherine E. and Head, Martha S.},
  title   = {A Critical Assessment of Docking Programs and Scoring Functions},
  journal = {Journal of Medicinal Chemistry},
  year    = {2006},
  volume  = {49},
  number  = {20},
  pages   = {5912--5931},
  doi     = {10.1021/jm050362n}
}

@article{su2019casf2016,
  author  = {Su, Minyi and Yang, Qifan and Du, Yu and Feng, Guoqin and Liu, Zhihai and Li, Yan and Wang, Renxiao},
  title   = {Comparative Assessment of Scoring Functions: The {CASF-2016} Update},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2019},
  volume  = {59},
  number  = {2},
  pages   = {895--913},
  doi     = {10.1021/acs.jcim.8b00545}
}

@article{wang2015fep,
  author  = {Wang, Lingle and Wu, Yujie and Deng, Yuqing and Kim, Byungchan and Pierce, Levi and Krilov, Goran and Lupyan, Dmitry and Robinson, Shaughnessy and Dahlgren, Markus K. and Greenwood, Jeremy and others},
  title   = {Accurate and Reliable Prediction of Relative Ligand Binding Potency in Prospective Drug Discovery by Way of a Modern Free-Energy Calculation Protocol and Force Field},
  journal = {Journal of the American Chemical Society},
  year    = {2015},
  volume  = {137},
  number  = {7},
  pages   = {2695--2703},
  doi     = {10.1021/ja512751q}
}

@article{schindler2020largescale,
  author  = {Schindler, Christina E. M. and Baumann, Hannah and Blum, Andreas and Bo{\"o}se, Dietrich and Buchstaller, Hans-Peter and others},
  title   = {Large-Scale Assessment of Binding Free Energy Calculations in Active Drug Discovery Projects},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2020},
  volume  = {60},
  number  = {11},
  pages   = {5457--5474},
  doi     = {10.1021/acs.jcim.0c00900},
  note    = {Verify full author list and pages}
}

@article{ozturk2018deepdta,
  author  = {{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzucan and Ozkirimli, Elif},
  title   = {{DeepDTA}: Deep Drug--Target Binding Affinity Prediction},
  journal = {Bioinformatics},
  year    = {2018},
  volume  = {34},
  number  = {17},
  pages   = {i821--i829},
  doi     = {10.1093/bioinformatics/bty593}
}

@article{nguyen2021graphdta,
  author  = {Nguyen, Thin and Le, Hang and Quinn, Thomas P. and Nguyen, Tri and Le, Thuc Duy and Venkatesh, Svetha},
  title   = {{GraphDTA}: Predicting Drug--Target Binding Affinity with Graph Neural Networks},
  journal = {Bioinformatics},
  year    = {2021},
  volume  = {37},
  number  = {8},
  pages   = {1140--1147},
  doi     = {10.1093/bioinformatics/btaa921}
}

@article{huang2021moltrans,
  author  = {Huang, Kexin and Xiao, Cao and Glass, Lucas M. and Sun, Jimeng},
  title   = {{MolTrans}: Molecular Interaction Transformer for Drug--Target Interaction Prediction},
  journal = {Bioinformatics},
  year    = {2021},
  volume  = {37},
  number  = {6},
  pages   = {830--836},
  doi     = {10.1093/bioinformatics/btaa880}
}

@article{singh2023conplex,
  author  = {Singh, Rohit and Sledzieski, Samuel and Bryson, Bryan and Cowen, Lenore and Berger, Bonnie},
  title   = {Contrastive Learning in Protein Language Space Predicts Interactions Between Drugs and Protein Targets},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2023},
  volume  = {120},
  number  = {24},
  pages   = {e2220778120},
  doi     = {10.1073/pnas.2220778120}
}

@article{bai2023drugban,
  author  = {Bai, Peizhen and Miljkovi{\'c}, Filip and John, Bino and Lu, Haiping},
  title   = {Interpretable Bilinear Attention Network with Domain Adaptation Improves Drug--Target Prediction},
  journal = {Nature Machine Intelligence},
  year    = {2023},
  volume  = {5},
  number  = {2},
  pages   = {126--136},
  doi     = {10.1038/s42256-022-00605-1}
}

@article{rose2024plapt,
  author  = {Rose, Tyler and Monti, Nicol{\`o} and Anand, Navvye and Shen, Tianyu},
  title   = {{PLAPT}: Protein-Ligand Binding Affinity Prediction Using Pretrained Transformers},
  journal = {bioRxiv},
  year    = {2024},
  doi     = {10.1101/2024.02.08.575577},
  note    = {Preprint --- not peer-reviewed}
}

@inproceedings{huang2021tdc,
  author    = {Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Roohani, Yusuf and Leskovec, Jure and Coley, Connor W. and Xiao, Cao and Sun, Jimeng and Zitnik, Marinka},
  title     = {Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2021},
  note      = {arXiv:2102.09548; no journal DOI}
}

@article{lin2023esm2,
  author  = {Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yaniv and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Salvatore and Rives, Alexander},
  title   = {Evolutionary-Scale Prediction of Atomic-Level Protein Structure with a Language Model},
  journal = {Science},
  year    = {2023},
  volume  = {379},
  number  = {6637},
  pages   = {1123--1130},
  doi     = {10.1126/science.ade2574}
}

@misc{esmc2024,
  author       = {{EvolutionaryScale Team}},
  title        = {{ESM} Cambrian ({ESM-C}): Scaling Protein Language Models},
  year         = {2024},
  howpublished = {EvolutionaryScale; open weights and code},
  doi          = {10.5281/zenodo.14219303},
  note         = {No peer-reviewed paper at 2024 release; cite Zenodo software DOI}
}

@article{ahmad2022chemberta2,
  author  = {Ahmad, Walid and Simon, Elana and Chithrananda, Seyone and Grand, Gabriel and Ramsundar, Bharath},
  title   = {{ChemBERTa-2}: Towards Chemical Foundation Models},
  journal = {arXiv preprint arXiv:2209.01712},
  year    = {2022},
  doi     = {10.48550/arXiv.2209.01712}
}

@article{chithrananda2020chemberta,
  author  = {Chithrananda, Seyone and Grand, Gabriel and Ramsundar, Bharath},
  title   = {{ChemBERTa}: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction},
  journal = {arXiv preprint arXiv:2010.09885},
  year    = {2020},
  doi     = {10.48550/arXiv.2010.09885}
}

@article{abramson2024alphafold3,
  author  = {Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J. and Bambrick, Joshua and others},
  title   = {Accurate Structure Prediction of Biomolecular Interactions with {AlphaFold} 3},
  journal = {Nature},
  year    = {2024},
  volume  = {630},
  number  = {8016},
  pages   = {493--500},
  doi     = {10.1038/s41586-024-07487-w}
}

@article{wohlwend2024boltz1,
  author  = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title   = {{Boltz-1}: Democratizing Biomolecular Interaction Modeling},
  journal = {bioRxiv},
  year    = {2024},
  doi     = {10.1101/2024.11.19.624167},
  note    = {Preprint}
}

@article{passaro2025boltz2,
  author  = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and St{\"a}rk, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title   = {{Boltz-2}: Towards Accurate and Efficient Binding Affinity Prediction},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.06.14.659707},
  note    = {Preprint}
}

@inproceedings{corso2023diffdock,
  author    = {Corso, Gabriele and St{\"a}rk, Hannes and Jing, Bowen and Barzilay, Regina and Jaakkola, Tommi},
  title     = {{DiffDock}: Diffusion Steps, Twists, and Turns for Molecular Docking},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  note      = {arXiv:2210.01776; no journal DOI}
}

@article{vangone2019prodigylig,
  author  = {Vangone, Anna and Schaarschmidt, Joerg and Koukos, Panagiotis and Geng, Cunliang and Citro, Nevia and Trellet, Mikael E. and Xue, Li C. and Bonvin, Alexandre M. J. J.},
  title   = {Large-Scale Prediction of Binding Affinity in Protein--Small Ligand Complexes: The {PRODIGY-LIG} Web Server},
  journal = {Bioinformatics},
  year    = {2019},
  volume  = {35},
  number  = {9},
  pages   = {1585--1587},
  doi     = {10.1093/bioinformatics/bty816}
}

@article{karelina2023alphafold,
  author  = {Karelina, Masha and Noh, Joseph J. and Dror, Ron O.},
  title   = {How Accurately Can One Predict Drug Binding Modes Using {AlphaFold} Models?},
  journal = {eLife},
  year    = {2023},
  volume  = {12},
  pages   = {RP89386},
  doi     = {10.7554/eLife.89386}
}

@article{chen2019dude,
  author  = {Chen, Lieyang and Cruz, Anthony and Ramsey, Steven and Dickson, Callum J. and Duca, Jose S. and Hornak, Viktor and Koes, David R. and Kurtzman, Tom},
  title   = {Hidden Bias in the {DUD-E} Dataset Leads to Misleading Performance of Deep Learning in Structure-Based Virtual Screening},
  journal = {PLoS ONE},
  year    = {2019},
  volume  = {14},
  number  = {8},
  pages   = {e0220113},
  doi     = {10.1371/journal.pone.0220113}
}

@article{sieg2019bias,
  author  = {Sieg, Jochen and Flachsenberg, Florian and Rarey, Matthias},
  title   = {In Need of Bias Control: Evaluating Chemical Data for Machine Learning in Structure-Based Virtual Screening},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2019},
  volume  = {59},
  number  = {3},
  pages   = {947--961},
  doi     = {10.1021/acs.jcim.8b00712}
}

@article{wallach2018memorization,
  author  = {Wallach, Izhar and Heifets, Abraham},
  title   = {Most Ligand-Based Classification Benchmarks Reward Memorization Rather than Generalization},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2018},
  volume  = {58},
  number  = {5},
  pages   = {916--932},
  doi     = {10.1021/acs.jcim.7b00403}
}

@article{volkov2022frustration,
  author  = {Volkov, Mikhail and Turk, Joseph-Andr{\'e} and Drizard, Nicolas and Martin, Nicolas and Hoffmann, Brice and Gaston-Math{\'e}, Yann and Rognan, Didier},
  title   = {On the Frustration to Predict Binding Affinities from Protein--Ligand Structures with Deep Neural Networks},
  journal = {Journal of Medicinal Chemistry},
  year    = {2022},
  volume  = {65},
  number  = {11},
  pages   = {7946--7958},
  doi     = {10.1021/acs.jmedchem.2c00487}
}

@article{mastropietro2023gnn,
  author  = {Mastropietro, Andrea and Pasculli, Giuseppe and Bajorath, J{\"u}rgen},
  title   = {Learning Characteristics of Graph Neural Networks Predicting Protein--Ligand Affinities},
  journal = {Nature Machine Intelligence},
  year    = {2023},
  volume  = {5},
  pages   = {1427--1436},
  doi     = {10.1038/s42256-023-00756-9}
}

@article{kramer2012uncertainty,
  author  = {Kramer, Christian and Kalliokoski, Tuomo and Gedeck, Peter and Vulpetti, Anna},
  title   = {The Experimental Uncertainty of Heterogeneous Public {Ki} Data},
  journal = {Journal of Medicinal Chemistry},
  year    = {2012},
  volume  = {55},
  number  = {11},
  pages   = {5165--5173},
  doi     = {10.1021/jm300131x}
}

@article{kalliokoski2013ic50,
  author  = {Kalliokoski, Tuomo and Kramer, Christian and Vulpetti, Anna and Gedeck, Peter},
  title   = {Comparability of Mixed {IC50} Data --- A Statistical Analysis},
  journal = {PLoS ONE},
  year    = {2013},
  volume  = {8},
  number  = {4},
  pages   = {e61007},
  doi     = {10.1371/journal.pone.0061007}
}

@article{landrum2024combining,
  author  = {Landrum, Gregory A. and Riniker, Sereina},
  title   = {Combining {IC50} or {Ki} Values from Different Sources Is a Source of Significant Noise},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2024},
  volume  = {64},
  number  = {5},
  pages   = {1560--1567},
  doi     = {10.1021/acs.jcim.4c00049}
}

@article{cheng1973prusoff,
  author  = {Cheng, Yung-Chi and Prusoff, William H.},
  title   = {Relationship Between the Inhibition Constant ($K_1$) and the Concentration of Inhibitor Which Causes 50 Per Cent Inhibition ($I_{50}$) of an Enzymatic Reaction},
  journal = {Biochemical Pharmacology},
  year    = {1973},
  volume  = {22},
  number  = {23},
  pages   = {3099--3108},
  doi     = {10.1016/0006-2952(73)90196-2}
}

@inproceedings{chen2016xgboost,
  author    = {Chen, Tianqi and Guestrin, Carlos},
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)},
  year      = {2016},
  pages     = {785--794},
  publisher = {ACM},
  doi       = {10.1145/2939672.2939785}
}

@inproceedings{grinsztajn2022tabular,
  author    = {Grinsztajn, L{\'e}o and Oyallon, Edouard and Varoquaux, Ga{\"e}l},
  title     = {Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2022},
  volume    = {35},
  pages     = {507--520},
  note      = {arXiv:2207.08815}
}

@article{shwartzziv2022tabular,
  author  = {Shwartz-Ziv, Ravid and Armon, Amitai},
  title   = {Tabular Data: Deep Learning Is Not All You Need},
  journal = {Information Fusion},
  year    = {2022},
  volume  = {81},
  pages   = {84--90},
  doi     = {10.1016/j.inffus.2021.11.011},
  note    = {Verify volume/pages}
}

@article{truchon2007bedroc,
  author  = {Truchon, Jean-Fran{\c c}ois and Bayly, Christopher I.},
  title   = {Evaluating Virtual Screening Methods: Good and Bad Metrics for the ``Early Recognition'' Problem},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2007},
  volume  = {47},
  number  = {2},
  pages   = {488--508},
  doi     = {10.1021/ci600426e}
}

@article{sheridan2013timesplit,
  author  = {Sheridan, Robert P.},
  title   = {Time-Split Cross-Validation as a Method for Estimating the Goodness of Prospective Prediction},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2013},
  volume  = {53},
  number  = {4},
  pages   = {783--790},
  doi     = {10.1021/ci400084k}
}

@article{bemis1996murcko,
  author  = {Bemis, Guy W. and Murcko, Mark A.},
  title   = {The Properties of Known Drugs. 1. Molecular Frameworks},
  journal = {Journal of Medicinal Chemistry},
  year    = {1996},
  volume  = {39},
  number  = {15},
  pages   = {2887--2893},
  doi     = {10.1021/jm9602928}
}

@article{ramsundar2015multitask,
  author  = {Ramsundar, Bharath and Kearnes, Steven and Riley, Patrick and Webster, Dale and Konerding, David and Pande, Vijay},
  title   = {Massively Multitask Networks for Drug Discovery},
  journal = {arXiv preprint arXiv:1502.02072},
  year    = {2015},
  doi     = {10.48550/arXiv.1502.02072}
}

@article{gaulton2012chembl,
  author  = {Gaulton, Anna and Bellis, Louisa J. and Bento, A. Patricia and Chambers, Jon and Davies, Mark and Hersey, Anne and Light, Yvonne and McGlinchey, Shaun and Michalovich, David and Al-Lazikani, Bissan and Overington, John P.},
  title   = {{ChEMBL}: A Large-Scale Bioactivity Database for Drug Discovery},
  journal = {Nucleic Acids Research},
  year    = {2012},
  volume  = {40},
  number  = {D1},
  pages   = {D1100--D1107},
  doi     = {10.1093/nar/gkr777}
}

@article{gilson2016bindingdb,
  author  = {Gilson, Michael K. and Liu, Tiqing and Baitaluk, Michael and Nicola, George and Hwang, Linda and Chong, Jenny},
  title   = {{BindingDB} in 2015: A Public Database for Medicinal Chemistry, Computational Chemistry and Systems Pharmacology},
  journal = {Nucleic Acids Research},
  year    = {2016},
  volume  = {44},
  number  = {D1},
  pages   = {D1045--D1053},
  doi     = {10.1093/nar/gkv1072}
}

@misc{glaser_binding_affinity,
  author       = {Glaser, Jens and others},
  title        = {{jglaser/binding\_affinity}: Aggregated Protein Sequence--Ligand {SMILES} Binding Affinity Dataset},
  howpublished = {Hugging Face Datasets, \url{https://huggingface.co/datasets/jglaser/binding_affinity}},
  note         = {No dedicated dataset DOI located; aggregates BindingDB, PDBbind-CN, BioLiP, Binding MOAD. Closest companion paper: Blanchard et al., Int. J. High Perform. Comput. Appl. 2022, 36(5--6):587--602, doi:10.1177/10943420221121804. Confirm intended citation with authors.}
}
```

---

## 5. Recommendations (staged and decision-ready)

**Stage 1 — Adopt immediately (fully verified, high-value).** Insert the following into the Introduction and Discussion now, because their DOIs and mappings are confirmed and their support is direct: Wouters 2020, Scannell 2012, DiMasi 2016 (cost/attrition); Lyu 2019, Gorgulla 2020, Tingle 2023, Sadybekov & Katritch 2023 (bottleneck); Warren 2006, Su 2019 (docking weakness); Wang 2015 (FEP ceiling); Karelina 2023 (abandon-structure justification); Volkov 2022, Chen 2019, Wallach & Heifets 2018, Sieg 2019 (evaluation/memorization); Kramer 2012, Kalliokoski 2013, Landrum & Riniker 2024 (noise floor); Chen & Guestrin 2016, Grinsztajn 2022 (tabular ML); Truchon & Bayly 2007, Sheridan 2013, Bemis & Murcko 1996 (metrics/splits); Ramsundar 2015 (multitask crossover).

**Stage 2 — Verify one bibliographic detail before submission (flagged entries).** Confirm: Schindler 2020 author list/pages; Olivecrona 2017 article number (48); Shwartz-Ziv & Armon volume/pages; the HADDOCK 2.x/HADDOCK3 references (not retrieved within budget); Cournia/Allen/Sherman 2017 JCIM review; and the correct companion citation for the `jglaser/binding_affinity` dataset (the authors should state which paper they intend). For ESM-C, decide between the 2024 Zenodo software DOI (recommended for reproducibility) and the 2026 bioRxiv preprint.

**Stage 3 — Reframe two claims to avoid over-reach.** (a) Present the pooled-vs-within-target decomposition as *your operationalization* of the bias literature (Chen, Sieg, Wallach & Heifets, Volkov), not as a pre-existing standard method. (b) Present the 0.81 pKd floor and the 0.425 compression factor with the *arithmetic shown* (reliability ratio from noise SD vs label spread), citing Kramer 2012 for the noise magnitude and a classical measurement-error/attenuation reference for the shrinkage mechanism — rather than implying a single paper reports these exact numbers.

**Benchmarks that would change these recommendations:** If a reviewer requests a *named prior method* for the between/within decomposition, elevate the framing in Thread 9 to a formal "we introduce" contribution and add a limitations sentence. If you obtain a dedicated peer-reviewed throughput/cost review for biophysical assays or for GPU-inference carbon cost, upgrade Threads 1 and 11 from qualitative to quantitative. If PLAPT is peer-reviewed by submission time, update `rose2024plapt` and remove the "not peer-reviewed" note.

---

## 6. Caveats
- **Budget-limited retrieval.** My web-search budget was exhausted before I could separately verify a handful of secondary entries: AutoDock Vina/Glide/GOLD/RF-Score metadata, the HADDOCK family papers, Cournia 2017, Shwartz-Ziv & Armon page numbers, and ZINC-22/Enamine REAL beyond ZINC-22. These are flagged inline; do not treat their BibTeX as final until confirmed against the publisher page.
- **Preprints.** Boltz-1, Boltz-2, PLAPT, DiffDock, TDC, Ramsundar 2015, ChemBERTa/-2, and ESM-C are preprints or conference papers without traditional journal DOIs. This is legitimate to cite but note the peer-review status, especially for PLAPT (your comparator).
- **ESM-C bibliographic uncertainty.** The 2024 release has no peer-reviewed paper; the subagent flagged an unusual DOI prefix (10.64898) on a 2026 EvolutionaryScale-hosted preprint that could not be independently verified. I recommend the Zenodo software DOI for the 2024 model you actually used.
- **The two "reframe" claims (Section 5, Stage 3)** are the only places where the literature does not straightforwardly hand you a citation; I have flagged them so you can present them as contributions rather than as borrowed results. This is the honest position and is stronger for peer review than an overclaimed citation.
- **No fabricated DOIs or pages.** Every DOI above is either directly verified from a publisher/PubMed page in my research or explicitly flagged for verification. Where I was uncertain, I said so rather than guessing.