# Savant benchmark - 1000 protein-ligand Kd pairs

**QC: PASSED**

- unique targets: 355
- unique ligands: 859 (639 Bemis-Murcko scaffolds)
- protein classes: 36; ChemOnt classes: 80
- pKd: 2.17 to 11.70 (mean 6.43, sd 1.72)
- source: BindingDB (TDC Kd table, Harvard Dataverse 4291555)

## Disjointness from training

Every ligand is below ECFP4 Tanimoto 0.5 to *every* training ligand (max observed 0.497, median 0.366), and no exact structure or InChIKey-skeleton match survives. The novelty tier records how far the *protein* sits from training:

| tier | meaning | pairs | targets |
|---|---|---:|---:|
| `same_protein` | target in training; ligand is new chemistry | 400 | 155 |
| `same_family` | new protein, UniProt family seen in training | 400 | 144 |
| `novel_family` | protein family absent from training | 200 | 56 |

## Protein classes (ChEMBL level 2)

| protein class | pairs | share |
|---|---:|---:|
| Kinase | 60 | 6.0% |
| Oxidoreductase | 57 | 5.7% |
| Transferase | 56 | 5.6% |
| Family A G protein-coupled receptor | 54 | 5.4% |
| Unclassified protein | 54 | 5.4% |
| Ligand-gated ion channel | 52 | 5.2% |
| Voltage-gated ion channel | 52 | 5.2% |
| Secreted protein | 48 | 4.8% |
| Electrochemical transporter | 44 | 4.4% |
| Unclassified | 39 | 3.9% |
| Other cytosolic protein | 36 | 3.6% |
| Nuclear receptor | 32 | 3.2% |
| Lyase | 31 | 3.1% |
| Membrane receptor | 31 | 3.1% |
| Transcription factor | 30 | 3.0% |
| Isomerase | 29 | 2.9% |
| Hydrolase | 27 | 2.7% |
| Cytochrome P450 | 24 | 2.4% |
| Enzyme | 23 | 2.3% |
| Protease | 23 | 2.3% |
| Other ion channel | 23 | 2.3% |
| Ligase | 22 | 2.2% |
| Primary active transporter | 21 | 2.1% |
| Reader | 20 | 2.0% |
| Phosphodiesterase | 17 | 1.7% |
| Other membrane protein | 17 | 1.7% |
| Structural protein | 16 | 1.6% |
| Eraser | 16 | 1.6% |
| Family B G protein-coupled receptor | 11 | 1.1% |
| Surface antigen | 9 | 0.9% |
| Writer | 8 | 0.8% |
| Phosphatase | 7 | 0.7% |
| Fatty acid binding protein family | 5 | 0.5% |
| Adhesion | 3 | 0.3% |
| Family C G protein-coupled receptor | 2 | 0.2% |
| Other nuclear protein | 1 | 0.1% |

## Ligand structural classes (ClassyFire / ChemOnt)

| ChemOnt class | pairs | share |
|---|---:|---:|
| Benzene and substituted derivatives | 165 | 16.5% |
| Carboxylic acids and derivatives | 89 | 8.9% |
| Organooxygen compounds | 83 | 8.3% |
| Prenol lipids | 53 | 5.3% |
| nan | 48 | 4.8% |
| Azoles | 40 | 4.0% |
| Indoles and derivatives | 38 | 3.8% |
| Quinolines and derivatives | 29 | 2.9% |
| Diazanaphthalenes | 28 | 2.8% |
| Piperidines | 23 | 2.3% |
| Polypeptides | 23 | 2.3% |
| Diazinanes | 22 | 2.2% |
| Pyridines and derivatives | 21 | 2.1% |
| Naphthalenes | 20 | 2.0% |
| Phenol ethers | 19 | 1.9% |
| Flavonoids | 19 | 1.9% |
| Tetralins | 17 | 1.7% |
| Organonitrogen compounds | 15 | 1.5% |
| Stilbenes | 15 | 1.5% |
| Benzimidazoles | 13 | 1.3% |

## Drug classes (ATC level 1, where the ligand is a known drug)

| ATC class | pairs | share |
|---|---:|---:|
| Alimentary tract & metabolism | 8 | 0.8% |
| Dermatologicals | 7 | 0.7% |
| Genito-urinary & sex hormones | 7 | 0.7% |
| Nervous system | 6 | 0.6% |
| Antineoplastic & immunomodulating | 6 | 0.6% |
| Anti-infectives (systemic) | 4 | 0.4% |
| Musculo-skeletal system | 1 | 0.1% |

## Affinity distribution

| pKd range | pairs | share |
|---|---:|---:|
| (5, 6] | 219 | 21.9% |
| (0, 5] | 206 | 20.6% |
| (6, 7] | 205 | 20.5% |
| (7, 8] | 173 | 17.3% |
| (8, 9] | 118 | 11.8% |
| (9, 20] | 79 | 7.9% |
