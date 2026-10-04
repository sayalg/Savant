# Sprint results

Generated from `/scratch/sguirale/Savant/sprint_oct8/runs`. Primary metric is **median per-target Spearman**;
`rho_ligand_only` is the memorisation floor and `rho_oracle` is the
ligand-blind ceiling. A model that does not beat `rho_ligand_only` has
learned nothing about proteins.

## 1. Zero-shot (headline)

v1 reference: per-target rho **+0.28**, RMSE 1.473 vs 1.495 for a constant,
offset slope **0.089**, interaction share **6%**, EF@10 **1.03x**.

| run                                     | split     | objective   |   rho |   rho_lo |   rho_hi | rho_ligand_only   | rho_oracle   |   n_targets | pos   |   rmse |   rmse_const |   std_ratio |   slope | pct_interaction   |   ef10 | gate   |
|:----------------------------------------|:----------|:------------|------:|---------:|---------:|:------------------|:-------------|------------:|:------|-------:|-------------:|------------:|--------:|:------------------|-------:|:-------|
| xgboost_v1_equivalent_squared_clustered | clustered | squared     | 0.158 |    0.129 |    0.179 | +0.325            | -            |         613 | 71%   |  1.416 |        1.54  |       0.092 |   0.083 | 17.8%             |  1.2   | False  |
| catboost_full_ranking_clustered         | clustered | ranking     | 0.153 |    0.128 |    0.177 | +0.325            | -            |         613 | 73%   |  1.859 |        1.54  |       0.267 |   0.283 | 14.2%             |  1.4   | False  |
| gnn_cross_attention_ranking_viral       | viral     | ranking     | 0.139 |    0.049 |    0.219 | -                 | -            |          42 | 74%   |  2.854 |        1.951 |       0.466 |  -0.416 | 4.3%              |  1     | False  |
| gnn_cross_attention_ranking_clustered   | clustered | ranking     | 0.12  |    0.106 |    0.137 | -                 | -            |         613 | 72%   |  1.847 |        1.54  |       0.329 |   0.14  | 9.9%              |  1.2   | False  |
| xgboost_full_squared_clustered          | clustered | squared     | 0.109 |    0.082 |    0.135 | +0.325            | -            |         613 | 68%   |  1.433 |        1.54  |       0.063 |   0.061 | 25.5%             |  1     | False  |
| xgboost_pooled_plus_squared_clustered   | clustered | squared     | 0.108 |    0.076 |    0.129 | +0.325            | -            |         613 | 68%   |  1.457 |        1.54  |       0.047 |   0.038 | 18.4%             |  1.005 | False  |
| gnn_bilinear_ranking_clustered          | clustered | ranking     | 0.107 |    0.085 |    0.129 | -                 | -            |         613 | 68%   |  1.821 |        1.54  |       0.914 |   0.188 | 6.0%              |  0.965 | False  |
| gnn_concat_ranking_clustered            | clustered | ranking     | 0.07  |    0.057 |    0.094 | -                 | -            |         613 | 64%   |  2.652 |        1.54  |       0.133 |  -0.013 | 15.6%             |  0.977 | False  |


## 2. Per-target (lead optimisation - NOT comparable to table 1)

Trained on some ligands of a target and tested on others, scaffold-split
within target. Better numbers, different question.

| run                         |   rho |   n_targets |   n_pairs |   rmse |   rmse_const |
|:----------------------------|------:|------------:|----------:|-------:|-------------:|
| tabpfn_pertarget_cross_only | 0.602 |          13 |      2123 |  0.887 |        1.629 |


## 3. Case studies

| run                               | case               | role         |   n |    rho |   ef10 |
|:----------------------------------|:-------------------|:-------------|----:|-------:|-------:|
| gnn_cross_attention_ranking_viral | influenza_A_NA_PR8 | contaminated | 111 |  0.22  |   2.77 |
| gnn_cross_attention_ranking_viral | HCMV_UL80_protease | held_out     |  30 | -0.169 |   1    |
| gnn_cross_attention_ranking_viral | SARS2_Mpro_nsp5    | external     |  38 | -0.166 |   1.9  |
| tabpfn_pertarget_cross_only       | influenza_A_NA_PR8 | contaminated |  50 |  0.351 |   3    |

## Provenance

```json
{
 "protein_set_meta.json": {
  "source": "work/merged_kd_ki.csv",
  "rows_in": 633838,
  "rows_kept": 633838,
  "n_proteins": 6176,
  "n_ligands": 300109,
  "max_len": 1022,
  "n_truncated": 615,
  "order_checksum": "86d5dce339fa3b59",
  "has_accession": false,
  "label_col_source": "neg_log10_affinity_M"
 },
 "esm_meta.json": {
  "model": "facebook/esm2_t33_650M_UR50D",
  "dim": 1280,
  "pools": [
   "mean",
   "max",
   "min",
   "std",
   "p10",
   "p90"
  ],
  "topk": 64,
  "min_sep": 6,
  "n_proteins": 6176,
  "pooled_width": 7680
 },
 "pocket_meta.json": {
  "source": "esm_centrality",
  "pocket_k": 32,
  "names": [
   "pk_hydropathy_mean",
   "pk_hydropathy_min",
   "pk_hydropathy_max",
   "pk_volume_mean",
   "pk_volume_sum",
   "pk_net_charge",
   "pk_abs_charge",
   "pk_hbd_sum",
   "pk_hba_sum",
   "pk_arom_frac",
   "pk_polar_frac",
   "pk_gly_frac",
   "pk_pro_frac",
   "pk_cys_frac",
   "pk_his_frac",
   "pk_acidic_frac",
   "pk_basic_frac",
   "pk_aliphatic_frac",
   "pk_span",
   "pk_n_segments",
   "pk_score_mean",
   "pk_len"
  ]
 },
 "viral_panel.json": {
  "n_viral_proteins": 67,
  "n_in_panel": 18,
  "n_panel_pairs": 6488,
  "min_ligands": 25,
  "case_studies": {
   "influenza_A_NA_PR8": {
    "accession": "P03468",
    "role": "contaminated",
    "note": "protein_id 11771 was in the v1 fit fold; 55/81 ChEMBL Ki ligands were training pairs"
   },
   "HCMV_UL80_protease": {
    "accession": "P16753",
    "role": "held_out",
    "note": "v1 protein_id 14017, test fold, zero fit-fold homologs at >=0.50 12-mer containment"
   },
   "SARS2_Mpro_nsp5": {
    "accession": "P0DTD1",
    "role": "external",
    "region": [
     3264,
     3569
    ],
    "note": "only 3 corpus pairs for the exact 306-aa domain; the full 7096-aa polyprotein entry truncates past nsp2"
   }
  }
 }
}
```
