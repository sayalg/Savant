# v3 benchmark - 500 pairs, 20 targets, 373 ligands

source: savant_benchmark_v3_scored.csv

## A. pooled
```
                    n  pearson  spearman   rmse    mae   bias     r2  pred_std_ratio
tool                                                                                
Boltz-2           496    0.412     0.393  1.388  1.078 -0.633 -0.193           0.750
PRODIGY-LIG       496    0.333     0.340  1.537  1.258 -0.073 -0.461           1.087
Savant esm8       500    0.330     0.276  1.316  1.035 -0.515 -0.069           0.453
Savant-v2 esm8    500    0.290     0.218  1.335  1.033 -0.509 -0.099           0.443
Savant-v2 esm650  500    0.229     0.153  1.363  1.050 -0.524 -0.146           0.400
Savant esm650     500    0.227     0.147  1.353  1.052 -0.480 -0.129           0.425
Tanimoto kNN      458    0.102     0.066  1.679  1.339 -0.544 -0.688           0.823
QSAR per-target   458    0.089     0.050  1.733  1.327 -0.891 -0.797           0.663
PLAPT             500    0.021    -0.032  1.827  1.445 -0.821 -1.060           0.824
Constant mean     500      NaN       NaN  1.273  1.031  0.000  0.000           0.000
```

constant-mean RMSE 1.273 - a tool whose RMSE is above this is not beating 'predict the average'

## Between vs within decomposition
```
                    n  between_r  within_r  pooled_r  target_mean_spread
tool                                                                    
Boltz-2           496      0.351     0.386     0.412               0.818
PRODIGY-LIG       496      0.295     0.237     0.333               1.137
Savant esm8       500      0.257     0.433     0.330               0.536
Savant-v2 esm8    500      0.190     0.405     0.290               0.532
Savant-v2 esm650  500      0.119     0.429     0.229               0.466
Savant esm650     500      0.100     0.436     0.227               0.472
Tanimoto kNN      458      0.135     0.066     0.102               1.006
QSAR per-target   458      0.124     0.155     0.089               0.829
PLAPT             500     -0.225     0.208     0.021               0.668
Constant mean     500     -0.028       NaN       NaN               0.000
```

## B. one protein, many ligands
```
                  n_targets  mean_spearman  median_spearman  n_positive
Savant-v2 esm650         20          0.311            0.288          17
Savant-v2 esm8           20          0.278            0.211          16
Savant esm650            20          0.278            0.278          16
PRODIGY-LIG              20          0.233            0.196          15
Boltz-2                  20          0.228            0.242          15
Savant esm8              20          0.212            0.169          13
QSAR per-target          18          0.127            0.226          12
PLAPT                    20          0.067            0.111          12
Tanimoto kNN             18          0.031            0.002           9
```

paired vs Boltz-2
```
                  n_groups  mean_diff  ci_low  ci_high  P_better  wins  wilcoxon_p  sign_test_p
tool                                                                                           
Savant-v2 esm650        20      0.082  -0.035    0.204     0.921    11       0.177        0.824
Savant-v2 esm8          20      0.050  -0.085    0.198     0.762     9       0.475        0.824
Savant esm650           20      0.050  -0.059    0.159     0.824     9       0.648        0.824
PRODIGY-LIG             20      0.005  -0.144    0.176     0.532     9       0.869        0.824
Savant esm8             20     -0.016  -0.167    0.142     0.439     7       0.927        0.263
PLAPT                   20     -0.161  -0.305   -0.019     0.010     8       0.070        0.503
QSAR per-target         20        NaN     NaN      NaN     0.012     8       0.523        0.815
Tanimoto kNN            20        NaN     NaN      NaN     0.000     5       0.074        0.096
```

## C. one ligand, many proteins (selectivity)
```
                  n_raw  raw_accuracy  raw_ci_halfwidth  raw_beats_chance  n_corrected  corrected_accuracy  corrected_ci_halfwidth  corrected_beats_chance
Savant-v2 esm650    115         0.730             0.081              True           82               0.634                   0.104                    True
Savant esm650       115         0.730             0.081              True           82               0.598                   0.106                   False
PLAPT               115         0.487             0.091             False           82               0.598                   0.106                   False
Savant-v2 esm8      115         0.704             0.083              True           82               0.585                   0.107                   False
Savant esm8         115         0.748             0.079              True           82               0.585                   0.107                   False
Boltz-2             115         0.739             0.080              True           82               0.573                   0.107                   False
QSAR per-target     115         0.670             0.086              True           82               0.537                   0.108                   False
PRODIGY-LIG         114         0.746             0.080              True           82               0.500                   0.108                   False
Tanimoto kNN        115         0.687             0.085              True           82               0.427                   0.107                   False
```

read corrected_accuracy; raw_accuracy is inflated by target base rates

paired vs Boltz-2
```
                  n_pairs  n_ligands  mean_diff  ci_low  ci_high  P_better  wins  losses  mcnemar_p
tool                                                                                               
Savant-v2 esm650       82         51      0.061  -0.081    0.198     0.762    17      12      0.458
Savant esm650          82         51      0.024  -0.114    0.160     0.596    16      14      0.856
PLAPT                  82         51      0.024  -0.117    0.169     0.618    17      15      0.860
Savant esm8            82         51      0.012  -0.135    0.153     0.524    16      15      1.000
Savant-v2 esm8         82         51      0.012  -0.153    0.167     0.499    17      16      1.000
QSAR per-target        82         51     -0.037  -0.176    0.091     0.244    18      21      0.749
PRODIGY-LIG            82         51     -0.073  -0.275    0.132     0.219    22      28      0.480
Tanimoto kNN           82         51     -0.146  -0.281    0.000     0.018    12      24      0.065
```

## D. protein taxonomy
```
                        n  targets  ligands  Savant esm650 | pooled_r  Savant esm650 | within  Savant esm8 | pooled_r  Savant esm8 | within  Savant-v2 esm650 | pooled_r  Savant-v2 esm650 | within  Savant-v2 esm8 | pooled_r  Savant-v2 esm8 | within  Boltz-2 | pooled_r  Boltz-2 | within  PRODIGY-LIG | pooled_r  PRODIGY-LIG | within  PLAPT | pooled_r  PLAPT | within  QSAR per-target | pooled_r  QSAR per-target | within  Tanimoto kNN | pooled_r  Tanimoto kNN | within
protein_family                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
carbonic anhydrase    130        4       49                     0.426                   0.372                   0.400                 0.220                        0.422                      0.325                      0.429                    0.291               0.404             0.147                   0.026                 0.069             0.220           0.003                       0.421                     0.253                    0.289                  0.004
GPCR class A           92        3       63                     0.705                   0.371                   0.655                 0.225                        0.690                      0.378                      0.620                    0.397               0.717             0.574                   0.554                 0.276             0.028           0.129                       0.286                    -0.352                    0.517                 -0.308
protein kinase         79        4       79                    -0.302                   0.062                  -0.286                 0.195                       -0.340                      0.254                     -0.428                    0.251              -0.077            -0.041                  -0.539                 0.357            -0.388          -0.258                      -0.009                     0.169                    0.215                  0.338
transcription factor   65        2       65                    -0.457                   0.110                  -0.175                -0.020                       -0.463                     -0.051                      0.206                   -0.133               0.215             0.132                   0.419                 0.101            -0.375          -0.077                       0.044                     0.148                   -0.046                 -0.027
small GTPase / PPI     48        2       48                     0.777                   0.601                   0.814                 0.633                        0.760                      0.565                      0.652                    0.465               0.468             0.286                   0.669                 0.406             0.375           0.490                      -0.195                     0.107                   -0.382                 -0.257
other                  31        2       31                     0.042                   0.207                   0.005                 0.078                        0.095                      0.446                      0.080                    0.497               0.206             0.271                   0.105                 0.268             0.250           0.191                       0.039                     0.122                   -0.089                 -0.241
isomerase              23        1       23                     0.515                   0.642                   0.489                 0.521                        0.438                      0.455                      0.382                    0.268               0.448             0.753                   0.202                 0.285             0.306           0.401                       0.321                     0.260                    0.496                  0.585
ion channel            17        1       17                       NaN                     NaN                     NaN                   NaN                          NaN                        NaN                        NaN                      NaN                 NaN               NaN                     NaN                   NaN               NaN             NaN                         NaN                       NaN                      NaN                    NaN
nuclear receptor       15        1       15                       NaN                     NaN                     NaN                   NaN                          NaN                        NaN                        NaN                      NaN                 NaN               NaN                     NaN                   NaN               NaN             NaN                         NaN                       NaN                      NaN                    NaN
```

## E. ligand chemotype
```
                        n  targets  ligands  Savant esm650 | pooled_r  Savant esm650 | within  Savant esm8 | pooled_r  Savant esm8 | within  Savant-v2 esm650 | pooled_r  Savant-v2 esm650 | within  Savant-v2 esm8 | pooled_r  Savant-v2 esm8 | within  Boltz-2 | pooled_r  Boltz-2 | within  PRODIGY-LIG | pooled_r  PRODIGY-LIG | within  PLAPT | pooled_r  PLAPT | within  QSAR per-target | pooled_r  QSAR per-target | within  Tanimoto kNN | pooled_r  Tanimoto kNN | within
ligand_class                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
fused heteroaromatic  212       14      174                     0.469                   0.163                   0.500                 0.102                        0.457                      0.223                      0.432                    0.216               0.555             0.274                   0.194                 0.191             0.160          -0.045                       0.189                     0.133                    0.214                 -0.027
primary sulfonamide   116        4       41                     0.140                   0.207                   0.219                 0.023                        0.156                      0.155                      0.209                    0.104               0.162            -0.110                   0.077                 0.090            -0.152          -0.286                       0.223                     0.042                    0.136                 -0.217
peptide-like           67        4       67                     0.052                   0.139                   0.272                 0.003                        0.107                     -0.029                      0.501                   -0.120               0.366             0.150                   0.505                 0.133            -0.335          -0.079                       0.344                     0.148                    0.337                 -0.027
macrocycle             36        2       36                     0.420                   0.354                   0.386                 0.247                        0.343                      0.230                      0.255                    0.072               0.430             0.342                  -0.017                 0.197             0.128           0.192                       0.097                     0.014                    0.491                  0.285
other                  35        8       23                     0.522                  -0.154                   0.491                -0.325                        0.497                     -0.200                      0.496                   -0.286               0.145             0.418                   0.231                 0.114             0.317           0.200                       0.546                     0.164                    0.319                  0.429
sulfonamide            22        3       22                     0.731                   0.446                   0.754                 0.537                        0.730                      0.452                      0.792                    0.599               0.704             0.087                   0.678                 0.412             0.530          -0.015                       0.059                     0.225                   -0.138                  0.456
halo-aromatic          10        4        9                       NaN                     NaN                     NaN                   NaN                          NaN                        NaN                        NaN                      NaN                 NaN               NaN                     NaN                   NaN               NaN             NaN                         NaN                       NaN                      NaN                    NaN
basic amine             2        2        1                       NaN                     NaN                     NaN                   NaN                          NaN                        NaN                        NaN                      NaN                 NaN               NaN                     NaN                   NaN               NaN             NaN                         NaN                       NaN                      NaN                    NaN
```

## F. vs models trained on the target itself
```
        n_train  n_bench  Savant esm650  Savant esm8  Savant-v2 esm650  Savant-v2 esm8  QSAR per-target  Tanimoto kNN  Boltz-2
Q01973        0       10         -0.394        0.018            -0.091          -0.018              NaN           NaN   -0.491
P28347       27       32         -0.030       -0.209            -0.255          -0.385              NaN           NaN   -0.028
Q99640       33       30          0.314        0.360             0.515           0.222            0.537         0.273    0.371
P42226      105       33          0.249        0.169             0.152           0.119            0.148        -0.027    0.292
P01116      190       33          0.786        0.807             0.786           0.557           -0.284        -0.454    0.382
Q07889      196       15          0.417        0.458             0.343           0.374            0.498        -0.059    0.190
P37840      242       13          0.288        0.169             0.631           0.907           -0.219        -0.692    0.329
P62937      473       23          0.642        0.521             0.455           0.268            0.260         0.585    0.753
P68400      704       19          0.446        0.537             0.452           0.599            0.225         0.456    0.087
P30518      926       29          0.268       -0.220             0.297           0.320           -0.479        -0.133    0.635
P31391      932       33          0.718        0.693             0.630           0.723           -0.526        -0.485    0.876
P37288     1364       30          0.127        0.203             0.207           0.149           -0.052        -0.306    0.212
P30530     1694       20         -0.117       -0.135             0.141           0.200           -0.254         0.284   -0.132
Q13224     1851       17         -0.098       -0.132             0.110           0.154            0.402         0.559    0.441
Q60492     2061       18          0.127       -0.013             0.261           0.086            0.463         0.209    0.214
O43570     3611       33          0.663        0.634             0.646           0.658            0.228        -0.029    0.270
P37231     4146       15          0.325        0.136             0.279           0.125            0.546         0.329   -0.157
Q16790     4565       32          0.221       -0.014             0.200          -0.024            0.332        -0.007    0.273
P00915     6791       32          0.059       -0.350            -0.137          -0.093            0.128         0.044   -0.049
P00918     7873       33          0.545        0.610             0.589           0.625            0.325         0.010    0.096
```

paired, best savant variant (Savant-v2 esm650)
```
                 n_targets  savant_minus  ci_low  ci_high  savant_wins  wilcoxon_p
vs                                                                                
QSAR per-target         18         0.238   0.038    0.461           11       0.108
Tanimoto kNN            18         0.333   0.127    0.565           12       0.016
Boltz-2                 20         0.082  -0.046    0.211           11       0.177
```

n_train is how many prior ligands that target had in training; read the crossover, not the mean