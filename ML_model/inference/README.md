# Predict with Savant

## Classification Model

```sh
python predict_classifier.py \
    --input example.csv \
    --output example_cls_output_20260730.csv \
    --model ../train/tender_comb_63w6mzz3_local.pkl
```

## Regression Model

```sh
python predict_regressor.py \
    --input example.csv \
    --output example_reg_output.csv \
    --model ../train/mango_sponge_66k38p1d_local.pkl
```