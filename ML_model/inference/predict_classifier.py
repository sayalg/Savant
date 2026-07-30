#!/usr/bin/env python3

import argparse
import json
import sys

import joblib
import pandas as pd


def load_input(args):
    """Load input data from JSON or CSV."""

    if args.record:
        return pd.DataFrame([json.loads(args.record)])

    if args.records:
        return pd.DataFrame(json.loads(args.records))

    if args.input:
        return pd.read_csv(args.input)

    raise ValueError("No input specified.")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using a trained scikit-learn model."
    )

    parser.add_argument(
        "--model",
        default="../tender_comb_63w6mzz3_local.pkl",
        help="Path to trained model (.pkl file)",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--record",
        help='Single JSON record, e.g. \'{"molecular_weight":312.538,"logP":7.1127}\'',
    )

    input_group.add_argument(
        "--records",
        help='JSON array of records, e.g. \'[{"molecular_weight":312.538,"logP":7.1127}]\'',
    )

    input_group.add_argument(
        "--input",
        help="Input CSV file",
    )

    parser.add_argument(
        "--output",
        help="Output CSV file (required when using --input)",
    )

    args = parser.parse_args()

    ## Validate arguments
    if args.input and not args.output:
        parser.error("--output is required when using --input")

    ## Load model
    model = joblib.load(args.model)

    ## Load input data
    X = load_input(args)

    ## Remove any columns from X that are not in the model's expected features
    model_features = model.named_steps["scaler"].get_feature_names_out()
    X = X[model_features]

    ## Predictions
    predictions = model.predict(X)

    ## Probabilities (if available)
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

    ##############
    ## CSV Mode
    ##############
    if args.input:
        output_df = X.copy()
        output_df["prediction"] = predictions

        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                output_df[f"probability_{i}"] = probabilities[:, i]

        output_df.to_csv(args.output, index=False)

        print(f"Wrote {len(output_df)} predictions to '{args.output}'")
        return

    ##############
    ## JSON Mode
    ##############
    result = {
        "predictions": predictions.tolist()
    }

    if probabilities is not None:
        result["probabilities"] = probabilities.tolist()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()