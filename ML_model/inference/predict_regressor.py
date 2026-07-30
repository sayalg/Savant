#!/usr/bin/env python3

import argparse
import json

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


def get_model_features(model, X):
    """
    Determine the feature names expected by the model.
    """

    ## sklearn >=1.0 pipelines usually expose feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    ## Many sklearn transformers expose this
    if hasattr(model.named_steps["scaler"], "feature_names_in_"):
        return list(model.named_steps["scaler"].feature_names_in_)

    if hasattr(model.named_steps["scaler"], "get_feature_names_out"):
        return list(model.named_steps["scaler"].get_feature_names_out())

    ## Last resort: assume input columns are already correct
    return list(X.columns)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using a trained XGBoost regression pipeline."
    )

    parser.add_argument(
        "--model",
        default="../mango_sponge_66k38p1d_local.pkl",
        help="Path to trained Savant regression model (.pkl)",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument("--record")
    input_group.add_argument("--records")
    input_group.add_argument("--input")

    parser.add_argument("--output")

    args = parser.parse_args()

    if args.input and not args.output:
        parser.error("--output is required when using --input")

    ## Load model
    model = joblib.load(args.model)

    ## Patch Scaler if clip is missing
    scaler = model.named_steps["scaler"]
    if not hasattr(scaler, "clip"):
        scaler.clip = False

    ## Load input
    X = load_input(args)

    ## Match training feature order
    model_features = get_model_features(model, X)

    missing = [c for c in model_features if c not in X.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required feature(s): {missing}"
        )

    X = X.loc[:, model_features]

    ## Predict
    predictions = model.predict(X)

    ################
    # CSV mode
    ################

    if args.input:
        output_df = X.copy()
        output_df["prediction"] = predictions
        output_df.to_csv(args.output, index=False)

        print(f"Wrote {len(output_df)} predictions to '{args.output}'")
        return

    ################
    # JSON mode
    ################

    print(
        json.dumps(
            {
                "predictions": predictions.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()