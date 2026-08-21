"""
Reshape benchmark_v2_scored.csv (wide, one row per benchmark entry)
into long format (one row per benchmark entry x tool) for Tableau.

Usage:
    python make_long_format.py
"""

import pandas as pd

INPUT_PATH = "benchmark_v2_scored.csv"
OUTPUT_PATH = "benchmark_v2_long.csv"

# Context columns to carry along for every row
ID_COLS = [
    "benchmark_id", "uniprot_id", "gene", "protein_name", "organism",
    "inchikey", "novelty_tier", "protein_class", "source",
    "molecular_weight", "logP",
    "max_train_protein_containment", "max_train_ligand_tanimoto",
    "num_rotatable_bonds", "num_rings", "boltz_status",
]

TRUE_COL = "neg_log10_affinity_M"

# Map each tool to its predicted-pKd column (and optional CI columns)
TOOL_MAP = {
    "boltz-2":     {"pred": "boltz_affinity_pic50"},
    "prodigy-lig": {"pred": "prodigy_pkd"},
    "savant":      {"pred": "savant_pk"},
    "savant v2":   {"pred": "savant2_pk", "lower": "savant2_pk_lower", "upper": "savant2_pk_upper"},
}


def main():
    df = pd.read_csv(INPUT_PATH)

    frames = []
    for tool, cols in TOOL_MAP.items():
        sub = df[ID_COLS + [TRUE_COL, cols["pred"]]].copy()
        sub = sub.rename(columns={TRUE_COL: "true_pkd", cols["pred"]: "predicted_pkd"})
        sub["tool"] = tool

        if "lower" in cols:
            sub["pred_ci_lower"] = df[cols["lower"]]
            sub["pred_ci_upper"] = df[cols["upper"]]
            sub["pred_ci_width"] = df[cols["upper"]] - df[cols["lower"]]
        else:
            sub["pred_ci_lower"] = pd.NA
            sub["pred_ci_upper"] = pd.NA
            sub["pred_ci_width"] = pd.NA

        frames.append(sub)

    long_df = pd.concat(frames, ignore_index=True)
    long_df["residual"] = long_df["predicted_pkd"] - long_df["true_pkd"]
    long_df["abs_error"] = long_df["residual"].abs()

    # Reorder columns for readability
    front = [
        "benchmark_id", "uniprot_id", "gene", "protein_name", "organism", "inchikey",
        "tool", "true_pkd", "predicted_pkd", "residual", "abs_error",
        "pred_ci_lower", "pred_ci_upper", "pred_ci_width",
        "novelty_tier", "protein_class", "source",
        "max_train_protein_containment", "max_train_ligand_tanimoto",
        "molecular_weight", "logP", "num_rotatable_bonds", "num_rings", "boltz_status",
    ]
    long_df = long_df[front]

    long_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(long_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
