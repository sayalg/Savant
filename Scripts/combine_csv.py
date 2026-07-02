#!/usr/bin/python3
import argparse
import glob
import json
import os

import pandas as pd


def collect_iptm(predictions_dir: str) -> pd.DataFrame:
    """Collect iptm values from AlphaFold summary_confidences.json files.

    Expects files matching:
        <predictions_dir>/<run_id>/<run_id>_summary_confidences.json

    Returns:
        DataFrame with columns [Job_ID, iptm].
    """
    pattern = os.path.join(predictions_dir, "*", "*_summary_confidences.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"No summary_confidences.json files found under: {predictions_dir}")
        return pd.DataFrame(columns=["Job_ID", "iptm"])

    records = []
    for path in json_files:
        run_id = os.path.basename(os.path.dirname(path))  # e.g. dock_1
        try:
            with open(path) as fh:
                data = json.load(fh)
            records.append({"Job_ID": run_id, "iptm": data["iptm"]})
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Skipping {path} — could not parse iptm: {e}")

    print(f"Collected iptm for {len(records)} run(s)")
    return pd.DataFrame(records)


def combine_csv(directory, csv_file, predictions_dir, existing_csv, output_csv):
    """Append newly finished per-job CSVs onto an existing combined CSV and
    backfill iptm for every row (existing and new) from predictions_dir."""
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    existing_df = pd.DataFrame()
    known_job_ids = set()
    if existing_csv and os.path.exists(existing_csv):
        existing_df = pd.read_csv(existing_csv)
        known_job_ids = set(existing_df['Job_ID'])
        print(f"Loaded {len(existing_df)} existing row(s) from {existing_csv}")

    dfs = [pd.read_csv(os.path.join(directory, f)) for f in csv_files]
    raw = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='Job_ID')

    new_rows = raw[~raw['Job_ID'].isin(known_job_ids)].copy()
    skipped = len(raw) - len(new_rows)
    if skipped:
        print(f"Skipping {skipped} row(s) already present in {existing_csv}")

    if new_rows.empty:
        print("No new rows to add.")
        combined = existing_df
    else:
        df_kd = pd.read_csv(csv_file)
        df_kd = df_kd[['ID', 'neg_log10_affinity_M', 'smiles']]
        new_rows = pd.merge(new_rows, df_kd, left_on='Job_ID', right_on='ID', how='left').drop('ID', axis=1)
        print(f"Adding {len(new_rows)} new row(s)")
        combined = pd.concat([existing_df, new_rows], ignore_index=True)

    # Backfill iptm for every row (existing rows may currently be blank).
    if 'iptm' in combined.columns:
        combined = combined.drop(columns=['iptm'])
    df_iptm = collect_iptm(predictions_dir)
    combined = pd.merge(combined, df_iptm, on='Job_ID', how='left')

    combined.to_csv(output_csv, index=False)
    print(f"Wrote {len(combined)} total row(s) to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append new per-job result CSVs onto an existing combined CSV, "
                     "merging affinity/smiles for new rows and backfilling iptm for all rows."
    )
    parser.add_argument("directory", help="Directory of per-job result CSVs (e.g. results/csv/)")
    parser.add_argument("csv_file", help="Master CSV with ID, neg_log10_affinity_M, smiles columns")
    parser.add_argument("predictions_dir", help="AlphaFold predictions dir containing summary_confidences.json files")
    parser.add_argument("existing_csv", help="Existing combined CSV to append onto (e.g. combined_40k.csv)")
    parser.add_argument("output_csv", help="Path to write the resulting combined CSV (e.g. combined_50k.csv)")
    args = parser.parse_args()

    combine_csv(args.directory, args.csv_file, args.predictions_dir, args.existing_csv, args.output_csv)
