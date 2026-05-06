#!/usr/bin/python3
import os
import sys
import pandas as pd

def combine_csv(directory, csv_file):
    """Combine all CSV files in a directory into a single file."""
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    df_kd = pd.read_csv(csv_file)
    df_kd = df_kd[['ID', 'neg_log10_affinity_M']]
    
    dfs = [pd.read_csv(os.path.join(directory, f)) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = pd.merge(combined, df_kd, left_on='Job_ID', right_on='ID', how='left').drop('ID', axis=1)
    combined.to_csv('combined.csv', index=False)
    print(f"Combined {len(csv_files)} CSV files into combined.csv")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 combine_csv.py <directory> <csv_file>")
        sys.exit(1)

    combine_csv(sys.argv[1], sys.argv[2])
