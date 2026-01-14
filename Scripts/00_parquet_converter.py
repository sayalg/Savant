#!/usr/bin/python3

import pandas as pd
import sys

# Define input and output file paths
parquet_file = sys.argv[1]
csv_output = sys.argv[2]

df = pd.read_parquet(parquet_file, engine='pyarrow')

df.to_csv(csv_output, index=False)

print(f"Successfully converted {parquet_file} to {csv_output}")