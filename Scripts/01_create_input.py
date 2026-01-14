#!/usr/bin/python3

import pandas as pd

# Import data
input = pd.read_csv("all.csv", sep = ",")

# Separate columns of interest
df = input[['seq', 'smiles', 'neg_log10_affinity_M']]

# Create ID column and convert neg_log10 to Kd
prefix = "dock_"
ids = prefix + (df.index + 1).astype(str)
df.insert(0, "ID", value=ids)

def convert_pKd(pKd):
    Kd = 10**(-pKd)
    nKd = round(Kd * 1e9, 5)
    return nKd

df['affinity_Kd_nM'] = df['neg_log10_affinity_M'].apply(convert_pKd)

df.to_csv('input.csv', index=False)