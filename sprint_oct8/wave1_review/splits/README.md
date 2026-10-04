# Frozen splits

Built from `work/pairs.parquet` (682,755 pairs). Protein clustering: **mmseqs @ 0.3**.

These files are frozen and version-controlled. Rebuilding them with a different seed invalidates every number measured against the old ones.

| split | train | val | test | discarded | protein overlap | scaffold overlap |
|---|---|---|---|---|---|---|
| `random` | 477,928 | 68,276 | 136,551 | 0 | 1864 | 34475 |
| `scaffold` | 477,917 | 68,283 | 136,555 | 0 | 1509 | 0 |
| `protein_cluster` | 474,327 | 68,666 | 139,762 | 0 | 0 | 7256 |
| `double` | 381,228 | 54,879 | 29,665 | 216,983 | 0 | 0 |

## What each one is for

- `random` - **sanity only**. Section 0.4.4 forbids headlining it.
- `scaffold` - ligand generalisation with proteins shared across folds.
- `protein_cluster` - **the headline split.** Whole mmseqs clusters move together, so a 95%-identical construct cannot be 'held out' while its twin trains. Section 9's comparison rule is defined on this split.
- `double` - unseen protein cluster AND unseen Murcko scaffold. Pairs that share exactly one axis are DISCARDED, which is the point: a pair whose protein is held out but whose scaffold was trained on measures memorisation, and mixing the two is how 0.580 internal became 0.327 external.
- `leave_viral_out` - the viral panel as a held-out application domain.

## Guarantees

`pfpl.data.splits.describe(strict=True)` raises if a split has any group in both folds. `tests/test_leakage.py` re-checks it in CI.
