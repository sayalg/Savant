#!/usr/bin/env python3
"""
check_failures.py
─────────────────
Parses run_manifest.csv produced by the pipeline and:
  1. Prints a pass/fail summary
  2. Writes failed_jobs.csv — ready to pipe straight back into the pipeline
  3. Optionally annotates failures with Slurm exit codes from the Nextflow trace

Usage:
    python3 scripts/check_failures.py results/run_manifest.csv
    python3 scripts/check_failures.py results/run_manifest.csv \\
        --original-csv jobs.csv \\
        --trace results/pipeline_info/trace.tsv
"""

import csv
import sys
import argparse
from pathlib import Path
from collections import Counter


def load_csv(path, delimiter=','):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def print_summary(rows):
    counts = Counter(r['status'] for r in rows)
    total  = len(rows)
    n_ok   = counts.get('SUCCESS', 0)
    n_fail = counts.get('FAILED', 0)

    print()
    print("═" * 50)
    print(f"  RUN SUMMARY  ({total:,} total jobs)")
    print("═" * 50)
    print(f"  ✓  SUCCESS  {n_ok:>10,}  ({n_ok / total * 100:.1f}%)")
    print(f"  ✗  FAILED   {n_fail:>10,}  ({n_fail / total * 100:.1f}%)")
    print("═" * 50)
    print()


def write_rerun_csv(manifest_rows, original_csv_path, out_path):
    """Cross-reference failed names back to original CSV to rebuild input."""
    failed = {r['job_name'] for r in manifest_rows if r['status'] == 'FAILED'}
    if not failed:
        print("No failures found — skipping rerun CSV.")
        return

    if not Path(original_csv_path).exists():
        print(f"WARNING: original CSV not found at {original_csv_path}; "
              "cannot write rerun CSV.")
        return

    written = 0
    with open(original_csv_path, newline='') as fin, \
         open(out_path, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=['job_name', 'sequence', 'smiles'])
        writer.writeheader()
        for row in reader:
            if row['job_name'] in failed:
                writer.writerow(row)
                written += 1

    print(f"Rerun input written → {out_path}  ({written:,} jobs)")
    print(f"To rerun:  nextflow run main.nf --input_csv {out_path} -resume")
    print()


def annotate_with_trace(manifest_rows, trace_path):
    """Print Slurm exit codes for failures using Nextflow's trace TSV."""
    trace = {}
    for row in load_csv(trace_path, delimiter='\t'):
        tag = row.get('tag', '').strip()
        trace.setdefault(tag, []).append(row)

    failed = [r for r in manifest_rows if r['status'] == 'FAILED']
    if not failed:
        return

    print(f"{'job_name':<40} {'exit':>6}  {'slurm status':<14}  {'duration'}")
    print("─" * 80)
    for r in failed:
        name  = r['job_name']
        tasks = trace.get(name, [])
        if tasks:
            # Show the last attempt (highest attempt number)
            t = max(tasks, key=lambda x: int(x.get('attempt', 0) or 0))
            print(f"{name:<40} {t.get('exit', '?'):>6}  "
                  f"{t.get('status', '?'):<14}  {t.get('duration', '?')}")
        else:
            print(f"{name:<40} {'?':>6}  {'NOT IN TRACE':<14}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest",        help="Path to run_manifest.csv")
    parser.add_argument("--original-csv",  default="jobs.csv")
    parser.add_argument("--rerun-csv",     default="failed_jobs.csv")
    parser.add_argument("--trace",         default=None,
                        help="Nextflow trace TSV for Slurm exit code details")
    parser.add_argument("--no-rerun",      action="store_true")
    args = parser.parse_args()

    if not Path(args.manifest).exists():
        print(f"ERROR: {args.manifest} not found", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(args.manifest)
    print_summary(rows)

    if args.trace and Path(args.trace).exists():
        annotate_with_trace(rows, args.trace)

    if not args.no_rerun:
        write_rerun_csv(rows, args.original_csv, args.rerun_csv)

    # Non-zero exit if any failures — useful in monitoring / CI
    sys.exit(1 if any(r['status'] == 'FAILED' for r in rows) else 0)


if __name__ == "__main__":
    main()
