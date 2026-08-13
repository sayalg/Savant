#!/bin/bash
#SBATCH --job-name=savant-bench-tiers
#SBATCH --output=logs/mmseqs_tiers_%j.out
#SBATCH --error=logs/mmseqs_tiers_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# ---- EDIT ME: cluster-specific scheduler directives ----
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
# --------------------------------------------------------
set -euo pipefail

# Confirms the benchmark's novelty tiers with a real all-vs-all sequence search.
# Local tiering used UniProt accession + 8-mer containment, which cannot see
# remote homology at ~30% identity; mmseqs can. Run this, copy the .m8 back,
# then: python scripts/08_reconcile_mmseqs.py --m8 <file>

# ---- EDIT ME: path to the container that carries mmseqs ----
# The colabfold image already ships it; adjust the path/runtime for your site.
CONTAINER="${CONTAINER:-$HOME/containers/colabfold.sif}"
RUN="apptainer exec ${CONTAINER}"      # use `singularity exec` on older sites
# If mmseqs is a module instead, comment the two lines above and use:
#   module load mmseqs2 && RUN=""
# ------------------------------------------------------------

cd "$(dirname "$0")"
mkdir -p logs work/mmseqs
cd work/mmseqs

THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Thresholds mirror the project's clustered split (mmseqs easy-cluster
# --min-seq-id 0.3): a hit at >=30% identity over >=80% of BOTH sequences
# means the benchmark target is homologous to a training chain.
MIN_SEQ_ID=0.30
COVERAGE=0.80

echo "== mmseqs easy-search: benchmark targets vs training chains =="
${RUN} mmseqs easy-search \
    bench_targets.fasta \
    train_chains.fasta \
    bench_vs_train.m8 \
    tmp_mmseqs \
    --min-seq-id "${MIN_SEQ_ID}" \
    -c "${COVERAGE}" \
    --cov-mode 0 \
    -s 7.5 \
    --alignment-mode 3 \
    --max-seqs 2000 \
    --format-output 'query,target,fident,alnlen,qcov,tcov,evalue,bits' \
    --threads "${THREADS}" \
    -v 1

rm -rf tmp_mmseqs
echo "== done: $(wc -l < bench_vs_train.m8) hits in work/mmseqs/bench_vs_train.m8 =="
echo "Copy it back and run: python scripts/08_reconcile_mmseqs.py --m8 work/mmseqs/bench_vs_train.m8"
