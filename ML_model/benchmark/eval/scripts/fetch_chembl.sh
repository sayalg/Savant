#!/bin/bash
# Fetch any archived ChEMBL release. fetch_chembl34.sh is the v34-only original,
# kept as-is because the contamination audit references it; this one takes the
# release as an argument, which the benchmark build needs for both sides of the diff.
#
#   bash scripts/fetch_chembl.sh 37            # chemreps only  (~250M)
#   bash scripts/fetch_chembl.sh 37 --full     # + sqlite       (5.4G, ~30G on disk)
#   bash scripts/fetch_chembl.sh 34 --full     # the Boltz-2 reference side
#
# build_benchmark.py needs --full on BOTH releases: v37 supplies the candidate Kd
# measurements, v34 supplies the ligand skeletons to exclude. chemreps alone is
# enough only if you skip the temporal gate, which defeats the point.
#
# Sizes as of 2026-08: v34 sqlite 4.5G, v37 sqlite 5.4G. Extracted is roughly
# 4-5x the tarball, so budget ~30G per release and ~60G for both.
set -euo pipefail

REL="${1:-}"
if ! [[ "${REL}" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 <release-number> [--full]     e.g. $0 37 --full" >&2
    exit 2
fi
FULL=0
[ "${2:-}" = "--full" ] && FULL=1

BASE="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_${REL}"
DEST="${CHEMBL_DIR:-$(cd "$(dirname "$0")/.." && pwd)/work/chembl_${REL}}"

mkdir -p "${DEST}"
cd "${DEST}"
echo "== ChEMBL v${REL} -> ${DEST} =="

# -C - resumes a partial file, so a dropped connection on the multi-GB tarball
# does not restart it. Login nodes often cap wall time; just re-run.
fetch() {
    if [ -s "$1" ]; then
        echo "   have $1"
    else
        echo "   downloading $1"
        curl -fL -C - -o "$1" "${BASE}/$1"
    fi
}

fetch "chembl_${REL}_chemreps.txt.gz"

if [ "${FULL}" -eq 1 ]; then
    fetch "chembl_${REL}_sqlite.tar.gz"
    if ! find . -name "chembl_${REL}*.db" | grep -q .; then
        echo "   extracting (needs ~30G free)"
        tar -xzf "chembl_${REL}_sqlite.tar.gz"
    fi
    DB=$(find . -name "chembl_${REL}*.db" | head -1)
    if [ -n "${DB}" ]; then
        echo "   db: ${DB} ($(du -h "${DB}" | cut -f1))"
        # Fail loudly here rather than inside build_benchmark's first query.
        if ! sqlite3 "${DB}" "SELECT COUNT(*) FROM activities LIMIT 1;" >/dev/null 2>&1; then
            echo "   WARNING: could not query activities in ${DB}" >&2
        fi
    else
        echo "   WARNING: no .db found after extraction" >&2
    fi
else
    echo "   skipping sqlite (pass --full; build_benchmark.py needs it)"
fi

cat <<EOF

Done. Benchmark v3 needs BOTH releases with --full:
  bash \$0 34 --full
  bash \$0 37 --full

Then, with DB34 / DB37 set to the two .db paths printed above:
  python ../../scripts/build_benchmark.py --chembl37 "\$DB37" --chembl34 "\$DB34" \
      --train ../../../../Input/input.csv --n-pairs 500 --workers 16 --dry-run
EOF
