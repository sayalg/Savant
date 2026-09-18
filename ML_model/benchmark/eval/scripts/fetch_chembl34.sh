#!/bin/bash
# Fetch ChEMBL v34 - the release Boltz-2's affinity module was trained on.
#
# v34 specifically, not "latest": the whole point is to reconstruct what Boltz
# could have seen. The live ChEMBL API only ever serves the current release
# (v37 as of 2026-05), so it cannot answer this question - the archived dump is
# the only source. Files are dated 2024-04-15 and never change, so a completed
# download is good forever.
#
#   bash scripts/fetch_chembl34.sh            # chemreps only  (226M)
#   bash scripts/fetch_chembl34.sh --full     # + sqlite       (4.5G, ~20G on disk)
#
# chemreps alone gets you ligand_seen and analog_seen. pair_seen - the stratum
# the audit actually turns on - needs the SQLite db, so --full is the real run.
set -euo pipefail

BASE="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34"
DEST="${CHEMBL34_DIR:-$(cd "$(dirname "$0")/.." && pwd)/work/chembl_34}"
FULL=0
[ "${1:-}" = "--full" ] && FULL=1

mkdir -p "${DEST}"
cd "${DEST}"
echo "== ChEMBL v34 -> ${DEST} =="

# -C - resumes a partial file, so a dropped connection on the 4.5G tarball does
# not restart it. Login nodes often cap wall time; re-run to pick up where it
# stopped.
fetch() {
    if [ -s "$1" ]; then
        echo "   have $1"
    else
        echo "   downloading $1"
        curl -fL -C - -o "$1" "${BASE}/$1"
    fi
}

fetch chembl_34_chemreps.txt.gz

if [ "${FULL}" -eq 1 ]; then
    fetch chembl_34_sqlite.tar.gz
    if ! ls chembl_34/chembl_34_sqlite/*.db >/dev/null 2>&1 \
       && ! ls ./*.db >/dev/null 2>&1; then
        echo "   extracting (needs ~20G free)"
        tar -xzf chembl_34_sqlite.tar.gz
    fi
    DB=$(find . -name 'chembl_34*.db' | head -1)
    if [ -n "${DB}" ]; then
        echo "   db: ${DB} ($(du -h "${DB}" | cut -f1))"
    else
        echo "   WARNING: no .db found after extraction" >&2
    fi
else
    echo "   skipping sqlite (pass --full for pair_seen)"
fi

cat <<EOF

Done. Next:
  export CHEMBL34_DIR="${DEST}"
  python scripts/08_contamination_audit.py $([ "${FULL}" -eq 1 ] || echo '# ligand_seen only')
EOF
