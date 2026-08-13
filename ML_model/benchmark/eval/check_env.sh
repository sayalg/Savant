#!/bin/bash
# Pre-flight check: everything the three stages need, before burning an allocation.
#   bash check_env.sh
cd "$(dirname "$0")"

ok=0
fail=0
note() { printf '  %-28s %s\n' "$1" "$2"; }
have() {
    if command -v "$1" >/dev/null 2>&1; then
        note "$1" "OK  $(command -v "$1")"; ok=$((ok + 1))
    else
        note "$1" "MISSING  ($2)"; fail=$((fail + 1))
    fi
}
pymod() {
    if python -c "import $1" 2>/dev/null; then
        note "python: $1" "OK  $(python -c "import $1; print(getattr($1, '__version__', '?'))" 2>/dev/null)"
        ok=$((ok + 1))
    else
        note "python: $1" "MISSING  ($2)"; fail=$((fail + 1))
    fi
}
file() {
    if [ -e "$1" ]; then note "$(basename "$1")" "OK  $1"; ok=$((ok + 1))
    else note "$(basename "$1")" "MISSING  $1"; fail=$((fail + 1)); fi
}

echo "== executables =="
have boltz       "pip install boltz[cuda] -U"
have savant      "poetry install / pip install -e . at the repo root"
have haddock3    "hard dependency of Savant, Linux only"
have prodigy_lig "pip install prodigy-lig"

echo
echo "== python packages =="
pymod rdkit  "needed by steps 01, 03"
pymod pandas "needed everywhere"
pymod numpy  "needed by step 06"

echo
echo "== inputs =="
file "../output/savant_benchmark_1000.csv"
file "../../train"                      # savant .pkl models live here
echo "  boltz cache                  ${BOLTZ_CACHE:-$HOME/.boltz}"
if [ -d "${BOLTZ_CACHE:-$HOME/.boltz}" ]; then
    note "boltz weights" "OK  $(du -sh "${BOLTZ_CACHE:-$HOME/.boltz}" 2>/dev/null | cut -f1) cached"
else
    note "boltz weights" "not cached - the first run downloads them (needs internet)"
fi

echo
echo "== gpu =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  /'
else
    echo "  no nvidia-smi (expected on a login node; Boltz needs a GPU node)"
fi

echo
echo "== msa server reachability =="
if curl -s -m 10 -o /dev/null -w '  api.colabfold.com HTTP %{http_code}\n' https://api.colabfold.com/ ; then :; else
    echo "  unreachable from here - run 02_fetch_msas.py somewhere with internet"
fi

echo
echo "$ok checks passed, $fail failed"
[ "$fail" -eq 0 ]
