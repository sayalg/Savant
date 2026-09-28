#!/usr/bin/env python3
"""Pre-download every model weight to $HF_HOME. Run on the login node.

The compute nodes DO have outbound network, so this is an optimisation rather than
a hard requirement - a cache miss on a compute node will simply download. It is
still worth running, for two reasons:

  * a cold 12-task array would have every task pull the same 2.5 GB ESM-2
    checkpoint simultaneously, which is slower than one staged copy and invites
    Hub rate-limiting;
  * staging once means the arrays fail fast on a missing model instead of burning
    GPU walltime on a download.

Set HF_OFFLINE=1 when submitting if you want a run pinned to the cache (useful for
reproducing a result months later); otherwise the sbatch scripts leave the Hub
reachable.

Usage
-----
    export HF_HOME=$PFPL_WORK/hf
    python env/stage_weights.py                # ESM-2 650M (the default plm.mode)
    python env/stage_weights.py --esm2-3b      # the Section 6.7 compute ablation
    python env/stage_weights.py --check        # verify the cache offline, download nothing

The model behind each `plm.mode` is listed in src/pfpl/features/embed_esm.py:MODELS.
Section 0.4.8: ESM-C 600M is non-commercial and 6B is API-only, so neither is
staged here without an explicit confirmation of a non-commercial context.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

## Importing pfpl.paths points HF_HOME (and every other HOME-bound cache) inside the
## project root, so staging lands where the jobs will look for it without anyone
## having to remember an export. Done before transformers is imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfpl import paths  # noqa: E402

MODELS = {
    "esm2_650m": ("facebook/esm2_t33_650M_UR50D", "esm", True,
                  "plm.mode=esm2_650m_full / esm2_bottleneck / esm2_pocket_window"),
    "esm2_3b": ("facebook/esm2_t36_3B_UR50D", "esm", False,
                "plm.mode=esm2_3b - the Section 6.7 compute ablation, ~11 GB"),
    "esmc_300m": ("EvolutionaryScale/esmc-300m-2024-12", "auto", False,
                  "plm.mode=esmc_300m - Cambrian Open License, commercial OK"),
}


def stage(key: str, check_only: bool) -> bool:
    repo, kind, _default, why = MODELS[key]
    print(f"\n--- {key}: {repo}\n    {why}")
    try:
        if check_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if kind == "esm":
            from transformers import AutoTokenizer, EsmModel
            AutoTokenizer.from_pretrained(repo)
            ## eager attention, because the contact-map centrality that feeds the
            ## pocket head needs attention weights. transformers 5.x defaults to SDPA,
            ## which returns none, and the contact path then dies with "stack expects
            ## a non-empty TensorList" - an error naming neither cause nor fix.
            ## Staging with the same setting catches it here instead of on a GPU node.
            m = EsmModel.from_pretrained(repo, attn_implementation="eager")
            print(f"    hidden size {m.config.hidden_size}")
        else:
            from transformers import AutoModel, AutoTokenizer
            AutoTokenizer.from_pretrained(repo)
            AutoModel.from_pretrained(repo)
        print("    OK")
        return True
    except Exception as e:
        msg = str(e)
        print(f"    FAILED: {type(e).__name__}: {msg.strip().splitlines()[0][:110]}")
        low = msg.lower()
        if "spiece.model" in low or "sentencepiece" in low or "tiktoken" in low:
            ## The weights usually downloaded fine and only the TOKENIZER failed. A
            ## protobuf parse error reads like a corrupt download when it actually
            ## means SentencePiece is not installed.
            print("    -> This is the TOKENIZER, not the weights.\n"
                  "       Fix:  pip install sentencepiece protobuf")
        elif "gated" in low or "401" in low or "403" in low:
            print("    -> The repo is gated. Accept its licence on the Hub and set\n"
                  "       HF_TOKEN. Section 0.4.8: ESM-C 600M is NON-COMMERCIAL and 6B\n"
                  "       is API-only; neither may be used without confirming the\n"
                  "       commercial context first.")
        elif check_only:
            print("    -> not in the cache. Re-run without --check on the login node.")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm2-3b", action="store_true",
                    help="also stage ESM-2 3B (Section 6.7 compute ablation, ~11 GB)")
    ap.add_argument("--esmc", action="store_true", help="also stage ESM-C 300M")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true", help="offline verification only")
    a = ap.parse_args()

    paths.ensure()
    home = os.environ["HF_HOME"]          # set by pfpl.paths.contain() on import
    print(f"PFPL_ROOT={paths.ROOT}\nHF_HOME={home}")
    if paths.escapes():
        print(f"  NOTE: {paths.escapes()} point outside the project root. Staging "
              "here will not be where the jobs look.", file=sys.stderr)

    want = ["esm2_650m"]
    if a.esm2_3b or a.all:
        want.append("esm2_3b")
    if a.esmc or a.all:
        want.append("esmc_300m")

    results = {k: stage(k, a.check) for k in want}
    print("\n" + "=" * 62)
    for k, ok in results.items():
        print(f"  {k:<12} {'OK' if ok else 'MISSING'}")
    print("=" * 62)
    if not results.get("esm2_650m"):
        print("\nesm2_650m is missing. Every plm.mode except `none` needs it.\n"
              "  plm.mode=none needs no weights at all and Section 6.7 asks for it to\n"
              "  be treated as a serious candidate rather than a fallback, so Wave 1\n"
              "  is not blocked by this.")
        sys.exit(1)
    for opt in ("esm2_3b", "esmc_300m"):
        if opt in results and not results[opt]:
            print(f"\n{opt} unavailable. It is an ABLATION ARM only (Section 6.7), so\n"
                  "nothing on the critical path depends on it - drop that cell from\n"
                  "the Phase 9 matrix and carry on.")


if __name__ == "__main__":
    main()
