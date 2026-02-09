#!/usr/bin/env python3

import pandas as pd
import subprocess
import hashlib
import json
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Structure, Model
import os

WORKDIR = Path("/scratch/sguirale/Savant/test")
WORKDIR.mkdir(exist_ok=True)

# ----------------------------
# Utilities
# ----------------------------

def run(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def uid(*args):
    return hashlib.md5("_".join(args).encode()).hexdigest()[:10]

# ----------------------------
# Prepare for AlphaFold3
# ----------------------------

def write_af3_json(
    protein_seq: str,
    smiles: str,
    out_json: Path,
    name: str,
    chain_id: str = "A",
    ligand_id: str = "B"
):
    af3_input = {
        "name": name,
        "modelSeeds": [1234],
        "sequences": [
            {
                "protein": {
                    "id": chain_id,
                    "sequence": protein_seq
                }
            },
            {
                "ligand": {
                    "id": ligand_id,
                    "smiles": smiles
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 1
    }

    with out_json.open("w") as f:
        json.dump(af3_input, f, indent=4)

# ----------------------------
# Run for AlphaFold3
# ----------------------------    

def run_alphafold3(
        json_input: Path, 
        outdir: Path
):

    AF3_RESOURCES_DIR = "/scratch/sguirale/AlphaFold3"
    AF3_IMAGE = f"{AF3_RESOURCES_DIR}/image/alphafold3.sif"
    AF3_CODE_DIR = f"{AF3_RESOURCES_DIR}/code"
    AF3_DATABASES_DIR = f"{AF3_RESOURCES_DIR}/databases"
    AF3_MODEL_PARAMETERS_DIR = "/projects/datasets/alphafold/3.0.0/weights"

    cmd = (
        f"singularity exec "
        "--nv "
        f"--bind {json_input}:/root/af_input "
        f"--bind {outdir}:/root/af_output "
        f"--bind {AF3_MODEL_PARAMETERS_DIR}:/root/models "
        f"--bind {AF3_DATABASES_DIR}:/root/public_databases "
        f"{AF3_IMAGE}"
        f"python {AF3_CODE_DIR}/alphafold3/run_alphafold.py "
        "--json_path=/root/af_input/af3_input.json "
        "--model_dir=/root/models "
        "--db_dir=/root/public_databases "
        "--output_dir=/root/af_output"
    )

    run(cmd)

# ----------------------------
# Docking Metrics
# ----------------------------

def score_complex(
    pdb_path: str, 
    ligand_chain: str = "B",
    n_cores: int = 4
) -> dict:
    
    pdb_file = Path(pdb_path).resolve()

    if not pdb_file.exists():
        print(f"Error: PDB file '{pdb_file}' not found.")
        return {}

    ## Parse the PDB file to extract the ligand
    print("Parsing PDB file to extract ligand...") 
    pdb_dir = pdb_file.parent
    ligand_pdb = pdb_dir / "ligand.pdb"
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein_structure", str(pdb_file))

    extracted_chain = None
    for model in structure:
        for chain in model:
            if chain.get_id().strip().upper() == ligand_chain:
                extracted_chain = chain
                break
        if extracted_chain:
            break

    if not extracted_chain:
        print(f"Chain '{ligand_chain}' not found in '{pdb_file.name}'")
        return {}
    
    # Wrap chain into new structure
    new_structure = Structure.Structure("ligand_only")
    new_model = Model.Model(0)
    new_model.add(extracted_chain.copy())
    new_structure.add(new_model)

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(str(ligand_pdb))
    print(f"Chain '{ligand_chain}' extracted to '{ligand_pdb}'")


    ## Run ACPYPE on ligand to generate topology files
    print("Running ACPYPE...")
    try:
        run(
            ["acpype", "-i", str(ligand_pdb), "-o", "cns"],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        print("ACPYPE Error:", e.stderr)
        return {}

    ## Create a configuration file for HADDOCK3 scoring
    cfg_body = f'''
    ## Docked protein complex scoring
    run_dir = "run0"
    mode = "local"
    ncores = {n_cores}
    molecules =  [
        "{pdb_file}"
        ]
    [topoaa]
    autohis = true
    ligand_param_fname = "{pdb_dir}/ligand.acpype/ligand_CNS.par"
    ligand_top_fname = "{pdb_dir}/ligand.acpype/ligand_CNS.top"
    [mdscoring]
    ligand_param_fname = "{pdb_dir}/ligand.acpype/ligand_CNS.par"
    ligand_top_fname = "{pdb_dir}/ligand.acpype/ligand_CNS.top"
    nemsteps = 1
    per_interface_scoring = false
    '''

    cfg_file = pdb_dir / "score.cfg"
    cfg_file.write_text(cfg_body.strip())

    ## Run HADDOCK3 scoring
    print("Running HADDOCK3 scoring...")
    try:
        run(
            ["haddock3", str(cfg_file)],
            capture_output=True, text=True, check=True
        )

        capri_file = pdb_dir / "run0/analysis/1_mdscoring_analysis/capri_ss.tsv"
        capri_ss = pd.read_csv(capri_file, sep="\t")
        metrics = capri_ss[['score', 'total', 'vdw', 'elec', 'desolv', 'bsa']].iloc[0].to_dict()

    except subprocess.CalledProcessError as e:
        print("HADDOCK3 Error:", e.stderr)
        return {}
    except Exception as e:
        print("Unexpected error:", e)
        return {}
    
    return metrics

# ----------------------------
# Main
# ----------------------------

df = pd.read_csv("input_test.csv")
rows = []

for i, row in df.iterrows():
    pid = uid(row.protein_seq, row.smiles)

    dock_dir = WORKDIR / "docking" / pid
    dock_dir.mkdir(parents=True, exist_ok=True)

    af3_json = WORKDIR / f"{pid}_af3_input.json"
    write_af3_json(
        protein_seq=row.protein_seq,
        smiles=row.smiles,
        out_json=af3_json,
        name=pid
    )

    run_alphafold3(
        json_input=af3_json,
        outdir=dock_dir
    )

    pdb = dock_dir / "output" / pid / f"{pid}_model.pdb"

    feats = score_complex(pdb, ligand_chain="B")
    feats["kd"] = row.kd
    rows.append(feats)

pd.DataFrame(rows).to_csv(WORKDIR / "dataset.csv", index=False)
