#!/usr/bin/env python3
"""
AlphaFold3 to HADDOCK3 Protein-Ligand Docking Pipeline
"""

import json
import os
import subprocess
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
from rdkit.Chem import AllChem
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Structure, Model
import pymol

# Import MMseqs2 function directly to avoid subprocess overhead
from run_mmseqs2 import run_mmseqs2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AF3HaddockPipeline:

    def __init__(self, output_dir: str = "./pipeline_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.af3_dir = self.output_dir / "alphafold3"
        self.msa_dir = self.output_dir / "msa"
        self.ligand_dir = self.output_dir / "ligands"
        self.topology_dir = self.output_dir / "topologies"
        self.haddock_dir = self.output_dir / "haddock3"
        self.csv_dir = self.output_dir / "csv"

        for d in [self.af3_dir, self.msa_dir, self.ligand_dir,
                  self.topology_dir, self.haddock_dir, self.csv_dir]:
            d.mkdir(exist_ok=True)

    def run_mmseqs2_msa(self, protein_sequence: str, job_name: str,
                        max_seqs: int = 512) -> str:
        """
        Call the ColabFold MMseqs2 API, cap MSA depth, and save as .a3m.

        Returns the path to the saved a3m file.
        """
        logger.info(f"Running MMseqs2 MSA generation (max_seqs={max_seqs})...")

        prefix = str(self.msa_dir / job_name)
        a3m_path = Path(f"{prefix}_alignment.a3m")

        if a3m_path.exists():
            logger.info(f"Cached MSA found, skipping API call: {a3m_path}")
            return str(a3m_path)

        result = run_mmseqs2(protein_sequence, prefix)
        a3m_lines: List[str] = result[0] if isinstance(result, tuple) else result
        a3m_content = "".join(a3m_lines)

        # Cap MSA depth
        sequences, current_block = [], []
        for line in a3m_content.splitlines(keepends=True):
            if line.startswith(">") and current_block:
                sequences.append("".join(current_block))
                current_block = []
            current_block.append(line)
        if current_block:
            sequences.append("".join(current_block))

        if len(sequences) > max_seqs:
            logger.info(f"Truncating MSA from {len(sequences)} → {max_seqs} sequences")
            sequences = sequences[:max_seqs]

        with open(a3m_path, "w") as f:
            f.write("".join(sequences))

        logger.info(f"MSA saved: {a3m_path} ({len(sequences)} sequences)")
        return str(a3m_path)

    def create_af3_json(self, protein_sequence: str, smiles: str,
                        job_name: str = "protein_ligand",
                        msa_path: Optional[str] = None) -> str:
        """
        Create AF3 JSON input. When msa_path is provided the a3m content is
        embedded in unpairedMsa so AF3 skips its own MSA search.
        """
        logger.info("Creating AlphaFold3 JSON input...")

        protein_entry: Dict = {"id": "A", "sequence": protein_sequence}

        if msa_path:
            logger.info(f"Embedding pre-computed MSA from: {msa_path}")
            with open(msa_path) as f:
                protein_entry["unpairedMsa"] = f.read()
            protein_entry["pairedMsa"] = ""
            protein_entry["templates"] = []

        af3_input = {
            "name": job_name,
            "modelSeeds": [1],
            "sequences": [
                {"protein": protein_entry},
                {"ligand": {"id": "B", "smiles": smiles}}
            ],
            "dialect": "alphafold3",
            "version": 1
        }

        json_path = self.af3_dir / f"{job_name}_input.json"
        with open(json_path, 'w') as f:
            json.dump(af3_input, f, indent=2)

        logger.info(f"AlphaFold3 JSON created: {json_path}")
        return str(json_path)

    def run_alphafold3(self, job_name: str, json_input_path: str,
                       singularity_image: str, model_params_dir: str,
                       databases_dir: str, code_dir: str,
                       skip_data_pipeline: bool = False) -> str:
        """
        Run AF3 via Singularity. Passes --norun_data_pipeline when
        skip_data_pipeline=True (i.e. MSA is already embedded in the JSON).
        """
        logger.info("Running AlphaFold3...")

        output_path = self.af3_dir / "predictions"
        output_path.mkdir(exist_ok=True)
        json_input_dir = Path(json_input_path).parent

        cmd = [
            "singularity", "exec", "--nv",
            "--bind", f"{json_input_dir}:/root/af_input",
            "--bind", f"{output_path}:/root/af_output",
            "--bind", f"{model_params_dir}:/root/models",
            "--bind", f"{databases_dir}:/root/public_databases",
            singularity_image,
            "python", f"{code_dir}/alphafold3/run_alphafold.py",
            f"--json_path=/root/af_input/{job_name}_input.json",
            "--model_dir=/root/models",
            "--db_dir=/root/public_databases",
            "--output_dir=/root/af_output"
        ]

        if skip_data_pipeline:
            cmd.append("--norun_data_pipeline")
            logger.info("Data pipeline disabled — using embedded MSA")

        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("AlphaFold3 completed successfully")
            logger.debug(f"STDOUT: {result.stdout}")
            return str(output_path / job_name)
        except subprocess.CalledProcessError as e:
            logger.error(f"AlphaFold3 failed: {e.stderr}")
            raise

    def patch_cif_residue_name(self, cif_dir, job_name,
                                old_name="LIG_B", new_name="LIG"):
        cif_file = Path(cif_dir) / f"{job_name}_model.cif"
        if not cif_file.exists():
            raise FileNotFoundError(f"CIF file not found: {cif_file}")
        with open(cif_file) as f:
            content = f.read()
        if old_name not in content:
            logger.warning(f"'{old_name}' not found in CIF — skipping patch")
            return
        with open(cif_file, 'w') as f:
            f.write(content.replace(old_name, new_name))
        logger.info(f"Patched CIF: {cif_file}")

    def convert_cif_to_pdb(self, cif_dir, output_name="structure"):
        cif_file = Path(cif_dir) / f"{output_name}_model.cif"
        pdb_path = Path(cif_dir) / f"{output_name}_model.pdb"
        pymol.cmd.load(cif_file, "structure")
        pymol.cmd.save(str(pdb_path), "structure")
        pymol.cmd.delete("structure")
        logger.info(f"CIF converted to PDB: {pdb_path}")
        return str(pdb_path)

    def extract_ligand(self, af3_output_dir, job_name):
        logger.info("Extracting ligand from AlphaFold3 output...")
        structure_file = Path(af3_output_dir) / f"{job_name}_model.pdb"
        if not structure_file.exists():
            raise FileNotFoundError(f"PDB not found: {structure_file}")
        ligand_pdb = self.ligand_dir / f"{job_name}_ligand.pdb"
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", str(structure_file))
        extracted_chain = None
        for model in structure:
            for chain in model:
                if chain.get_id().strip().upper() == "B":
                    extracted_chain = chain
                    break
            if extracted_chain:
                break
        if not extracted_chain:
            raise ValueError(f"Chain B not found in {structure_file}")
        new_s = Structure.Structure("ligand_only")
        new_m = Model.Model(0)
        new_m.add(extracted_chain.copy())
        new_s.add(new_m)
        io = PDBIO()
        io.set_structure(new_s)
        io.save(str(ligand_pdb))
        logger.info(f"Ligand extracted: {ligand_pdb}")
        return str(ligand_pdb)

    # def extract_ligand(self, smiles, job_name):
    #     logger.info("Generating ligand structure from SMILES with Open Babel...")

    #     ligand_mol2 = self.ligand_dir / f"{job_name}_ligand.mol2"

    #     # Create molecule
    #     mol = pybel.readstring("smi", smiles)
    #     mol.addh()

    #     try:
    #         mol.make3D(forcefield="mmff94", steps=1000)
    #         mol.localopt(forcefield="mmff94", steps=2000)
    #     except:
    #         logger.warning("MMFF failed, switching to UFF")
    #         mol.make3D(forcefield="uff", steps=1000)
    #         mol.localopt(forcefield="uff", steps=2000)

    #     # remove weird geometries
    #     mol.OBMol.DeleteHydrogens()
    #     mol.addh()

    #     mol.write("mol2", str(ligand_mol2), overwrite=True)

    #     logger.info(f"Ligand generated: {ligand_mol2}")

    #     return str(ligand_mol2)
    
    def run_acpype(self, ligand_pdb_path, smiles):
        ligand_name = Path(ligand_pdb_path).stem
        mol_smiles = Chem.MolFromSmiles(smiles)
        if mol_smiles is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        net_charge = Chem.GetFormalCharge(mol_smiles)
        acpype_output = self.topology_dir / ligand_name
        acpype_output.mkdir(parents=True, exist_ok=True)
        cmd = [
            "singularity", "exec", "/users/sguirale/sw/acpype/acpype.sif",
            "acpype", 
            "-i", str(Path(ligand_pdb_path).resolve()),
            "-o", "cns", 
            "-n", str(net_charge)
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       cwd=str(acpype_output.resolve()))
        acpype_subdir = acpype_output / f"{ligand_name}.acpype"
        return {
            "par": str(acpype_subdir / f"{ligand_name}_CNS.par"),
            "top": str(acpype_subdir / f"{ligand_name}_CNS.top"),
            "acpype_dir": str(acpype_subdir)
        }

    def calculate_ligand_properties(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "num_h_donors": Lipinski.NumHDonors(mol),
            "num_h_acceptors": Lipinski.NumHAcceptors(mol),
            "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "tpsa": Descriptors.TPSA(mol),
            "num_rings": Lipinski.RingCount(mol),
            "num_aromatic_rings": Lipinski.NumAromaticRings(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": Lipinski.HeavyAtomCount(mol),
            "formal_charge": Chem.GetFormalCharge(mol),
        }

    def run_haddock3(self, job_name, protein_pdb, output_dir, topology_files):
        logger.info("Running HADDOCK3 docking...")
        # Create Job directory
        (Path(output_dir) / job_name).mkdir(parents=True, exist_ok=True)
        
        output_dir_path = Path(output_dir) / job_name / "output"
        cfg_body = f'''
        run_dir = "{output_dir_path}"
        mode = "local"
        ncores = {os.cpu_count()}
        molecules = ["{protein_pdb}"]
        [topoaa]
        autohis = true
        ligand_param_fname = "{topology_files["par"]}"
        ligand_top_fname = "{topology_files["top"]}"
        [mdscoring]
        ligand_param_fname = "{topology_files["par"]}"
        ligand_top_fname = "{topology_files["top"]}"
        nemsteps = 1
        per_interface_scoring = false
        '''
        cfg_file = Path(output_dir) / job_name / "score.cfg"
        cfg_file.write_text(cfg_body.strip())
        try:
            subprocess.run(["haddock3", str(cfg_file)],
                           capture_output=True, text=True, check=True)
            capri_ss = pd.read_csv(
                output_dir_path / "analysis/1_mdscoring_analysis/capri_ss.tsv",
                sep="\t"
            )
            return capri_ss[['score','total','vdw','elec','desolv','bsa']].iloc[0].to_dict()
        except Exception as e:
            logger.error(f"HADDOCK3 error: {e}")
            return {}

    def generate_csv_output(self, ligand_properties, docking_metrics, job_name):
        all_data = {**ligand_properties, **docking_metrics}
        df = pd.DataFrame([all_data])
        df.insert(0, 'Job_ID', job_name)
        csv_path = self.csv_dir / f"{job_name}_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved: {csv_path}")
        return str(csv_path)

    def run_pipeline(self, protein_sequence: str, smiles: str,
                     singularity_image: str, model_params_dir: str,
                     databases_dir: str, code_dir: str,
                     job_name: str = "docking_job",
                     start_from: str = "mmseqs2",
                     max_msa_seqs: int = 512) -> str:

        steps = ["mmseqs2", "alphafold3", "convert", "extract", "acpype", "haddock3"]
        skip = steps[:steps.index(start_from)]
        logger.info(f"Starting pipeline from step: '{start_from}'")

        try:
            af3_output = str(self.af3_dir / "predictions" / job_name)

            # Step 1 — MMseqs2
            msa_path: Optional[str] = None
            if "mmseqs2" not in skip:
                msa_path = self.run_mmseqs2_msa(protein_sequence, job_name,
                                                max_seqs=max_msa_seqs)
            else:
                candidate = self.msa_dir / f"{job_name}_alignment.a3m"
                if candidate.exists():
                    msa_path = str(candidate)
                    logger.info(f"Using cached MSA: {msa_path}")
                else:
                    logger.info("No cached MSA — AF3 will run its own MSA search")

            # Step 2 — AlphaFold3
            if "alphafold3" not in skip:
                json_path = self.create_af3_json(protein_sequence, smiles,
                                                 job_name, msa_path=msa_path)
                af3_output = self.run_alphafold3(
                    job_name, json_path, singularity_image,
                    model_params_dir, databases_dir, code_dir,
                    skip_data_pipeline=(msa_path is not None)
                )
            else:
                logger.info("Skipping AlphaFold3 — using existing outputs")

            # Step 3 — CIF → PDB
            if "convert" not in skip:
                self.patch_cif_residue_name(af3_output, job_name)
                self.convert_cif_to_pdb(af3_output, job_name)
            else:
                logger.info("Skipping CIF conversion")

            # Step 4 — Extract ligand
            ligand_pdb = str(self.ligand_dir / f"{job_name}_ligand.pdb")
            if "extract" not in skip:
                ligand_pdb = self.extract_ligand(af3_output, job_name)
            else:
                logger.info(f"Skipping ligand extraction — using: {ligand_pdb}")

            # Step 5 — ACPYPE
            if "acpype" not in skip:
                topology_files = self.run_acpype(ligand_pdb, smiles)
            else:
                ligand_name = Path(ligand_pdb).stem
                acpype_subdir = self.topology_dir / ligand_name / f"{ligand_name}.acpype"
                topology_files = {
                    "par": str(acpype_subdir / f"{ligand_name}_CNS.par"),
                    "top": str(acpype_subdir / f"{ligand_name}_CNS.top"),
                    "acpype_dir": str(acpype_subdir)
                }
                logger.info(f"Skipping ACPYPE — using: {acpype_subdir}")

            # Step 6 — HADDOCK3
            protein_pdb = str(
                self.af3_dir / "predictions" / job_name / f"{job_name}_model.pdb"
            )
            ligand_properties = self.calculate_ligand_properties(smiles)
            docking_metrics = {}
            if "haddock3" not in skip:
                docking_metrics = self.run_haddock3(
                    job_name, protein_pdb, str(self.haddock_dir), topology_files
                )
            else:
                logger.info("Skipping HADDOCK3")

            return self.generate_csv_output(ligand_properties, docking_metrics, job_name)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="AlphaFold3 to HADDOCK3 protein-ligand docking pipeline"
    )
    parser.add_argument("--protein", "-p", required=True)
    parser.add_argument("--smiles", "-s", required=True)
    parser.add_argument("--singularity-image", required=True)
    parser.add_argument("--model-params", required=True)
    parser.add_argument("--databases", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--output-dir", "-o", default="./pipeline_output")
    parser.add_argument("--job-name", "-n", default="docking_job")
    parser.add_argument(
        "--start-from",
        choices=["mmseqs2", "alphafold3", "convert", "extract", "acpype", "haddock3"],
        default="mmseqs2"
    )
    parser.add_argument("--max-msa-seqs", type=int, default=512)

    args = parser.parse_args()

    if os.path.isfile(args.protein):
        with open(args.protein) as f:
            protein_sequence = ''.join(l.strip() for l in f if not l.startswith('>'))
    else:
        protein_sequence = args.protein

    pipeline = AF3HaddockPipeline(output_dir=args.output_dir)
    csv_path = pipeline.run_pipeline(
        protein_sequence=protein_sequence,
        smiles=args.smiles,
        singularity_image=args.singularity_image,
        model_params_dir=args.model_params,
        databases_dir=args.databases,
        code_dir=args.code_dir,
        job_name=args.job_name,
        start_from=args.start_from,
        max_msa_seqs=args.max_msa_seqs
    )

    print(f"\n{'='*60}\nPipeline completed! Results: {csv_path}\n{'='*60}\n")


if __name__ == "__main__":
    main()