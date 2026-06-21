#!/usr/bin/env python3
"""
AlphaFold3 to HADDOCK3 Protein-Ligand Docking Pipeline
"""

import json
import os
import subprocess
import tempfile
import sys
import glob
import gzip
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AF3HaddockPipeline:

    def __init__(self, output_dir: str = "./pipeline_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.af3_dir = self.output_dir / "alphafold3"
        self.msa_dir = (self.output_dir / "msa").resolve()
        self.protein_dir = self.output_dir / "proteins"
        self.ligand_dir = self.output_dir / "ligands"
        self.topology_dir = self.output_dir / "topologies"
        self.haddock_dir = self.output_dir / "haddock3"
        self.csv_dir = self.output_dir / "csv"
        
        # Speed up MMseqs2 by using a local copy of the UniRef30 database and tuning parameters for HPC
        default_mmseqs2_db = "/scratch/sguirale/mmseq2_db"  # <-- Point this to your local MMseqs2 DB prefix
        self.mmseqs2_db_path = Path(os.environ.get("MMSEQS2_DB", default_mmseqs2_db))  # point to the DB prefix
        self.mmseqs2_threads = 16  # tune to your CPU

        for d in [self.af3_dir, self.msa_dir, self.protein_dir, self.ligand_dir,
                  self.topology_dir, self.haddock_dir, self.csv_dir]:
            d.mkdir(exist_ok=True)

    def run_mmseqs2_msa(self, protein_sequence: str, job_name: str,
                        max_seqs: int = 512) -> str:
        """
        Run MMseqs2 locally against UniRef30, cap MSA depth, and save as .a3m.
        Returns the path to the saved a3m file.
        """
        logger.info(f"Running local MMseqs2 MSA generation (max_seqs={max_seqs})...")
        a3m_path = self.msa_dir / f"{job_name}_alignment.a3m"

        if a3m_path.exists():
            logger.info(f"Cached MSA found, skipping: {a3m_path}")
            return str(a3m_path)

        query_fasta = self.msa_dir / f"{job_name}_query.fasta"
        result_dir = self.msa_dir / f"{job_name}_results"
        result_dir.mkdir(exist_ok=True)
        query_fasta.write_text(f">{job_name}\n{protein_sequence}\n")

        cmd = [
            "singularity", "run",
            "-B", "/users/sguirale/alphafold2/alphafold_weights:/cache",
            "-B", f"{self.msa_dir}:{self.msa_dir}",
            "-B", f"{self.mmseqs2_db_path}:/tmp/mmseqs2_db",
            "/users/sguirale/alphafold2/colabfold_1.6.1-cuda12.sif",
            "colabfold_search",
            str(query_fasta),
            "/tmp/mmseqs2_db",   # container-side path
            str(result_dir),
            "--threads", str(self.mmseqs2_threads),
            "--gpu", "0",
            "-s", "4",                     # <--- Lower sensitivity (Default is 8). Huge speedup.
            "--db-load-mode", "3",          # <--- Tells MMseqs2 to hold the index layout efficiently in memory
        ]
        logger.info(f"MMseqs2 command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"colabfold_search stderr:\n{e.stderr}")
            logger.error(f"colabfold_search stdout:\n{e.stdout}")
            raise

        raw_a3m = result_dir / f"{job_name}.a3m"
        if not raw_a3m.exists():
            a3m_files = list(result_dir.glob("*.a3m"))
            if a3m_files:
                raw_a3m = a3m_files[0]
            else:
                raise FileNotFoundError(f"MMseqs2 produced no .a3m output in {result_dir}")
                
        a3m_content = raw_a3m.read_text()

        # Parse sequences
        sequences, current_block = [], []
        for line in a3m_content.splitlines(keepends=True):
            if line.startswith(">") and current_block:
                sequences.append("".join(current_block))
                current_block = []
            current_block.append(line)
        if current_block:
            sequences.append("".join(current_block))

        # Capping MSA depth
        if len(sequences) > max_seqs:
            logger.info(f"Truncating MSA from {len(sequences)} → {max_seqs} sequences")
            sequences = sequences[:max_seqs]

        a3m_path.write_text("".join(sequences))
        logger.info(f"MSA saved: {a3m_path} ({len(sequences)} sequences)")
        return str(a3m_path)

    def run_mmseqs2_msa_batch(self, batch_input_dir: Path, batch_output_dir: Path) -> None:
        """
        Run colabfold_search in batch mode over a directory containing multiple FASTA files.
        """
        logger.info(f"Running batch MMseqs2 MSA generation on directory: {batch_input_dir}")
        
        cmd = [
            "singularity", "exec",
            "-B", "/users/sguirale/alphafold2/alphafold_weights:/cache",
            "-B", f"{self.msa_dir}:{self.msa_dir}",
            "-B", f"{self.mmseqs2_db_path}:/tmp/mmseqs2_db",
            "/users/sguirale/alphafold2/colabfold_1.6.1-cuda12.sif",
            "env", "MMSEQS_IGNORE_INDEX=1",
            "colabfold_search",
            str(batch_input_dir),
            "/tmp/mmseqs2_db",
            str(batch_output_dir),
            "--threads", str(self.mmseqs2_threads),
            "--gpu", "0",
            "-s", "4",                     # Sensitivity 4 for massive speedup
            "--db-load-mode", "3",          # Optimized memory paging for HPC filesystems
            "--prefilter-mode", "1"        # MMseqs2 prefiltering for even faster batch processing
        ]
        
        logger.info(f"Executing Batch MMseqs2 command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Batch MMseqs2 completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"colabfold_search batch stderr:\n{e.stderr}")
            logger.error(f"colabfold_search batch stdout:\n{e.stdout}")
            raise

    def process_and_cap_a3m(self, raw_a3m: Path, job_name: str, max_seqs: int = 512) -> str:
        """
        Takes a raw generated .a3m file, caps its depth, and writes it to the final tracking path.
        """
        a3m_path = self.msa_dir / f"{job_name}_alignment.a3m"
        
        if a3m_path.exists():
            return str(a3m_path)
            
        if not raw_a3m.exists():
            raise FileNotFoundError(f"Expected batch output alignment missing: {raw_a3m}")
            
        a3m_content = raw_a3m.read_text()

        # Parse sequences
        sequences, current_block = [], []
        for line in a3m_content.splitlines(keepends=True):
            if line.startswith(">") and current_block:
                sequences.append("".join(current_block))
                current_block = []
            current_block.append(line)
        if current_block:
            sequences.append("".join(current_block))

        # Cap MSA depth
        if len(sequences) > max_seqs:
            logger.info(f"Truncating MSA for {job_name} from {len(sequences)} → {max_seqs} sequences")
            sequences = sequences[:max_seqs]

        a3m_path.write_text("".join(sequences))
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

    def extract_chains(self, af3_output_dir, job_name):
        logger.info("Extracting ligand and protein from AlphaFold3 output...")
        structure_file = Path(af3_output_dir) / f"{job_name}_model.pdb"
        if not structure_file.exists():
            raise FileNotFoundError(f"PDB not found: {structure_file}")

        ligand_pdb = self.ligand_dir / f"{job_name}_ligand.pdb"
        protein_pdb = self.protein_dir / f"{job_name}_protein.pdb"

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", str(structure_file))

        model = next(iter(structure))
        chains = {chain.get_id().strip().upper(): chain for chain in model}

        if "A" not in chains:
            raise ValueError(f"Chain A not found in {structure_file}")
        if "B" not in chains:
            raise ValueError(f"Chain B not found in {structure_file}")

        io = PDBIO()

        for chain_id, out_path, label in [("A", protein_pdb, "Protein"), ("B", ligand_pdb, "Ligand")]:
            new_s = Structure.Structure(label.lower())
            new_m = Model.Model(0)
            new_m.add(chains[chain_id].copy())
            new_s.add(new_m)
            io.set_structure(new_s)
            io.save(str(out_path))
            logger.info(f"{label} extracted: {out_path}")

        return str(protein_pdb), str(ligand_pdb)

    def calculate_ligand_properties(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        morgan_fp_dict = {f"morgan_fp_{i}": int(b) for i, b in enumerate(fp)}
        
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
            **morgan_fp_dict,
        }
    
    def count_interacting_residues(self, docked_pdb: str, cutoff: float = 4.5) -> dict:
        """Count protein residues within cutoff (Å) of the ligand (chain B)."""
        with gzip.open(docked_pdb, 'rt') as f:
            structure = PDBParser(QUIET=True).get_structure("docked", f)
        model = next(iter(structure))

        ligand_atoms = [atom for atom in model.get_atoms()
                        if atom.get_full_id()[2] == "B"]
        protein_residues = {residue for residue in model.get_residues()
                            if residue.get_full_id()[2] == "A"}

        interacting = set()
        for res in protein_residues:
            for p_atom in res.get_atoms():
                for l_atom in ligand_atoms:
                    if (p_atom.get_vector() - l_atom.get_vector()).norm() <= cutoff:
                        interacting.add(res.get_id())
                        break
                else:
                    continue
                break

        return {"num_interacting_residues": len(interacting)}

    def run_haddock3(self, job_name, protein_pdb, ligand_pdb, output_dir):
        logger.info("Running HADDOCK3 docking...")
        # Create Job directory
        (Path(output_dir) / job_name).mkdir(parents=True, exist_ok=True)
        
        output_dir_path = Path(output_dir) / job_name / "output"
        cfg_body = f'''
        run_dir = "{output_dir_path}"
        mode = "local"
        debug = true
        ncores = {os.cpu_count()}

        molecules = [
            "{protein_pdb}",
            "{ligand_pdb}"
        ]

        [topoaa]
        autotoppar = true

        [emref]
        nemsteps = 50
        '''
        cfg_file = Path(output_dir) / job_name / "score.cfg"
        cfg_file.write_text(cfg_body.strip())
        try:
            subprocess.run(["haddock3", str(cfg_file)],
                           capture_output=True, text=True, check=True)
            capri_ss = pd.read_csv(
                output_dir_path / "analysis/1_emref_analysis/capri_ss.tsv",
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

        steps = ["mmseqs2", "alphafold3", "convert", "extract", "haddock3"]
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

            # Step 4 — Extract Chains
            protein_pdb = str(self.protein_dir / f"{job_name}_protein.pdb")
            ligand_pdb = str(self.ligand_dir / f"{job_name}_ligand.pdb")
            if "extract" not in skip:
                protein_pdb, ligand_pdb = self.extract_chains(af3_output, job_name)
            else:
                logger.info(f"Skipping chain extraction — using: {protein_pdb} and {ligand_pdb}")

            # Step 5 - Calculate Ligand Properties
            ligand_properties = self.calculate_ligand_properties(smiles)

            # Step 6 - Docking
            docking_metrics = {}
            if "haddock3" not in skip:
                docking_metrics = self.run_haddock3(
                    job_name, protein_pdb, ligand_pdb, str(self.haddock_dir)
                )
            else:
                logger.info("Skipping HADDOCK3")
            
            # Step 7 - Get protein-ligand residue interactions
            # Count interacting residues from docked structure
            interaction_metrics = {}
            if "haddock3" not in skip:
                # Use the HADDOCK output PDB — adjust path to match your emref output
                docked_pdb = str(self.haddock_dir / job_name / "output" / "1_emref" / "emref_1.pdb.gz")
                interaction_metrics = self.count_interacting_residues(docked_pdb)

            # Step 7 - Generate final CSV
            return self.generate_csv_output({**ligand_properties, **interaction_metrics}, docking_metrics, job_name)

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
        choices=["mmseqs2", "alphafold3", "convert", "extract", "haddock3"],
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