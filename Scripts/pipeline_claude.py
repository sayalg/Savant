#!/usr/bin/env python3
"""
AlphaFold3 to HADDOCK3 Protein-Ligand Docking Pipeline

This script processes protein sequences and ligands through AlphaFold3 structure prediction,
ligand extraction, topology generation, and HADDOCK3 docking.

Dependencies:
    - rdkit
    - json
    - subprocess
    - pandas
    - biopython
    - pymol
"""

import json
import os
import subprocess
import sys
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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AF3HaddockPipeline:
    """Pipeline for protein-ligand docking using AlphaFold3 and HADDOCK3"""
    
    def __init__(self, output_dir: str = "./pipeline_output"):
        """
        Initialize the pipeline
        
        Args:
            output_dir: Directory for all output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define subdirectories
        self.af3_dir = self.output_dir / "alphafold3"
        self.ligand_dir = self.output_dir / "ligands"
        self.topology_dir = self.output_dir / "topologies"
        self.haddock_dir = self.output_dir / "haddock3"
        
        for dir_path in [self.af3_dir, self.ligand_dir, self.topology_dir, self.haddock_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def create_af3_json(self, protein_sequence: str, smiles: str, 
                        job_name: str = "protein_ligand") -> str:
        """
        Create AlphaFold3 JSON input file
        
        Args:
            protein_sequence: Amino acid sequence
            smiles: SMILES string for ligand
            job_name: Name for the job
            
        Returns:
            Path to created JSON file
        """
        logger.info("Creating AlphaFold3 JSON input...")
        
        af3_input = {
            "name": job_name,
            "modelSeeds": [1],
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": protein_sequence
                    }
                },
                {
                    "ligand": {
                        "id": "B",
                        "smiles": smiles
                    }
                }
            ],
            "dialect": "alphafold3",
            "version": 1
        }
        
        json_path = self.af3_dir / f"{job_name}_input.json"
        with open(json_path, 'w') as f:
            json.dump(af3_input, f, indent=2)
        
        logger.info(f"AlphaFold3 JSON created: {json_path}")
        return str(json_path)
    
    def run_alphafold3(self, job_name: str,
                       json_input_path: str, 
                       singularity_image: str,
                       model_params_dir: str,
                       databases_dir: str,
                       code_dir: str) -> str:
        """
        Run AlphaFold3 using singularity container
        
        Args:
            json_input_path: Path to AF3 JSON input
            singularity_image: Path to AF3 singularity image (.sif file)
            model_params_dir: Path to AlphaFold3 model parameters
            databases_dir: Path to sequence databases
            
        Returns:
            Path to output directory
        """
        logger.info("Running AlphaFold3...")
        
        output_path = self.af3_dir / "predictions"
        output_path.mkdir(exist_ok=True)

        json_input_dir = Path(json_input_path).parent
        
        cmd = [
            "singularity", "exec",
            "--nv",
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
        
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("AlphaFold3 completed successfully")
            logger.debug(f"STDOUT: {result.stdout}")
            return str(output_path / job_name)
        except subprocess.CalledProcessError as e:
            logger.error(f"AlphaFold3 failed: {e}")
            logger.error(f"STDERR: {e.stderr}")
            raise
    
    def convert_cif_to_pdb(self, cif_dir: str, output_name: str = "structure") -> str:
        """
        Convert CIF file to PDB format using PyMOL
        
        Args:
            cif_file: Path to input CIF file
            output_name: Name for output PDB file (without extension)
            
        Returns:
            Path to converted PDB file
        """
        cif_file = Path(cif_dir) / f"{output_name}_model.cif"
        
        logger.info(f"Converting CIF to PDB: {cif_file}")
        
        try:
            
            pdb_path = Path(cif_dir) / f"{output_name}_model.pdb"
            
            pymol.cmd.load(cif_file, "structure")
            pymol.cmd.save(str(pdb_path), "structure")
            pymol.cmd.delete("structure")
            
            logger.info(f"CIF converted to PDB: {pdb_path}")
            
        except Exception as e:
            logger.error(f"CIF to PDB conversion failed: {e}")
            raise
    
    def extract_ligand(self, af3_output_dir: str, job_name: str) -> str:
        """
        Extract ligand structure from AlphaFold3 output
        
        Args:
            af3_output_dir: Directory containing AF3 predictions
            ligand_name: Name for extracted ligand file
            
        Returns:
            Path to extracted ligand PDB file
        """
        logger.info("Extracting ligand from AlphaFold3 output...")
        
        # Find the predicted structure file
        af3_dir_path = Path(af3_output_dir)
        structure_file = list(af3_dir_path.glob("*.pdb"))
        
        if not structure_file:
            raise FileNotFoundError(f"No PDB files found in {af3_output_dir}")

        # Get AF3 PDB file
        logger.info(f"Using structure: {structure_file}")
        
        # Extract ligand
        ligand_pdb = self.ligand_dir / f"{job_name}_ligand.pdb"
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein_structure", str(structure_file))

        extracted_chain = None
        for model in structure:
            for chain in model:
                if chain.get_id().strip().upper() == "B":
                    extracted_chain = chain
                    break
            if extracted_chain:
                break

        if not extracted_chain:
            print(f"Chain 'B' not found in '{structure_file.name}'")
            return {}
        
        # Wrap chain into new structure
        new_structure = Structure.Structure("ligand_only")
        new_model = Model.Model(0)
        new_model.add(extracted_chain.copy())
        new_structure.add(new_model)

        io = PDBIO()
        io.set_structure(new_structure)
        io.save(str(ligand_pdb))
        
        logger.info(f"Ligand extracted to: {ligand_pdb}")
        return str(ligand_pdb)
    
    def run_acpype(self, ligand_pdb_path: str) -> Dict[str, str]:
        """
        Run ACPYPE to generate topology files
        
        Args:
            ligand_pdb_path: Path to ligand PDB file
            
        Returns:
            Dictionary with paths to topology files
        """
        logger.info("Running ACPYPE for topology generation...")
        
        ligand_name = Path(ligand_pdb_path).stem
        acpype_output = self.topology_dir / ligand_name
        
        # Ensure output directory exists and run acpype inside it
        acpype_output.mkdir(parents=True, exist_ok=True)
        acpype_output_str = str(acpype_output.resolve())
        ligand_pdb_abs = str(Path(ligand_pdb_path).resolve())

        # Use a shell wrapper to cd into the desired output directory before running acpype and then cd back to the original directory
        cmd = [
            f"cd {acpype_output_str} && acpype -i {ligand_pdb_abs} -o cns && cd -"
        ]
        
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("ACPYPE completed successfully")
            
            # Find generated files
            topology_files = {
                "par": str(acpype_output / f"{ligand_name}_CNS.par"),
                "top": str(acpype_output / f"{ligand_name}_CNS.top")
            }
            
            return topology_files
        except subprocess.CalledProcessError as e:
            logger.error(f"ACPYPE failed: {e}")
            logger.error(f"STDERR: {e.stderr}")
            raise
    
    def calculate_ligand_properties(self, smiles: str) -> Dict[str, float]:
        """
        Calculate chemical properties of ligand using RDKit
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of chemical properties
        """
        logger.info("Calculating ligand chemistry properties...")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Add hydrogens for accurate calculations
        mol = Chem.AddHs(mol)
        
        properties = {
            "molecular_weight": Descriptors.MolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "num_h_donors": Lipinski.NumHDonors(mol),
            "num_h_acceptors": Lipinski.NumHAcceptors(mol),
            "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "tpsa": Descriptors.TPSA(mol),  # Topological polar surface area
            "num_rings": Lipinski.RingCount(mol),
            "num_aromatic_rings": Lipinski.NumAromaticRings(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": Lipinski.HeavyAtomCount(mol),
            "formal_charge": Chem.GetFormalCharge(mol),
        }
        
        logger.info("Chemistry properties calculated")
        return properties
    
    def run_haddock3(self, job_name: str, protein_pdb: str, output_dir: str, 
                     topology_files: Dict[str, str]) -> Dict[str, float]:
        """
        Run HADDOCK3 docking (placeholder for custom implementation)
        
        Args:
            protein_pdb: Path to protein structure
            ligand_pdb: Path to ligand structure
            topology_files: Dictionary of topology file paths
            
        Returns:
            Dictionary of docking metrics
        """
        logger.info("Running HADDOCK3 docking...")
        
        topology_dir = Path(topology_files).parent
        output_dir_path = Path(output_dir)
        cores = os.cpu_count()
        
        cfg_body = f'''
        ## Docked protein complex scoring
        run_dir = "{output_dir_path}"
        mode = "local"
        ncores = {cores}
        molecules =  [
            "{protein_pdb}"
            ]
        [topoaa]
        autohis = true
        ligand_param_fname = "{topology_dir}/{job_name}_ligand.acpype/{topology_files["par"]}"
        ligand_top_fname = "{topology_dir}/{job_name}_ligand.acpype/{topology_files["top"]}"
        [mdscoring]
        ligand_param_fname = "{topology_dir}/{job_name}_ligand.acpype/{topology_files["par"]}"
        ligand_top_fname = "{topology_dir}/{job_name}_ligand.acpype/{topology_files["top"]}"
        nemsteps = 1
        per_interface_scoring = false
        '''

        cfg_file = output_dir_path / "score.cfg"
        cfg_file.write_text(cfg_body.strip())

        ## Run HADDOCK3 scoring
        logger.info("Running HADDOCK3 scoring...")
        try:
            subprocess.run(
                ["haddock3", str(cfg_file)],
                capture_output=True, text=True, check=True
            )

            capri_file = output_dir_path / "run0/analysis/1_mdscoring_analysis/capri_ss.tsv"
            capri_ss = pd.read_csv(capri_file, sep="\t")
            metrics = capri_ss[['score', 'total', 'vdw', 'elec', 'desolv', 'bsa']].iloc[0].to_dict()

        except subprocess.CalledProcessError as e:
            logger.error(f"HADDOCK3 Error: {e.stderr}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {}
        
        logger.info("HADDOCK3 completed")
        return metrics
    
    def generate_csv_output(self, ligand_properties: Dict[str, float],
                           docking_metrics: Dict[str, float],
                           output_filename: str = "results.csv") -> str:
        """
        Generate CSV file with ligand chemistry and docking metrics
        
        Args:
            ligand_properties: Dictionary of ligand chemical properties
            docking_metrics: Dictionary of docking metrics
            output_filename: Name for output CSV file
            
        Returns:
            Path to CSV file
        """
        logger.info("Generating CSV output...")
        
        # Combine all results
        all_data = {**ligand_properties, **docking_metrics}
        
        # Create DataFrame
        df = pd.DataFrame([all_data])
        
        # Save to CSV
        csv_path = self.output_dir / output_filename
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to: {csv_path}")
        return str(csv_path)
    
    def run_pipeline(self, protein_sequence: str, smiles: str,
                    singularity_image: str, model_params_dir: str,
                    databases_dir: str, code_dir: str, job_name: str = "docking_job") -> str:
        """
        Run the complete pipeline
        
        Args:
            protein_sequence: Amino acid sequence
            smiles: SMILES string for ligand
            singularity_image: Path to AlphaFold3 singularity image
            model_params_dir: Path to AF3 model parameters
            databases_dir: Path to sequence databases
            code_dir: Path to directory containing AF3 code
            job_name: Name for the job
            
        Returns:
            Path to results CSV file
        """
        logger.info(f"Starting pipeline for job: {job_name}")
        
        try:
            # Step 1: Create AlphaFold3 JSON input
            json_path = self.create_af3_json(protein_sequence, smiles, job_name)
            
            # Step 2: Run AlphaFold3
            af3_output = self.run_alphafold3(
                job_name, json_path, singularity_image, model_params_dir, databases_dir, code_dir
            )

            # Step 3: Convert AF3 CIF to PDB
            self.convert_cif_to_pdb(af3_output, job_name)
            
            # Step 4: Extract ligand
            ligand_pdb = self.extract_ligand(af3_output, job_name)
            
            # Step 5: Run ACPYPE for topology
            topology_files = self.run_acpype(ligand_pdb)
            
            # Step 6: Calculate ligand properties with RDKit
            ligand_properties = self.calculate_ligand_properties(smiles)
            
            # Step 7: Get protein structure for HADDOCK3
            protein_pdb = str(Path(af3_output) / job_name / "*.pdb")  # Adjust as needed
            
            # Step 8: Run HADDOCK3 docking
            docking_metrics = self.run_haddock3(job_name, protein_pdb, str(Path(self.haddock_dir)), topology_files)
            
            # Step 9: Generate CSV output
            csv_path = self.generate_csv_output(
                ligand_properties, docking_metrics, f"{job_name}_results.csv"
            )
            
            logger.info(f"Pipeline completed successfully! Results: {csv_path}")
            return csv_path
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="AlphaFold3 to HADDOCK3 protein-ligand docking pipeline"
    )
    parser.add_argument(
        "--protein", "-p",
        required=True,
        help="Amino acid sequence or path to FASTA file"
    )
    parser.add_argument(
        "--smiles", "-s",
        required=True,
        help="SMILES string for ligand"
    )
    parser.add_argument(
        "--singularity-image",
        required=True,
        help="Path to AlphaFold3 singularity image (.sif)"
    )
    parser.add_argument(
        "--model-params",
        required=True,
        help="Path to AlphaFold3 model parameters directory"
    )
    parser.add_argument(
        "--databases",
        required=True,
        help="Path to sequence databases directory"
    )
    parser.add_argument(
        "--code-dir",
        required=True,
        help="Path to directory containing AlphaFold3 code (run_alphafold.py)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./pipeline_output",
        help="Output directory (default: ./pipeline_output)"
    )
    parser.add_argument(
        "--job-name", "-n",
        default="docking_job",
        help="Job name (default: docking_job)"
    )
    
    args = parser.parse_args()
    
    # Check if protein is a file or sequence
    if os.path.isfile(args.protein):
        with open(args.protein, 'r') as f:
            # Simple FASTA parser
            protein_sequence = ''.join(line.strip() for line in f if not line.startswith('>'))
    else:
        protein_sequence = args.protein
    
    # Initialize and run pipeline
    pipeline = AF3HaddockPipeline(output_dir=args.output_dir)
    
    csv_path = pipeline.run_pipeline(
        protein_sequence=protein_sequence,
        smiles=args.smiles,
        singularity_image=args.singularity_image,
        model_params_dir=args.model_params,
        databases_dir=args.databases,
        code_dir=args.code_dir,
        job_name=args.job_name
    )
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed successfully!")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()