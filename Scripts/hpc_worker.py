#!/usr/bin/env python3
import os
import sys
import time
import random
import argparse
import traceback
import pandas as pd
from pathlib import Path
import logging

# Import your pipeline class
from pipeline import AF3HaddockPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_failure(log_file, job_name, stage, error_msg):
    """Appends failed jobs to a CSV log so they can be easily tracked and re-run."""
    with open(log_file, "a") as f:
        # Replace newlines in error_msg to keep CSV format clean
        clean_error = str(error_msg).replace('\n', ' | ').replace('\r', '')
        f.write(f"{job_name},{stage},{clean_error}\n")

def main():
    parser = argparse.ArgumentParser(description="HPC Batch Worker for AF3-HADDOCK Pipeline")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--chunk-size", type=int, required=True, help="Rows per Slurm task")
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--stage", choices=["msa", "af3", "docking"], required=True, help="Which stage to run")
    
    # HPC specific paths
    parser.add_argument("--singularity-image", default="/scratch/sguirale/AlphaFold3/image/alphafold3.sif")
    parser.add_argument("--model-params", default="/projects/datasets/alphafold/3.0.0/weights")
    parser.add_argument("--databases", default="/scratch/sguirale/AlphaFold3/databases")
    parser.add_argument("--code-dir", default="/scratch/sguirale/AlphaFold3/code")
    
    args = parser.parse_args()

    # 1. Setup Slurm Array tracking
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    if not task_id_str:
        logger.error("SLURM_ARRAY_TASK_ID not set. Defaulting to 1 for testing.")
        task_id = 1
    else:
        task_id = int(task_id_str)

    start_idx = (task_id - 1) * args.chunk_size

    # 2. Read the specific chunk
    try:
        df_chunk = pd.read_csv(args.csv, skiprows=range(1, start_idx + 1), nrows=args.chunk_size)
    except pd.errors.EmptyDataError:
        logger.info(f"Task {task_id}: Reached end of file. Exiting cleanly.")
        sys.exit(0)

    if df_chunk.empty:
        logger.info(f"Task {task_id}: No data to process.")
        sys.exit(0)

    # Ensure the output directory exists first!
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 3. Setup Failure Log
    failure_log = Path(args.output_dir) / "failed_jobs_log.csv"
    if not failure_log.exists():
        with open(failure_log, "w") as f:
            f.write("Job_Name,Stage,Error\n")

    # Instantiate the pipeline
    pipeline = AF3HaddockPipeline(output_dir=args.output_dir)

    # 4. Iterate over the chunk and execute the requested stage
    if args.stage == "msa":
        # Create isolated subdirectories for this Slurm array task's chunk input/output
        batch_input_dir = pipeline.msa_dir / f"batch_{task_id}_inputs"
        batch_output_dir = pipeline.msa_dir / f"batch_{task_id}_outputs"
        batch_input_dir.mkdir(exist_ok=True, parents=True)
        batch_output_dir.mkdir(exist_ok=True, parents=True)

        # Step A: Stage all individual query fastas into the batch directory first
        logger.info(f"Task {task_id}: Preparing FASTA inputs for batch processing...")
        for index, row in df_chunk.iterrows():
            job_name = str(row['ID'])
            protein_seq = str(row['seq'])
            
            fasta_file = batch_input_dir / f"{job_name}.fasta"
            fasta_file.write_text(f">{job_name}\n{protein_seq}\n")

        # Step B: Run colabfold_search ONCE for all 1,000 files in the directory
        try:
            pipeline.run_mmseqs2_msa_batch(batch_input_dir, batch_output_dir)
        except Exception as e:
            logger.error(f"Task {task_id}: Batch MSA execution failed entirely. Error: {e}")
            # Log all jobs in chunk as failed at MSA stage
            for index, row in df_chunk.iterrows():
                log_failure(failure_log, str(row['ID']), args.stage, f"Batch failure: {e}")
            sys.exit(1)

        # Step C: Post-process each alignment and build corresponding AF3 JSON inputs
        logger.info(f"Task {task_id}: Post-processing alignments and generating AlphaFold3 input JSONs...")
        for index, row in df_chunk.iterrows():
            job_name = str(row['ID'])
            protein_seq = str(row['seq'])
            smiles = str(row['smiles'])
            
            try:
                # colabfold_search names outputs matching the filename of inputs
                raw_a3m = batch_output_dir / f"{job_name}.a3m" 
                
                # Cap the depth and place it in the pipeline's expected tracking directory
                msa_path = pipeline.process_and_cap_a3m(raw_a3m, job_name)
                
                # Generate individual AF3 json inputs linking this alignment
                pipeline.create_af3_json(protein_seq, smiles, job_name, msa_path=msa_path)
                logger.info(f"Successfully processed MSA and generated AF3 JSON for {job_name}")
                
            except Exception as e:
                logger.error(f"Failed processing {job_name} during alignment capping/JSON stage. Error: {e}")
                log_failure(failure_log, job_name, args.stage, e)
                continue

    else:
        for index, row in df_chunk.iterrows():
            job_name = str(row['ID'])
            protein_seq = str(row['seq'])
            smiles = str(row['smiles'])

            logger.info(f"--- Task {task_id}: Starting {args.stage} for {job_name} ---")

            try:
                if args.stage == "af3":
                    # STAGE 2 (GPU): Run AlphaFold3 Inference
                    json_path = str(pipeline.af3_dir / f"{job_name}_input.json")
                    if not os.path.exists(json_path):
                        raise FileNotFoundError(f"Missing JSON input for {job_name}. Did MSA step fail?")
                    
                    pipeline.run_alphafold3(
                        job_name=job_name,
                        json_input_path=json_path,
                        singularity_image=args.singularity_image,
                        model_params_dir=args.model_params,
                        databases_dir=args.databases,
                        code_dir=args.code_dir,
                        skip_data_pipeline=True # Using pre-computed MSA
                    )

                elif args.stage == "docking":
                    # STAGE 3 (CPU): Post-processing and HADDOCK3
                    af3_output = str(pipeline.af3_dir / "predictions" / job_name)
                    
                    pipeline.patch_cif_residue_name(af3_output, job_name)
                    pipeline.convert_cif_to_pdb(af3_output, job_name)
                    protein_pdb, ligand_pdb = pipeline.extract_chains(af3_output, job_name)
                    
                    ligand_properties = pipeline.calculate_ligand_properties(smiles)
                    docking_metrics = pipeline.run_haddock3(
                        job_name, protein_pdb, ligand_pdb, str(pipeline.haddock_dir)
                    )
                    
                    pipeline.generate_csv_output(ligand_properties, docking_metrics, job_name)

            except Exception as e:
                logger.error(f"Failed {job_name} at stage '{args.stage}'. Logging and skipping. Error: {e}")
                log_failure(failure_log, job_name, args.stage, e)
                continue

if __name__ == "__main__":
    main()