#!/usr/bin/env python3
"""
Master Script for Batch AlphaFold3-HADDOCK3 Docking

Reads a CSV file with protein sequences and ligands, submits SLURM jobs,
tracks completion status, and manages concurrent GPU job limits.

CSV Format:
    job_name,protein_sequence,ligand_smiles
    compound1,MKTAY...,CC(=O)O
    compound2,MFGLA...,C1=CC=CC=C1

Directory Structure:
    parent_dir/                  (run master script from here)
    ├── input.csv               (your input CSV)
    ├── scripts/                (scripts directory)
    │   ├── pipeline.py
    │   ├── submit_job.sh
    │   └── run_mmseqs2.py
    └── results/                (output - created automatically)
        ├── compound1/
        ├── compound2/
        └── ...
"""

import pandas as pd
import sqlite3
import subprocess
import argparse
import time
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import getpass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_script.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SlurmJobManager:
    """Manages SLURM job submission and tracking for docking pipeline"""
    
    def __init__(self, db_path: str = "job_tracking.db", max_gpu_jobs: int = 20):
        """
        Initialize the SLURM job manager
        
        Args:
            db_path: Path to SQLite database for job tracking
            max_gpu_jobs: Maximum number of concurrent GPU jobs (default: 20)
        """
        self.db_path = db_path
        self.max_gpu_jobs = max_gpu_jobs
        self.conn = None
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for job tracking"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_name TEXT UNIQUE NOT NULL,
                protein_sequence TEXT NOT NULL,
                ligand_smiles TEXT NOT NULL,
                slurm_job_id TEXT,
                status TEXT DEFAULT 'pending',
                submission_time TIMESTAMP,
                completion_time TIMESTAMP,
                output_csv TEXT,
                error_message TEXT,
                attempts INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_name ON jobs(job_name)
        """)
        
        self.conn.commit()
        logger.info(f"Database initialized: {self.db_path}")
    
    def load_jobs_from_csv(self, csv_path: str, overwrite: bool = False):
        """
        Load jobs from CSV file into database
        
        Args:
            csv_path: Path to CSV file with job_name, protein_sequence, ligand_smiles columns
            overwrite: If True, reset all jobs; if False, skip existing
        """
        logger.info(f"Loading jobs from {csv_path}")
        
        df = pd.read_csv(csv_path)
        required_cols = ['job_name', 'protein_sequence', 'ligand_smiles']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        cursor = self.conn.cursor()
        
        if overwrite:
            logger.warning("Overwrite mode: Resetting all existing jobs")
            cursor.execute("DELETE FROM jobs")
            self.conn.commit()
        
        added = 0
        skipped = 0
        
        for _, row in df.iterrows():
            job_name = row['job_name']
            protein_seq = row['protein_sequence']
            smiles = row['ligand_smiles']
            
            try:
                cursor.execute("""
                    INSERT INTO jobs (job_name, protein_sequence, ligand_smiles, status)
                    VALUES (?, ?, ?, 'pending')
                """, (job_name, protein_seq, smiles))
                added += 1
            except sqlite3.IntegrityError:
                # Job already exists
                if overwrite:
                    cursor.execute("""
                        UPDATE jobs 
                        SET protein_sequence=?, ligand_smiles=?, status='pending', 
                            slurm_job_id=NULL, error_message=NULL
                        WHERE job_name=?
                    """, (protein_seq, smiles, job_name))
                    added += 1
                else:
                    skipped += 1
        
        self.conn.commit()
        logger.info(f"Jobs loaded: {added} added, {skipped} skipped (already exist)")
        
        return added, skipped
    
    def get_active_gpu_jobs(self, user: Optional[str] = None) -> int:
        """
        Get count of active GPU jobs (pending + running)
        
        Args:
            user: Username to filter by (None = current user)
            
        Returns:
            Number of active GPU jobs
        """
        cmd = ['squeue', '--noheader', '--format=%i %b']
        
        if user:
            cmd.extend(['-u', user])
        else:
            cmd.extend(['-u', getpass.getuser()])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Count jobs requesting GPUs (looking for "gpu:" in tres-per-node)
            gpu_jobs = 0
            for line in result.stdout.strip().split('\n'):
                if line and 'gpu' in line.lower():
                    gpu_jobs += 1
            
            # Fallback: count all jobs if we can't detect GPU specifically
            if gpu_jobs == 0 and result.stdout.strip():
                gpu_jobs = len(result.stdout.strip().split('\n'))
                
            return gpu_jobs
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get SLURM job count: {e}")
            return 0
    
    def get_pending_jobs(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get pending jobs from database
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of job dictionaries
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM jobs WHERE status='pending' ORDER BY id"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        
        jobs = []
        for row in cursor.fetchall():
            jobs.append(dict(zip(columns, row)))
        
        return jobs
    
    def get_job_statistics(self) -> Dict[str, int]:
        """Get statistics on job statuses"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM jobs 
            GROUP BY status
        """)
        
        stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("SELECT COUNT(*) FROM jobs")
        stats['total'] = cursor.fetchone()[0]
        
        return stats
    
    def submit_slurm_job(self, job: Dict, slurm_script_path: str,
                         singularity_image: str, model_params: str,
                         databases: str, code_dir: str,
                         output_base_dir: str, start_from: str = "mmseqs2",
                         max_msa_seqs: int = 512,
                         pipeline_script_path: str = "scripts/pipeline.py") -> Optional[str]:
        """
        Submit a SLURM job for docking
        
        Args:
            job: Job dictionary with job_name, protein_sequence, ligand_smiles
            slurm_script_path: Path to SLURM submission script
            singularity_image: Path to AlphaFold3 singularity image
            model_params: Path to model parameters
            databases: Path to databases
            code_dir: Path to AlphaFold3 code directory
            output_base_dir: Base directory for outputs
            start_from: Pipeline start step (default: mmseqs2)
            max_msa_seqs: Maximum MSA sequences (default: 512)
            pipeline_script_path: Path to pipeline.py script
            
        Returns:
            SLURM job ID or None if failed
        """
        job_name = job['job_name']
        job_output_dir = Path(output_base_dir) / job_name
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get absolute path to pipeline script
        pipeline_abs_path = Path(pipeline_script_path).resolve()
        
        if not pipeline_abs_path.exists():
            logger.error(f"Pipeline script not found: {pipeline_abs_path}")
            return None
        
        cmd = [
            'sbatch',
            f'--job-name={job_name}',
            f'--output={job_output_dir}/slurm_%j.out',
            f'--error={job_output_dir}/slurm_%j.err',
            slurm_script_path,
            str(pipeline_abs_path),    # FIRST: pipeline script path
            job['protein_sequence'],   # protein sequence directly
            job['ligand_smiles'],      # SMILES
            singularity_image,
            model_params,
            databases,
            code_dir,
            str(job_output_dir),       # output directory
            job_name,                  # job name
            start_from,                # start from step
            str(max_msa_seqs)          # max MSA sequences
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse job ID from sbatch output: "Submitted batch job 12345"
            slurm_job_id = result.stdout.strip().split()[-1]
            
            # Update database
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE jobs 
                SET slurm_job_id=?, status='submitted', submission_time=?, attempts=attempts+1
                WHERE job_name=?
            """, (slurm_job_id, datetime.now(), job_name))
            self.conn.commit()
            
            logger.info(f"Submitted job {job_name} (SLURM ID: {slurm_job_id})")
            return slurm_job_id
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job {job_name}: {e}")
            logger.error(f"STDERR: {e.stderr}")
            
            # Update database with error
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE jobs 
                SET status='failed', error_message=?, attempts=attempts+1
                WHERE job_name=?
            """, (str(e), job_name))
            self.conn.commit()
            
            return None
    
    def check_job_completion(self, job_name: str, expected_output: str) -> bool:
        """
        Check if a job has completed successfully
        
        Args:
            job_name: Name of the job
            expected_output: Path to expected output CSV
            
        Returns:
            True if completed successfully
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT status, slurm_job_id FROM jobs WHERE job_name=?", (job_name,))
        result = cursor.fetchone()
        
        if not result:
            return False
        
        status, slurm_job_id = result
        
        # Check if output file exists
        if Path(expected_output).exists():
            cursor.execute("""
                UPDATE jobs 
                SET status='completed', completion_time=?, output_csv=?
                WHERE job_name=?
            """, (datetime.now(), expected_output, job_name))
            self.conn.commit()
            logger.info(f"Job {job_name} completed successfully")
            return True
        
        # Check if SLURM job is still running
        if slurm_job_id and status in ['submitted', 'running']:
            try:
                result = subprocess.run(
                    ['squeue', '-j', slurm_job_id, '--noheader'],
                    capture_output=True, text=True, check=True
                )
                
                if result.stdout.strip():
                    # Job is still in queue
                    cursor.execute("""
                        UPDATE jobs SET status='running' WHERE job_name=?
                    """, (job_name,))
                    self.conn.commit()
                    return False
                else:
                    # Job finished but no output - likely failed
                    cursor.execute("""
                        UPDATE jobs 
                        SET status='failed', error_message='Job finished without output'
                        WHERE job_name=?
                    """, (job_name,))
                    self.conn.commit()
                    logger.warning(f"Job {job_name} finished without output")
                    return False
                    
            except subprocess.CalledProcessError:
                # Job not in queue anymore
                pass
        
        return False
    
    def update_all_job_statuses(self, output_base_dir: str):
        """Update status of all submitted/running jobs"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT job_name FROM jobs WHERE status IN ('submitted', 'running')
        """)
        
        jobs_to_check = [row[0] for row in cursor.fetchall()]
        
        for job_name in jobs_to_check:
            expected_output = Path(output_base_dir) / job_name / "csv" / f"{job_name}_results.csv"
            self.check_job_completion(job_name, str(expected_output))
    
    def wait_for_gpu_slots(self, check_interval: int = 60):
        """
        Wait until there are available GPU slots
        
        Args:
            check_interval: Seconds between checks
        """
        while True:
            active_jobs = self.get_active_gpu_jobs()
            
            if active_jobs < self.max_gpu_jobs:
                slots_available = self.max_gpu_jobs - active_jobs
                logger.info(f"GPU slots available: {slots_available}")
                return slots_available
            
            logger.info(f"At max GPU jobs ({active_jobs}/{self.max_gpu_jobs}). "
                       f"Waiting {check_interval}s...")
            time.sleep(check_interval)
    
    def run_batch_submission(self, slurm_script_path: str, singularity_image: str,
                            model_params: str, databases: str, code_dir: str,
                            output_base_dir: str, batch_size: int = 10,
                            check_interval: int = 60, start_from: str = "mmseqs2",
                            max_msa_seqs: int = 512, pipeline_script_path: str = "scripts/pipeline.py"):
        """
        Run batch job submission with monitoring
        
        Args:
            slurm_script_path: Path to SLURM script
            singularity_image: Path to AF3 singularity image
            model_params: Path to model parameters
            databases: Path to databases
            code_dir: Path to AlphaFold3 code directory
            output_base_dir: Base output directory
            batch_size: Number of jobs to submit per batch
            check_interval: Seconds between status checks
            start_from: Pipeline start step
            max_msa_seqs: Maximum MSA sequences
            pipeline_script_path: Path to pipeline.py script
        """
        logger.info("Starting batch job submission")
        
        while True:
            # Update status of running jobs
            self.update_all_job_statuses(output_base_dir)
            
            # Get statistics
            stats = self.get_job_statistics()
            logger.info(f"Job statistics: {stats}")
            
            # Get pending jobs
            pending_jobs = self.get_pending_jobs()
            
            if not pending_jobs:
                logger.info("No more pending jobs")
                break
            
            # Check how many jobs we can submit
            slots_available = self.wait_for_gpu_slots(check_interval)
            jobs_to_submit = min(batch_size, slots_available, len(pending_jobs))
            
            logger.info(f"Submitting {jobs_to_submit} jobs...")
            
            # Submit jobs
            for i, job in enumerate(pending_jobs[:jobs_to_submit]):
                logger.info(f"Submitting job {i+1}/{jobs_to_submit}: {job['job_name']}")
                
                self.submit_slurm_job(
                    job,
                    slurm_script_path,
                    singularity_image,
                    model_params,
                    databases,
                    code_dir,
                    output_base_dir,
                    start_from=start_from,
                    max_msa_seqs=max_msa_seqs,
                    pipeline_script_path=pipeline_script_path
                )
                
                # Small delay to avoid overwhelming scheduler
                time.sleep(1)
            
            # Wait before next batch
            logger.info(f"Waiting {check_interval}s before next status check...")
            time.sleep(check_interval)
        
        # Final status update
        logger.info("All jobs submitted. Waiting for completion...")
        self.monitor_remaining_jobs(output_base_dir, check_interval)
    
    def monitor_remaining_jobs(self, output_base_dir: str, check_interval: int = 60):
        """Monitor remaining jobs until all complete"""
        while True:
            self.update_all_job_statuses(output_base_dir)
            stats = self.get_job_statistics()
            
            active = stats.get('submitted', 0) + stats.get('running', 0)
            
            if active == 0:
                logger.info("All jobs completed!")
                logger.info(f"Final statistics: {stats}")
                break
            
            logger.info(f"Active jobs: {active}. Statistics: {stats}")
            time.sleep(check_interval)
    
    def export_results(self, output_csv: str):
        """Export job tracking database to CSV"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM jobs")
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(output_csv, index=False)
        logger.info(f"Job tracking data exported to: {output_csv}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main entry point"""
    
    # Get defaults from environment variables (set by config.sh)
    import os
    env_defaults = {
        'singularity_image': os.getenv('SINGULARITY_IMAGE'),
        'model_params': os.getenv('MODEL_PARAMS'),
        'databases': os.getenv('DATABASES'),
        'code_dir': os.getenv('CODE_DIR'),
        'output_dir': os.getenv('OUTPUT_DIR', './results'),
        'db_path': os.getenv('DB_PATH', 'job_tracking.db'),
        'max_gpu_jobs': int(os.getenv('MAX_GPU_JOBS', '20')),
        'batch_size': int(os.getenv('BATCH_SIZE', '10')),
        'check_interval': int(os.getenv('CHECK_INTERVAL', '300')),
        'start_from': os.getenv('START_FROM', 'mmseqs2'),
        'max_msa_seqs': int(os.getenv('MAX_MSA_SEQS', '512')),
        'slurm_script': os.getenv('SLURM_SCRIPT', 'scripts/submit_job.sh'),
    }
    
    parser = argparse.ArgumentParser(
        description="Master script for batch AlphaFold3-HADDOCK3 docking",
        epilog="TIP: Set environment variables in config.sh to avoid passing paths every time. "
               "Command-line arguments override environment variables."
    )
    parser.add_argument(
        '--input-csv', '-i',
        required=True,
        help='Input CSV with job_name, protein_sequence, ligand_smiles columns'
    )
    parser.add_argument(
        '--slurm-script',
        default=env_defaults['slurm_script'],
        help=f'Path to SLURM submission script (default: {env_defaults["slurm_script"]} or $SLURM_SCRIPT)'
    )
    parser.add_argument(
        '--pipeline-script',
        default=os.getenv('PIPELINE_SCRIPT', 'scripts/pipeline.py'),
        help='Path to pipeline.py script (default: scripts/pipeline.py or $PIPELINE_SCRIPT)'
    )
    parser.add_argument(
        '--singularity-image',
        default=env_defaults['singularity_image'],
        required=env_defaults['singularity_image'] is None,
        help='Path to AlphaFold3 singularity image (or set $SINGULARITY_IMAGE)'
    )
    parser.add_argument(
        '--model-params',
        default=env_defaults['model_params'],
        required=env_defaults['model_params'] is None,
        help='Path to AlphaFold3 model parameters (or set $MODEL_PARAMS)'
    )
    parser.add_argument(
        '--databases',
        default=env_defaults['databases'],
        required=env_defaults['databases'] is None,
        help='Path to sequence databases (or set $DATABASES)'
    )
    parser.add_argument(
        '--code-dir',
        default=env_defaults['code_dir'],
        required=env_defaults['code_dir'] is None,
        help='Path to AlphaFold3 code directory (or set $CODE_DIR)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=env_defaults['output_dir'],
        help=f'Base output directory (default: {env_defaults["output_dir"]} or $OUTPUT_DIR)'
    )
    parser.add_argument(
        '--db-path',
        default=env_defaults['db_path'],
        help=f'Path to job tracking database (default: {env_defaults["db_path"]} or $DB_PATH)'
    )
    parser.add_argument(
        '--max-gpu-jobs',
        type=int,
        default=env_defaults['max_gpu_jobs'],
        help=f'Maximum concurrent GPU jobs (default: {env_defaults["max_gpu_jobs"]} or $MAX_GPU_JOBS)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=env_defaults['batch_size'],
        help=f'Number of jobs to submit per batch (default: {env_defaults["batch_size"]} or $BATCH_SIZE)'
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=env_defaults['check_interval'],
        help=f'Seconds between status checks (default: {env_defaults["check_interval"]} or $CHECK_INTERVAL)'
    )
    parser.add_argument(
        '--start-from',
        choices=['mmseqs2', 'alphafold3', 'convert', 'extract', 'acpype', 'haddock3'],
        default=env_defaults['start_from'],
        help=f'Pipeline start step (default: {env_defaults["start_from"]} or $START_FROM)'
    )
    parser.add_argument(
        '--max-msa-seqs',
        type=int,
        default=env_defaults['max_msa_seqs'],
        help=f'Maximum MSA sequences (default: {env_defaults["max_msa_seqs"]} or $MAX_MSA_SEQS)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing jobs in database'
    )
    parser.add_argument(
        '--export-results',
        help='Export job tracking to CSV file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing database (skip CSV loading)'
    )
    
    args = parser.parse_args()
    
    # Log which settings came from environment
    if env_defaults['singularity_image']:
        logger.info(f"Using SINGULARITY_IMAGE from environment: {args.singularity_image}")
    if env_defaults['model_params']:
        logger.info(f"Using MODEL_PARAMS from environment: {args.model_params}")
    if env_defaults['databases']:
        logger.info(f"Using DATABASES from environment: {args.databases}")
    if env_defaults['code_dir']:
        logger.info(f"Using CODE_DIR from environment: {args.code_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize job manager
    manager = SlurmJobManager(
        db_path=args.db_path,
        max_gpu_jobs=args.max_gpu_jobs
    )
    
    try:
        # Load jobs from CSV unless resuming
        if not args.resume:
            manager.load_jobs_from_csv(args.input_csv, overwrite=args.overwrite)
        else:
            logger.info("Resuming from existing database")
        
        # Run batch submission
        manager.run_batch_submission(
            slurm_script_path=args.slurm_script,
            singularity_image=args.singularity_image,
            model_params=args.model_params,
            databases=args.databases,
            code_dir=args.code_dir,
            output_base_dir=args.output_dir,
            batch_size=args.batch_size,
            check_interval=args.check_interval,
            start_from=args.start_from,
            max_msa_seqs=args.max_msa_seqs,
            pipeline_script_path=args.pipeline_script
        )
        
        # Export results if requested
        if args.export_results:
            manager.export_results(args.export_results)
        
        # Print final summary
        stats = manager.get_job_statistics()
        print("\n" + "="*60)
        print("FINAL JOB STATISTICS")
        print("="*60)
        for status, count in stats.items():
            print(f"{status}: {count}")
        print("="*60)
        
    finally:
        manager.close()


if __name__ == "__main__":
    main()
