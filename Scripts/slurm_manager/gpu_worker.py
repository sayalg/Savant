#!/usr/bin/env python3
"""
GPU Worker Script - Processes compounds sequentially from a queue

Each worker:
- Runs on 1 GPU for up to 10 days
- Pulls compounds from database work queue
- Processes them one at a time
- Continues even if individual jobs fail
- Updates status in database
"""

import sqlite3
import subprocess
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Worker %(worker_id)s - %(levelname)s - %(message)s'
)


class GPUWorker:
    """Worker that processes compounds from a queue on a single GPU"""
    
    def __init__(self, db_path: str, worker_id: int, pipeline_script: str,
                 singularity_image: str, model_params: str, databases: str,
                 code_dir: str, output_base_dir: str, 
                 start_from: str = "mmseqs2", max_msa_seqs: int = 512):
        """
        Initialize GPU worker
        
        Args:
            db_path: Path to SQLite database
            worker_id: Unique worker ID (0, 1, 2, 3, 4 for 5 workers)
            pipeline_script: Path to pipeline.py
            singularity_image: Path to AF3 singularity image
            model_params: Path to model parameters
            databases: Path to databases
            code_dir: Path to AF3 code directory
            output_base_dir: Base output directory
            start_from: Pipeline start step
            max_msa_seqs: Maximum MSA sequences
        """
        self.db_path = db_path
        self.worker_id = worker_id
        self.pipeline_script = pipeline_script
        self.singularity_image = singularity_image
        self.model_params = model_params
        self.databases = databases
        self.code_dir = code_dir
        self.output_base_dir = Path(output_base_dir)
        self.start_from = start_from
        self.max_msa_seqs = max_msa_seqs
        
        self.logger = logging.getLogger(f'Worker-{worker_id}')
        self.conn = None
        self.processed = 0
        self.succeeded = 0
        self.failed = 0
    
    def connect_db(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        # Enable WAL mode for better concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")
    
    def get_next_job(self):
        """
        Get next pending job from database atomically
        
        Returns:
            Job dict or None if no jobs available
        """
        cursor = self.conn.cursor()
        
        # Use transaction to atomically claim a job
        try:
            self.conn.execute("BEGIN EXCLUSIVE")
            
            # Get one pending job
            cursor.execute("""
                SELECT id, job_name, protein_sequence, ligand_smiles 
                FROM jobs 
                WHERE status='pending'
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            
            if row:
                job_id, job_name, protein_seq, smiles = row
                
                # Mark as claimed by this worker
                cursor.execute("""
                    UPDATE jobs 
                    SET status='running', 
                        worker_id=?, 
                        start_time=?,
                        attempts=attempts+1
                    WHERE id=?
                """, (self.worker_id, datetime.now(), job_id))
                
                self.conn.commit()
                
                return {
                    'id': job_id,
                    'job_name': job_name,
                    'protein_sequence': protein_seq,
                    'ligand_smiles': smiles
                }
            else:
                self.conn.rollback()
                return None
                
        except sqlite3.OperationalError as e:
            self.logger.warning(f"Database locked, retrying: {e}")
            self.conn.rollback()
            time.sleep(1)
            return None
    
    def process_job(self, job):
        """
        Process a single docking job
        
        Args:
            job: Job dictionary
            
        Returns:
            True if successful, False if failed
        """
        job_name = job['job_name']
        self.logger.info(f"Processing job: {job_name}")
        
        # Create output directory
        job_output_dir = self.output_base_dir / job_name
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            'python3', self.pipeline_script,
            '-p', job['protein_sequence'],
            '-s', job['ligand_smiles'],
            '--singularity-image', self.singularity_image,
            '--model-params', self.model_params,
            '--databases', self.databases,
            '--code-dir', self.code_dir,
            '--output-dir', str(job_output_dir),
            '-n', job_name,
            '--start-from', self.start_from,
            '--max-msa-seqs', str(self.max_msa_seqs)
        ]
        
        # Run pipeline
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per job
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                # Check if output file exists
                expected_csv = job_output_dir / "csv" / f"{job_name}_results.csv"
                
                if expected_csv.exists():
                    self.logger.info(f"Job {job_name} completed successfully in {elapsed:.1f}s")
                    self._update_job_status(job['id'], 'completed', str(expected_csv))
                    return True
                else:
                    self.logger.warning(f"Job {job_name} finished but no output CSV")
                    self._update_job_status(job['id'], 'failed', None, 
                                           'No output file generated')
                    return False
            else:
                self.logger.error(f"Job {job_name} failed with code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr[:500]}")
                self._update_job_status(job['id'], 'failed', None, 
                                       f"Exit code {result.returncode}: {result.stderr[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Job {job_name} timed out after 30 minutes")
            self._update_job_status(job['id'], 'failed', None, 'Timeout after 30 minutes')
            return False
            
        except Exception as e:
            self.logger.error(f"Job {job_name} failed with exception: {e}")
            self._update_job_status(job['id'], 'failed', None, str(e))
            return False
    
    def _update_job_status(self, job_id, status, output_csv=None, error_msg=None):
        """Update job status in database"""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE jobs 
                SET status=?, 
                    output_csv=?, 
                    error_message=?,
                    completion_time=?
                WHERE id=?
            """, (status, output_csv, error_msg, datetime.now(), job_id))
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update job status: {e}")
            self.conn.rollback()
    
    def run(self):
        """Main worker loop - processes jobs until queue is empty"""
        self.logger.info("Worker started")
        self.logger.info(f"Pipeline: {self.pipeline_script}")
        self.logger.info(f"Output dir: {self.output_base_dir}")
        
        self.connect_db()
        
        try:
            while True:
                # Get next job
                job = self.get_next_job()
                
                if job is None:
                    # No jobs available
                    self.logger.info("No more jobs in queue, checking again in 30s...")
                    time.sleep(30)
                    
                    # Double-check there are really no jobs
                    job = self.get_next_job()
                    if job is None:
                        self.logger.info("Queue is empty, worker shutting down")
                        break
                
                # Process the job
                success = self.process_job(job)
                
                self.processed += 1
                if success:
                    self.succeeded += 1
                else:
                    self.failed += 1
                
                # Log progress
                if self.processed % 10 == 0:
                    self.logger.info(f"Progress: {self.processed} processed, "
                                   f"{self.succeeded} succeeded, {self.failed} failed")
        
        finally:
            if self.conn:
                self.conn.close()
            
            self.logger.info("Worker finished")
            self.logger.info(f"Total processed: {self.processed}")
            self.logger.info(f"Succeeded: {self.succeeded}")
            self.logger.info(f"Failed: {self.failed}")


def main():
    parser = argparse.ArgumentParser(description="GPU Worker for batch docking")
    parser.add_argument('--db-path', required=True, help='Path to SQLite database')
    parser.add_argument('--worker-id', type=int, required=True, help='Worker ID (0-4 for 5 workers)')
    parser.add_argument('--pipeline-script', required=True, help='Path to pipeline.py')
    parser.add_argument('--singularity-image', required=True, help='Path to AF3 singularity image')
    parser.add_argument('--model-params', required=True, help='Path to model parameters')
    parser.add_argument('--databases', required=True, help='Path to databases')
    parser.add_argument('--code-dir', required=True, help='Path to AF3 code directory')
    parser.add_argument('--output-dir', required=True, help='Base output directory')
    parser.add_argument('--start-from', default='mmseqs2', help='Pipeline start step')
    parser.add_argument('--max-msa-seqs', type=int, default=512, help='Max MSA sequences')
    
    args = parser.parse_args()
    
    worker = GPUWorker(
        db_path=args.db_path,
        worker_id=args.worker_id,
        pipeline_script=args.pipeline_script,
        singularity_image=args.singularity_image,
        model_params=args.model_params,
        databases=args.databases,
        code_dir=args.code_dir,
        output_base_dir=args.output_dir,
        start_from=args.start_from,
        max_msa_seqs=args.max_msa_seqs
    )
    
    worker.run()


if __name__ == "__main__":
    main()
