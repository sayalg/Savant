#!/usr/bin/env python3
"""
Master Script for GPU Worker-Based Batch Docking

Instead of submitting 1000s of individual jobs, this submits a small number
of long-running GPU workers that process compounds sequentially.

Example:
- 1000 compounds to process
- 5 GPUs available
- Each GPU runs for up to 10 days
- Each GPU processes ~200 compounds sequentially (2-10 min each)
"""

import pandas as pd
import sqlite3
import subprocess
import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('worker_master.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages GPU workers for batch processing"""
    
    def __init__(self, db_path: str = "docking_queue.db"):
        """Initialize worker manager"""
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with worker tracking"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_name TEXT UNIQUE NOT NULL,
                protein_sequence TEXT NOT NULL,
                ligand_smiles TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                worker_id INTEGER,
                start_time TIMESTAMP,
                completion_time TIMESTAMP,
                output_csv TEXT,
                error_message TEXT,
                attempts INTEGER DEFAULT 0
            )
        """)
        
        # Workers table  
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workers (
                worker_id INTEGER PRIMARY KEY,
                slurm_job_id TEXT,
                status TEXT DEFAULT 'pending',
                submission_time TIMESTAMP,
                node TEXT
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_status ON jobs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_worker_id ON jobs(worker_id)")
        
        self.conn.commit()
        logger.info(f"Database initialized: {self.db_path}")
    
    def load_jobs_from_csv(self, csv_path: str, overwrite: bool = False,
                           max_rows: int = None, skip_rows: int = 0):
        """
        Load jobs from CSV into database
        
        Args:
            csv_path: Path to CSV file
            overwrite: If True, clear existing jobs first
            max_rows: Maximum number of rows to load (None = all)
            skip_rows: Number of rows to skip from start (for chunking)
        """
        logger.info(f"Loading jobs from {csv_path}")
        
        # Read CSV with optional row limiting
        read_kwargs = {}
        if skip_rows > 0:
            read_kwargs['skiprows'] = range(1, skip_rows + 1)  # +1 to skip header
        if max_rows:
            read_kwargs['nrows'] = max_rows
        
        df = pd.read_csv(csv_path, **read_kwargs)
        
        required_cols = ['job_name', 'protein_sequence', 'ligand_smiles']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain: {required_cols}")
        
        logger.info(f"CSV has {len(df)} rows to process "
                   f"(skip_rows={skip_rows}, max_rows={max_rows if max_rows else 'all'})")
        
        cursor = self.conn.cursor()
        
        if overwrite:
            logger.warning("Overwrite mode: Clearing existing jobs")
            cursor.execute("DELETE FROM jobs")
            self.conn.commit()
        
        added = 0
        skipped = 0
        
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO jobs (job_name, protein_sequence, ligand_smiles, status)
                    VALUES (?, ?, ?, 'pending')
                """, (row['job_name'], row['protein_sequence'], row['ligand_smiles']))
                added += 1
            except sqlite3.IntegrityError:
                skipped += 1
        
        self.conn.commit()
        logger.info(f"Loaded: {added} new jobs, {skipped} already exist")
        
        return added, skipped
    
    def get_statistics(self):
        """Get job statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
        stats = dict(cursor.fetchall())
        
        cursor.execute("SELECT COUNT(*) FROM jobs")
        stats['total'] = cursor.fetchone()[0]
        
        # Worker stats
        cursor.execute("SELECT status, COUNT(*) FROM workers GROUP BY status")
        worker_stats = dict(cursor.fetchall())
        
        return {'jobs': stats, 'workers': worker_stats}
    
    def submit_workers(self, num_workers: int, worker_script_path: str,
                       pipeline_script: str, singularity_image: str,
                       model_params: str, databases: str, code_dir: str,
                       output_dir: str, start_from: str = "mmseqs2",
                       max_msa_seqs: int = 512):
        """
        Submit GPU worker jobs
        
        Args:
            num_workers: Number of GPU workers to submit (e.g., 5)
            worker_script_path: Path to submit_worker.sh
            pipeline_script: Path to pipeline.py
            singularity_image: Path to AF3 singularity image
            model_params: Path to model parameters
            databases: Path to databases
            code_dir: Path to AF3 code directory
            output_dir: Base output directory
            start_from: Pipeline start step
            max_msa_seqs: Max MSA sequences
        """
        logger.info(f"Submitting {num_workers} GPU workers...")
        
        # Get absolute paths
        db_abs_path = Path(self.db_path).resolve()
        pipeline_abs_path = Path(pipeline_script).resolve()
        
        if not pipeline_abs_path.exists():
            raise FileNotFoundError(f"Pipeline script not found: {pipeline_abs_path}")
        
        submitted = 0
        
        for worker_id in range(num_workers):
            logger.info(f"Submitting worker {worker_id}...")
            
            cmd = [
                'sbatch',
                f'--job-name=worker_{worker_id}',
                f'--output={output_dir}/worker_{worker_id}_%j.out',
                f'--error={output_dir}/worker_{worker_id}_%j.err',
                worker_script_path,
                str(worker_id),
                str(db_abs_path),
                str(pipeline_abs_path),
                singularity_image,
                model_params,
                databases,
                code_dir,
                output_dir,
                start_from,
                str(max_msa_seqs)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                slurm_job_id = result.stdout.strip().split()[-1]
                
                # Record worker submission
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO workers (worker_id, slurm_job_id, status, submission_time)
                    VALUES (?, ?, 'running', ?)
                """, (worker_id, slurm_job_id, datetime.now()))
                self.conn.commit()
                
                logger.info(f"Worker {worker_id} submitted (SLURM ID: {slurm_job_id})")
                submitted += 1
                
                time.sleep(1)  # Small delay between submissions
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to submit worker {worker_id}: {e}")
                logger.error(f"STDERR: {e.stderr}")
        
        logger.info(f"Submitted {submitted}/{num_workers} workers")
        return submitted
    
    def monitor_progress(self, interval: int = 60):
        """
        Monitor progress and display statistics
        
        Args:
            interval: Seconds between status updates
        """
        logger.info("Monitoring progress (Ctrl+C to stop)...")
        
        try:
            while True:
                stats = self.get_statistics()
                
                logger.info("="*60)
                logger.info("JOB STATUS:")
                for status, count in sorted(stats['jobs'].items()):
                    logger.info(f"  {status}: {count}")
                
                if stats['workers']:
                    logger.info("WORKERS:")
                    for status, count in sorted(stats['workers'].items()):
                        logger.info(f"  {status}: {count}")
                
                # Check if all jobs are done
                pending = stats['jobs'].get('pending', 0)
                running = stats['jobs'].get('running', 0)
                
                if pending == 0 and running == 0:
                    logger.info("All jobs completed!")
                    break
                
                logger.info("="*60)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped")
    
    def export_results(self, output_file: str):
        """Export job results to CSV"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM jobs")
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(output_file, index=False)
        logger.info(f"Results exported to: {output_file}")
    
    def reset_failed_jobs(self):
        """Reset failed jobs to pending for retry"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE status='failed'")
        count = cursor.fetchone()[0]
        
        if count == 0:
            logger.info("No failed jobs to reset")
            return
        
        cursor.execute("""
            UPDATE jobs 
            SET status='pending', worker_id=NULL, error_message=NULL
            WHERE status='failed'
        """)
        self.conn.commit()
        logger.info(f"Reset {count} failed jobs to pending")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    # Get defaults from environment
    env_defaults = {
        'singularity_image': os.getenv('SINGULARITY_IMAGE'),
        'model_params': os.getenv('MODEL_PARAMS'),
        'databases': os.getenv('DATABASES'),
        'code_dir': os.getenv('CODE_DIR'),
        'output_dir': os.getenv('OUTPUT_DIR', './results'),
        'pipeline_script': os.getenv('PIPELINE_SCRIPT', 'scripts/pipeline.py'),
        'worker_script': os.getenv('WORKER_SCRIPT', 'scripts/submit_worker.sh'),
        'num_workers': int(os.getenv('NUM_WORKERS', '5')),
        'start_from': os.getenv('START_FROM', 'mmseqs2'),
        'max_msa_seqs': int(os.getenv('MAX_MSA_SEQS', '512')),
    }
    
    parser = argparse.ArgumentParser(
        description="GPU Worker-Based Batch Docking System",
        epilog="Set environment variables in config.sh for defaults"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load jobs from CSV')
    load_parser.add_argument('--input-csv', '-i', required=True, help='Input CSV file')
    load_parser.add_argument('--db-path', default='docking_queue.db', help='Database path')
    load_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing jobs')
    load_parser.add_argument('--max-rows', type=int, help='Maximum rows to load (default: all)')
    load_parser.add_argument('--skip-rows', type=int, default=0, 
                            help='Skip N rows from start (for processing chunks, default: 0)')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit GPU workers')
    submit_parser.add_argument('--db-path', default='docking_queue.db', help='Database path')
    submit_parser.add_argument('--num-workers', type=int, default=env_defaults['num_workers'],
                               help=f'Number of GPU workers (default: {env_defaults["num_workers"]} or $NUM_WORKERS)')
    submit_parser.add_argument('--worker-script', default=env_defaults['worker_script'],
                               help='Path to submit_worker.sh')
    submit_parser.add_argument('--pipeline-script', default=env_defaults['pipeline_script'],
                               help='Path to pipeline.py')
    submit_parser.add_argument('--singularity-image', default=env_defaults['singularity_image'],
                               required=env_defaults['singularity_image'] is None,
                               help='Path to AF3 singularity image')
    submit_parser.add_argument('--model-params', default=env_defaults['model_params'],
                               required=env_defaults['model_params'] is None,
                               help='Path to model parameters')
    submit_parser.add_argument('--databases', default=env_defaults['databases'],
                               required=env_defaults['databases'] is None,
                               help='Path to databases')
    submit_parser.add_argument('--code-dir', default=env_defaults['code_dir'],
                               required=env_defaults['code_dir'] is None,
                               help='Path to AF3 code directory')
    submit_parser.add_argument('--output-dir', '-o', default=env_defaults['output_dir'],
                               help='Output directory')
    submit_parser.add_argument('--start-from', default=env_defaults['start_from'],
                               help='Pipeline start step')
    submit_parser.add_argument('--max-msa-seqs', type=int, default=env_defaults['max_msa_seqs'],
                               help='Max MSA sequences')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor progress')
    monitor_parser.add_argument('--db-path', default='docking_queue.db', help='Database path')
    monitor_parser.add_argument('--interval', type=int, default=60, help='Update interval (seconds)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show current status')
    status_parser.add_argument('--db-path', default='docking_queue.db', help='Database path')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset-failed', help='Reset failed jobs to pending')
    reset_parser.add_argument('--db-path', default='docking_queue.db', help='Database path')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export results to CSV')
    export_parser.add_argument('--db-path', default='docking_queue.db', help='Database path')
    export_parser.add_argument('--output', '-o', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = WorkerManager(db_path=args.db_path if hasattr(args, 'db_path') else 'docking_queue.db')
    
    try:
        if args.command == 'load':
            manager.load_jobs_from_csv(
                args.input_csv, 
                args.overwrite,
                max_rows=args.max_rows,
                skip_rows=args.skip_rows
            )
            stats = manager.get_statistics()
            print(f"\nTotal jobs in database: {stats['jobs']['total']}")
            print(f"Pending: {stats['jobs'].get('pending', 0)}")
        
        elif args.command == 'submit':
            manager.submit_workers(
                num_workers=args.num_workers,
                worker_script_path=args.worker_script,
                pipeline_script=args.pipeline_script,
                singularity_image=args.singularity_image,
                model_params=args.model_params,
                databases=args.databases,
                code_dir=args.code_dir,
                output_dir=args.output_dir,
                start_from=args.start_from,
                max_msa_seqs=args.max_msa_seqs
            )
        
        elif args.command == 'monitor':
            manager.monitor_progress(args.interval)
        
        elif args.command == 'status':
            stats = manager.get_statistics()
            print("\n" + "="*60)
            print("JOB STATUS")
            print("="*60)
            for status, count in sorted(stats['jobs'].items()):
                print(f"{status}: {count}")
            if stats['workers']:
                print("\nWORKERS")
                print("-"*60)
                for status, count in sorted(stats['workers'].items()):
                    print(f"{status}: {count}")
            print("="*60)
        
        elif args.command == 'reset-failed':
            manager.reset_failed_jobs()
        
        elif args.command == 'export':
            manager.export_results(args.output)
    
    finally:
        manager.close()


if __name__ == "__main__":
    main()
