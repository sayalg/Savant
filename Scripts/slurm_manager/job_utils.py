#!/usr/bin/env python3
"""
Utility script for managing job tracking database and analyzing results

Provides tools for:
- Viewing job statistics
- Resetting failed jobs
- Aggregating results from multiple jobs
- Database maintenance
"""

import sqlite3
import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import List, Dict


class JobTrackingUtils:
    """Utilities for managing job tracking database"""
    
    def __init__(self, db_path: str = "job_tracking.db"):
        """Initialize with database path"""
        self.db_path = db_path
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Connect to database"""
        if not Path(self.db_path).exists():
            print(f"Error: Database not found: {self.db_path}")
            sys.exit(1)
        
        self.conn = sqlite3.connect(self.db_path)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get comprehensive job statistics"""
        cursor = self.conn.cursor()
        
        # Overall stats
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM jobs 
            GROUP BY status
        """)
        stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Total
        cursor.execute("SELECT COUNT(*) FROM jobs")
        stats['total'] = cursor.fetchone()[0]
        
        # Failed with multiple attempts
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE status='failed' AND attempts > 1")
        stats['failed_multiple_attempts'] = cursor.fetchone()[0]
        
        # Average attempts
        cursor.execute("SELECT AVG(attempts) FROM jobs WHERE status='completed'")
        avg_attempts = cursor.fetchone()[0]
        stats['avg_attempts_completed'] = round(avg_attempts, 2) if avg_attempts else 0
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("JOB TRACKING STATISTICS")
        print("="*60)
        print(f"Total jobs:           {stats.get('total', 0)}")
        print(f"Pending:              {stats.get('pending', 0)}")
        print(f"Submitted:            {stats.get('submitted', 0)}")
        print(f"Running:              {stats.get('running', 0)}")
        print(f"Completed:            {stats.get('completed', 0)}")
        print(f"Failed:               {stats.get('failed', 0)}")
        print(f"Failed (multiple):    {stats.get('failed_multiple_attempts', 0)}")
        print(f"Avg attempts (done):  {stats.get('avg_attempts_completed', 0)}")
        print("="*60 + "\n")
    
    def list_failed_jobs(self, show_details: bool = False):
        """List all failed jobs"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name, attempts, error_message, slurm_job_id 
            FROM jobs 
            WHERE status='failed'
            ORDER BY attempts DESC
        """)
        
        failed = cursor.fetchall()
        
        if not failed:
            print("No failed jobs found.")
            return
        
        print(f"\nFailed Jobs ({len(failed)}):")
        print("-" * 80)
        
        for name, attempts, error_msg, slurm_id in failed:
            print(f"Job: {name}")
            print(f"  Attempts: {attempts}")
            print(f"  SLURM ID: {slurm_id or 'N/A'}")
            
            if show_details and error_msg:
                print(f"  Error: {error_msg}")
            
            print()
    
    def reset_failed_jobs(self, confirm: bool = False):
        """Reset failed jobs to pending status"""
        cursor = self.conn.cursor()
        
        # Count failed jobs
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE status='failed'")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("No failed jobs to reset.")
            return
        
        if not confirm:
            response = input(f"Reset {count} failed jobs to pending? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                return
        
        cursor.execute("""
            UPDATE jobs 
            SET status='pending', slurm_job_id=NULL, error_message=NULL
            WHERE status='failed'
        """)
        self.conn.commit()
        
        print(f"Reset {count} failed jobs to pending.")
    
    def reset_specific_jobs(self, job_names: List[str]):
        """Reset specific jobs by name"""
        cursor = self.conn.cursor()
        
        for job_name in job_names:
            cursor.execute("""
                UPDATE jobs 
                SET status='pending', slurm_job_id=NULL, error_message=NULL
                WHERE name=?
            """, (job_name,))
        
        self.conn.commit()
        print(f"Reset {len(job_names)} jobs to pending.")
    
    def delete_jobs(self, job_names: List[str], confirm: bool = False):
        """Delete specific jobs from database"""
        if not confirm:
            response = input(f"Delete {len(job_names)} jobs? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                return
        
        cursor = self.conn.cursor()
        
        for job_name in job_names:
            cursor.execute("DELETE FROM jobs WHERE name=?", (job_name,))
        
        self.conn.commit()
        print(f"Deleted {len(job_names)} jobs.")
    
    def export_to_csv(self, output_file: str, status_filter: str = None):
        """Export jobs to CSV, optionally filtering by status"""
        query = "SELECT * FROM jobs"
        if status_filter:
            query += f" WHERE status='{status_filter}'"
        
        df = pd.read_sql_query(query, self.conn)
        df.to_csv(output_file, index=False)
        print(f"Exported {len(df)} jobs to {output_file}")
    
    def aggregate_results(self, output_dir: str, output_file: str = "aggregated_results.csv"):
        """Aggregate all completed job results into single CSV"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name, output_csv 
            FROM jobs 
            WHERE status='completed' AND output_csv IS NOT NULL
        """)
        
        completed_jobs = cursor.fetchall()
        
        if not completed_jobs:
            print("No completed jobs with output files found.")
            return
        
        all_results = []
        
        for job_name, csv_path in completed_jobs:
            if not Path(csv_path).exists():
                # Try alternate path
                csv_path = Path(output_dir) / job_name / f"{job_name}_results.csv"
            
            if Path(csv_path).exists():
                try:
                    df = pd.read_csv(csv_path)
                    df.insert(0, 'job_name', job_name)
                    all_results.append(df)
                except Exception as e:
                    print(f"Warning: Could not read {csv_path}: {e}")
            else:
                print(f"Warning: Output file not found for {job_name}: {csv_path}")
        
        if not all_results:
            print("No valid result files found.")
            return
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        
        print(f"\nAggregated {len(all_results)} result files into {output_file}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Columns: {list(combined_df.columns)}")
    
    def find_missing_outputs(self, output_dir: str):
        """Find completed jobs that are missing output files"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name FROM jobs WHERE status='completed'
        """)
        
        completed_jobs = [row[0] for row in cursor.fetchall()]
        missing = []
        
        for job_name in completed_jobs:
            expected_csv = Path(output_dir) / job_name / f"{job_name}_results.csv"
            if not expected_csv.exists():
                missing.append(job_name)
        
        if missing:
            print(f"\nJobs marked completed but missing output files ({len(missing)}):")
            for job_name in missing:
                print(f"  - {job_name}")
            print(f"\nConsider resetting these jobs.")
        else:
            print("All completed jobs have output files.")
    
    def vacuum_database(self):
        """Optimize database"""
        self.conn.execute("VACUUM")
        print("Database optimized.")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Utilities for job tracking database management"
    )
    parser.add_argument(
        '--db-path',
        default='job_tracking.db',
        help='Path to job tracking database (default: job_tracking.db)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Stats command
    subparsers.add_parser('stats', help='Show job statistics')
    
    # List failed command
    list_failed = subparsers.add_parser('list-failed', help='List failed jobs')
    list_failed.add_argument('--details', action='store_true', help='Show error details')
    
    # Reset failed command
    reset_failed = subparsers.add_parser('reset-failed', help='Reset all failed jobs to pending')
    reset_failed.add_argument('--yes', action='store_true', help='Skip confirmation')
    
    # Reset specific jobs
    reset_jobs = subparsers.add_parser('reset', help='Reset specific jobs to pending')
    reset_jobs.add_argument('--jobs', nargs='+', required=True, help='Job names to reset')
    
    # Delete jobs
    delete_jobs = subparsers.add_parser('delete', help='Delete specific jobs')
    delete_jobs.add_argument('--jobs', nargs='+', required=True, help='Job names to delete')
    delete_jobs.add_argument('--yes', action='store_true', help='Skip confirmation')
    
    # Export command
    export = subparsers.add_parser('export', help='Export jobs to CSV')
    export.add_argument('--output', '-o', required=True, help='Output CSV file')
    export.add_argument('--status', help='Filter by status (pending, completed, failed, etc.)')
    
    # Aggregate results
    aggregate = subparsers.add_parser('aggregate', help='Aggregate all result CSVs')
    aggregate.add_argument('--output-dir', required=True, help='Base output directory')
    aggregate.add_argument('--output', '-o', default='aggregated_results.csv', 
                          help='Output file (default: aggregated_results.csv)')
    
    # Find missing outputs
    missing = subparsers.add_parser('find-missing', help='Find completed jobs with missing outputs')
    missing.add_argument('--output-dir', required=True, help='Base output directory')
    
    # Vacuum command
    subparsers.add_parser('vacuum', help='Optimize database')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize utils
    utils = JobTrackingUtils(db_path=args.db_path)
    
    try:
        if args.command == 'stats':
            utils.print_statistics()
        
        elif args.command == 'list-failed':
            utils.list_failed_jobs(show_details=args.details)
        
        elif args.command == 'reset-failed':
            utils.reset_failed_jobs(confirm=args.yes)
        
        elif args.command == 'reset':
            utils.reset_specific_jobs(args.jobs)
        
        elif args.command == 'delete':
            utils.delete_jobs(args.jobs, confirm=args.yes)
        
        elif args.command == 'export':
            utils.export_to_csv(args.output, status_filter=args.status)
        
        elif args.command == 'aggregate':
            utils.aggregate_results(args.output_dir, args.output)
        
        elif args.command == 'find-missing':
            utils.find_missing_outputs(args.output_dir)
        
        elif args.command == 'vacuum':
            utils.vacuum_database()
    
    finally:
        utils.close()


if __name__ == "__main__":
    main()
