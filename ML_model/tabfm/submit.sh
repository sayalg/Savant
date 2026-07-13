#!/bin/sh
#SBATCH --job-name="savant_tabfm"
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=128GB
#SBATCH --gres=gpu:H200
#SBATCH --time=24:00:00


module load pytorch/2.9.1-cuda12.9 

python train_tabfm_reg_model.py