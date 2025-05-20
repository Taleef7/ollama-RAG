#!/bin/bash

#SBATCH --job-name=gpu_test

#SBATCH --output=gpu_test_output_%j.txt  # %j will be replaced by the job ID

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=2

#SBATCH --mem=8G

#SBATCH --time=00:10:00

#SBATCH --account=standby       # Use 'standby' or 'debug'. Check 'slist' for your options

#SBATCH --gres=gpu:1              # Request 1 GPU



echo "Job started on $(hostname) at $(date)"

echo "SLURM Job ID: $SLURM_JOBID"

echo "SLURM Partition: $SLURM_JOB_PARTITION"

echo "SLURM GPU(s) assigned: $CUDA_VISIBLE_DEVICES" # This env var is set by Slurm with --gres=gpu



# Navigate to your project directory (IMPORTANT!)

cd /scratch/gilbreth/ttamsal/ollama-RAG  # <<< MAKE SURE THIS PATH IS CORRECT FOR YOU



# Activate your virtual environment

source venv/bin/activate

echo "Virtual environment activated."

echo "Python version: $(python --version)"

echo "Which python: $(which python)"





# Run the test script

echo "Running test_gpu.py..."

python -u test_gpu.py  # The -u makes output unbuffered, good for logs



echo "Job finished at $(date)"
