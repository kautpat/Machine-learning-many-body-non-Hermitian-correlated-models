#!/bin/bash
#SBATCH --job-name=test-model
#SBATCH --partition=gpu-debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=tf_train.log

# Load modules
module load cuda/12.2.1
module load mamba
source activate tensor-env

# Run your training script
python -u tester.py
