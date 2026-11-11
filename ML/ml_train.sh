#!/bin/bash
#SBATCH --job-name=tf-train
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=16_site-density-matrix/tf_train.log

# Load modules 
module load cuda/12.2.1
module load mamba
source activate tensor-env

# Job step
cp neural_network.py 16_site-density-matrix/nn.py

# Run your training script
cd 16_site-density-matrix
python -u nn.py
cd ..