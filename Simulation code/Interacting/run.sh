#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=stability_check/H_%a/task_run_dat.out   
#SBATCH --array=0-10

module load mamba
source activate conda-example

# Job step
srun python3 -u script.py $SLURM_ARRAY_TASK_ID
