#!/bin/bash
#SBATCH --job-name=16-sites
#SBATCH --exclude=milan[2,8,10,23,25,27]
#SBATCH --time=2-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=data_16_4/data_grp_%a/output.out    
#SBATCH --array=0-399


module load mamba
source activate conda-example
# Job step
cd data_16_4/data_grp_$SLURM_ARRAY_TASK_ID
cp /scch/work/patelk1/thesis_code/training_data/data_gen.py data_gen.py


python -u data_gen.py
cd ..