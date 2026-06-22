#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --account=rrg-camera
#SBATCH --cpus-per-task=4
#SBATCH --time=03-00:00
#SBATCH --mem=150G
#SBATCH --array=1997-2023
#SBATCH --output=output/slurm-%A_%a.out
#SBATCH --mail-user=robert.proner@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.12
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip
python -m pip install --no-index -r requirements.txt

TARGET=2
DATE=2026-06-22
python main_train.py --year $SLURM_ARRAY_TASK_ID --date $DATE --target $TARGET --model-type deep
