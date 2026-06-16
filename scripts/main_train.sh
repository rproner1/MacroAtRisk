#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --account=rrg-camera
#SBATCH --cpus-per-task=4
#SBATCH --time=03-00:00
#SBATCH --mem=240G
#SBATCH --array=1997-2023
#SBATCH --output=output/slurm-%A_%a.out
#SBATCH --mail-user=robert.proner@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.12
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip
#python -m pip install scikit-learn==1.5.0 statsmodels==0.14.5 tensorflow==2.17.0 quantile-forest==1.4.1 optuna==3.6.1 --no-index
#python -m pip install numpy==1.26.4 pandas==2.2.3 portalocker==2.10.1 scipy==1.13.1 pyyaml==6.0.2 python-dotenv==1.2.1 --no-index
python -m pip install --no-index -r requirements.txt
#python -m pip install --no-index ~/projects/rrg-camera/rproner/xlstm

TARGET=0
DATE=2026-06-16
python main_train.py --year $SLURM_ARRAY_TASK_ID --date $DATE --target $TARGET --model-type deep
