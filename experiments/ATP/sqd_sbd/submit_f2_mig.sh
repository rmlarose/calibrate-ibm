#!/bin/bash
#SBATCH --job-name=sqd_f2_mig
#SBATCH --account=data-machine
#SBATCH --partition=data-machine
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=25-52
#SBATCH --output=logs/f2_mig_%A_%a.out
#SBATCH --error=logs/f2_mig_%A_%a.err

module purge
module load NVHPC/23.7-CUDA-12.1.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.1.1
module load FlexiBLAS/3.3.1-NVHPC-23.7-CUDA-12.1.1
module load Miniforge3
conda activate thesisEnv
ulimit -c 0

cd /mnt/ffs24/home/rowlan91/ben/calibrate-ibm/experiments/ATP/sqd_sbd
python dispatch_campaign_f2.py $SLURM_ARRAY_TASK_ID
