#!/bin/bash
#SBATCH --job-name=sqd_full_gpu
#SBATCH --partition=general-long-gpu
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-19
#SBATCH --output=logs/full_gpu_%A_%a.out
#SBATCH --error=logs/full_gpu_%A_%a.err

# Map array indices to original dispatch_campaign.py task IDs
# Hardware first (f18 hardware 44-52, meta hardware 2), then semiclassical
TASK_IDS=(44 45 46 47 48 49 50 51 52 2 13 14 55 56 57 58 61 62 63 64)

DISPATCH_ID=${TASK_IDS[$SLURM_ARRAY_TASK_ID]}

module purge
module load NVHPC/23.7-CUDA-12.1.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.1.1
module load FlexiBLAS/3.3.1-NVHPC-23.7-CUDA-12.1.1
module load Miniforge3
conda activate thesisEnv
ulimit -c 0

cd /mnt/ffs24/home/rowlan91/ben/calibrate-ibm/experiments/ATP/sqd_sbd
echo "Array index $SLURM_ARRAY_TASK_ID -> dispatch ID $DISPATCH_ID"
python dispatch_campaign.py $DISPATCH_ID
