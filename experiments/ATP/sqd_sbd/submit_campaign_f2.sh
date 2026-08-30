#!/bin/bash
#SBATCH --job-name=sqd_f2
#SBATCH --partition=LocalQ
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-52
#SBATCH --output=logs/f2_campaign_%A_%a.out
#SBATCH --error=logs/f2_campaign_%A_%a.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate sqd

# NVHPC for SBD binary
export PATH=~/nvhpc/Linux_x86_64/25.1/comm_libs/12.6/hpcx/latest/ompi/bin:~/nvhpc/Linux_x86_64/25.1/compilers/bin:$PATH
export LD_LIBRARY_PATH=~/nvhpc/Linux_x86_64/25.1/compilers/lib:~/nvhpc/Linux_x86_64/25.1/cuda/12.6/lib64:~/nvhpc/Linux_x86_64/25.1/comm_libs/12.6/hpcx/latest/ompi/lib:$LD_LIBRARY_PATH
ulimit -c 0

cd /home/rowlan91/ben/calibrate-ibm/experiments/ATP/sqd_sbd
python dispatch_campaign_f2.py $SLURM_ARRAY_TASK_ID
