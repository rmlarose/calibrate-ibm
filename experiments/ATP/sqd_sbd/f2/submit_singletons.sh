#!/bin/bash
#SBATCH --job-name=sqd-f2-sing
#SBATCH --account=data-machine
#SBATCH --partition=data-machine
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-12
#SBATCH --output=f2/sqd_f2_sing_%a_%j.out
#SBATCH --error=f2/sqd_f2_sing_%a_%j.err

# Singleton SQD runs for atp_0_be2_f2 (44 orbitals, 44 electrons)
# Fresh start with pooled bitstrings (old 100k + new 100-300k)
# ADAPT iterations: 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50

module purge
module load NVHPC/23.7-CUDA-12.1.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.1.1
module load FlexiBLAS/3.3.1-NVHPC-23.7-CUDA-12.1.1
module load Miniforge3
conda activate thesisEnv

# Run from sqd_sbd/ so relative data paths (../circuits, etc.) resolve correctly
cd /mnt/ffs24/home/rowlan91/ben/calibrate-ibm/experiments/ATP/sqd_sbd

SBD_EXE="/mnt/ffs24/home/rowlan91/ben/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag_a100"

ADAPT_LIST=(1 2 3 4 5 10 15 20 25 30 35 40 50)
ADAPT=${ADAPT_LIST[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR="f2/results/hardware/singleton_${ADAPT}"
mkdir -p "$OUTPUT_DIR"

echo "=== Singleton ADAPT [${ADAPT}] for atp_0_be2_f2 ==="
echo "Output: $OUTPUT_DIR"

python run_sqd_sbd.py \
    --fragment atp_0_be2_f2 \
    --sbd_exe "$SBD_EXE" \
    --output_dir "$OUTPUT_DIR" \
    --max_iterations 500 \
    ${ADAPT}
