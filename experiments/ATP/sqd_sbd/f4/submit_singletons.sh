#!/bin/bash
#SBATCH --job-name=sqd-f4-sing
#SBATCH --account=data-machine
#SBATCH --partition=data-machine
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-9
#SBATCH --output=f4/sqd_f4_sing_%a_%j.out
#SBATCH --error=f4/sqd_f4_sing_%a_%j.err

# Singleton SQD runs for atp_0_be2_f4 (32 orbitals, 32 electrons)
# Purely quantum (no S² penalty), symmetrized spin
# ADAPT iterations: 1, 2, 3, 4, 5, 10, 20, 25, 30, 40

module purge
module load NVHPC/23.7-CUDA-12.1.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.1.1
module load FlexiBLAS/3.3.1-NVHPC-23.7-CUDA-12.1.1
module load Miniforge3
conda activate thesisEnv

cd /mnt/ffs24/home/rowlan91/ben/calibrate-ibm/experiments/ATP/sqd_sbd

SBD_EXE="/mnt/ffs24/home/rowlan91/ben/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag_a100"

ADAPT_LIST=(1 2 3 4 5 10 20 25 30 40)
ADAPT=${ADAPT_LIST[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR="f4/results/hardware/singleton_${ADAPT}"
mkdir -p "$OUTPUT_DIR"

echo "=== Singleton ADAPT [${ADAPT}] for atp_0_be2_f4 ==="
echo "Output: $OUTPUT_DIR"

python run_sqd_sbd.py \
    --fragment atp_0_be2_f4 \
    --sbd_exe "$SBD_EXE" \
    --output_dir "$OUTPUT_DIR" \
    --max_iterations 2000 \
    --resume \
    ${ADAPT}
