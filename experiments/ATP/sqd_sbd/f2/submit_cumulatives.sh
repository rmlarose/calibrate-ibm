#!/bin/bash
#SBATCH --job-name=sqd-f2-cum
#SBATCH --account=data-machine
#SBATCH --partition=data-machine
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-11
#SBATCH --output=f2/sqd_f2_cum_%a_%j.out
#SBATCH --error=f2/sqd_f2_cum_%a_%j.err

# Cumulative SQD runs for atp_0_be2_f2 (44 orbitals, 44 electrons)
# Fresh start with pooled bitstrings (old 100k + new 100-300k)

module purge
module load NVHPC/23.7-CUDA-12.1.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.1.1
module load FlexiBLAS/3.3.1-NVHPC-23.7-CUDA-12.1.1
module load Miniforge3
conda activate thesisEnv

# Run from sqd_sbd/ so relative data paths (../circuits, etc.) resolve correctly
cd /mnt/ffs24/home/rowlan91/ben/calibrate-ibm/experiments/ATP/sqd_sbd

SBD_EXE="/mnt/ffs24/home/rowlan91/ben/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag_a100"

case $SLURM_ARRAY_TASK_ID in
    0)  ADAPTS="1 2";                                           OUTPUT_DIR="f2/results/hardware/cumulative_1_2" ;;
    1)  ADAPTS="1 2 3";                                         OUTPUT_DIR="f2/results/hardware/cumulative_1_2_3" ;;
    2)  ADAPTS="1 2 3 4";                                       OUTPUT_DIR="f2/results/hardware/cumulative_1_2_3_4" ;;
    3)  ADAPTS="1 2 3 4 5";                                     OUTPUT_DIR="f2/results/hardware/cumulative_1_2_3_4_5" ;;
    4)  ADAPTS="1 2 3 4 5 10";                                  OUTPUT_DIR="f2/results/hardware/cumulative_1_2_3_4_5_10" ;;
    5)  ADAPTS="1 2 3 4 5 10 15";                               OUTPUT_DIR="f2/results/hardware/cumulative_1_2_3_4_5_10_15" ;;
    6)  ADAPTS="1 2 3 4 5 10 15 20";                            OUTPUT_DIR="f2/results/hardware/cumulative_1_2_3_4_5_10_15_20" ;;
    7)  ADAPTS="1 2 3 4 5 10 15 20 25";                         OUTPUT_DIR="f2/results/hardware/cumulative_1_2_3_4_5_10_15_20_25" ;;
    8)  ADAPTS="1 2 3 4 5 10 15 20 25 30";                      OUTPUT_DIR="f2/results/hardware/cumulative_1_2_3_4_5_10_15_20_25_30" ;;
    9)  ADAPTS="1 2 3 4 5 10 15 20 25 30 35";                   OUTPUT_DIR="f2/results/hardware/cumulative_1_2_3_4_5_10_15_20_25_30_35" ;;
    10) ADAPTS="1 2 3 4 5 10 15 20 25 30 35 40";                OUTPUT_DIR="f2/results/hardware/cumulative_1_2_3_4_5_10_15_20_25_30_35_40" ;;
    11) ADAPTS="1 2 3 4 5 10 15 20 25 30 35 40 50";             OUTPUT_DIR="f2/results/hardware/cumulative_1_2_3_4_5_10_15_20_25_30_35_40_50" ;;
esac

mkdir -p "$OUTPUT_DIR"

echo "=== Cumulative ADAPT [${ADAPTS}] for atp_0_be2_f2 ==="
echo "Output: $OUTPUT_DIR"

python run_sqd_sbd.py \
    --fragment atp_0_be2_f2 \
    --sbd_exe "$SBD_EXE" \
    --output_dir "$OUTPUT_DIR" \
    --max_iterations 500 \
    ${ADAPTS}
