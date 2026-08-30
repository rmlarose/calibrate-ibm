#!/bin/bash
#SBATCH --job-name=sqd-f4-cum
#SBATCH --account=data-machine
#SBATCH --partition=data-machine
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-8
#SBATCH --output=f4/sqd_f4_cum_%a_%j.out
#SBATCH --error=f4/sqd_f4_cum_%a_%j.err

# Cumulative SQD runs for atp_0_be2_f4 (32 orbitals, 32 electrons)
# Purely quantum (no S² penalty), symmetrized spin

module purge
module load NVHPC/23.7-CUDA-12.1.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.1.1
module load FlexiBLAS/3.3.1-NVHPC-23.7-CUDA-12.1.1
module load Miniforge3
conda activate thesisEnv

cd /mnt/ffs24/home/rowlan91/ben/calibrate-ibm/experiments/ATP/sqd_sbd

SBD_EXE="/mnt/ffs24/home/rowlan91/ben/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag_a100"

case $SLURM_ARRAY_TASK_ID in
    0) ADAPTS="1 2";                                 OUTPUT_DIR="f4/results/hardware/cumulative_1_2" ;;
    1) ADAPTS="1 2 3";                               OUTPUT_DIR="f4/results/hardware/cumulative_1_2_3" ;;
    2) ADAPTS="1 2 3 4";                             OUTPUT_DIR="f4/results/hardware/cumulative_1_2_3_4" ;;
    3) ADAPTS="1 2 3 4 5";                           OUTPUT_DIR="f4/results/hardware/cumulative_1_2_3_4_5" ;;
    4) ADAPTS="1 2 3 4 5 10";                        OUTPUT_DIR="f4/results/hardware/cumulative_1_2_3_4_5_10" ;;
    5) ADAPTS="1 2 3 4 5 10 20";                     OUTPUT_DIR="f4/results/hardware/cumulative_1_2_3_4_5_10_20" ;;
    6) ADAPTS="1 2 3 4 5 10 20 25";                  OUTPUT_DIR="f4/results/hardware/cumulative_1_2_3_4_5_10_20_25" ;;
    7) ADAPTS="1 2 3 4 5 10 20 25 30";               OUTPUT_DIR="f4/results/hardware/cumulative_1_2_3_4_5_10_20_25_30" ;;
    8) ADAPTS="1 2 3 4 5 10 20 25 30 40";            OUTPUT_DIR="f4/results/hardware/cumulative_1_2_3_4_5_10_20_25_30_40" ;;
esac

mkdir -p "$OUTPUT_DIR"

echo "=== Cumulative ADAPT [${ADAPTS}] for atp_0_be2_f4 ==="
echo "Output: $OUTPUT_DIR"

python run_sqd_sbd.py \
    --fragment atp_0_be2_f4 \
    --sbd_exe "$SBD_EXE" \
    --output_dir "$OUTPUT_DIR" \
    --max_iterations 2000 \
    --resume \
    ${ADAPTS}
