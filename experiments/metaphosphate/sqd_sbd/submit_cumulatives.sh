#!/bin/bash
#SBATCH --job-name=sqd-mp-cum
#SBATCH --account=data-machine
#SBATCH --partition=data-machine
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --output=results/hardware/cumulative_%a_%j.out
#SBATCH --error=results/hardware/cumulative_%a_%j.err

# Cumulative SQD runs for metaphosphate-2026 (22 orbitals, 32 electrons)
# Purely quantum (no S² penalty), symmetrized spin
# Sets: [1,2], [1,2,3], [1,2,3,4], [1,2,3,4,5], [1,2,3,4,5,10]

module purge
module load NVHPC/23.7-CUDA-12.1.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.1.1
module load FlexiBLAS/3.3.1-NVHPC-23.7-CUDA-12.1.1
module load Miniforge3
conda activate thesisEnv

cd /mnt/ffs24/home/rowlan91/ben/calibrate-ibm/experiments/metaphosphate/sqd_sbd

SBD_EXE="/mnt/ffs24/home/rowlan91/ben/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag_a100"

case $SLURM_ARRAY_TASK_ID in
    0) ADAPTS="1 2";           OUTPUT_DIR="results/hardware/cumulative_1_2" ;;
    1) ADAPTS="1 2 3";         OUTPUT_DIR="results/hardware/cumulative_1_2_3" ;;
    2) ADAPTS="1 2 3 4";       OUTPUT_DIR="results/hardware/cumulative_1_2_3_4" ;;
    3) ADAPTS="1 2 3 4 5";     OUTPUT_DIR="results/hardware/cumulative_1_2_3_4_5" ;;
    4) ADAPTS="1 2 3 4 5 10";  OUTPUT_DIR="results/hardware/cumulative_1_2_3_4_5_10" ;;
esac

mkdir -p "$OUTPUT_DIR"

echo "=== Cumulative ADAPT [${ADAPTS}] for metaphosphate-2026 ==="
echo "Output: $OUTPUT_DIR"

python run_sqd_sbd.py \
    --fragment metaphosphate-2026 \
    --circuit_dir ../circuits \
    --hamiltonian_dir ../hamiltonians \
    --results_dir ../results \
    --sbd_exe "$SBD_EXE" \
    --output_dir "$OUTPUT_DIR" \
    --max_iterations 500 \
    --resume \
    ${ADAPTS}
