#!/bin/bash
#SBATCH --job-name=sqd-sbd
#SBATCH --account=data-machine
#SBATCH --partition=data-machine
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --output=sqd_sbd_%a_%j.out
#SBATCH --error=sqd_sbd_%a_%j.err

module purge
module load NVHPC/23.7-CUDA-12.1.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.1.1
module load FlexiBLAS/3.3.1-NVHPC-23.7-CUDA-12.1.1
module load Miniforge3
conda activate thesisEnv

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

case $SLURM_ARRAY_TASK_ID in
    0) ADAPTS="1 2 3 4 5 10 20";            OUTPUT_DIR="results/cumulative_1_2_3_4_5_10_20" ;;
    1) ADAPTS="1 2 3 4 5 10 20 25 30";      OUTPUT_DIR="results/cumulative_1_2_3_4_5_10_20_25_30" ;;
    2) ADAPTS="1 2 3 4 5 10 20 25 30 40";   OUTPUT_DIR="results/cumulative_1_2_3_4_5_10_20_25_30_40" ;;
    3) ADAPTS="30";                          OUTPUT_DIR="results/singleton_30" ;;
esac

mkdir -p "$OUTPUT_DIR"

python run_sqd_sbd.py \
    --sbd_exe "${SBD_EXE:?Set SBD_EXE to path of diag executable}" \
    --output_dir "$OUTPUT_DIR" \
    --max_iterations 2000 \
    --resume \
    ${ADAPTS}
