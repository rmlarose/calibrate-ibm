#!/bin/bash
#SBATCH --job-name=sqd-nosym
#SBATCH --account=data-machine
#SBATCH --partition=data-machine
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-18
#SBATCH --output=sqd_nosym_%a_%j.out
#SBATCH --error=sqd_nosym_%a_%j.err

module purge
module load NVHPC/23.7-CUDA-12.1.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.1.1
module load FlexiBLAS/3.3.1-NVHPC-23.7-CUDA-12.1.1
module load Miniforge3
conda activate thesisEnv

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Array index -> (type, adapt_iterations)
# 0-9:   singletons for ADAPT 1, 2, 3, 4, 5, 10, 20, 25, 30, 40
# 10-18: cumulatives [1,2] through [1,...,40]
case $SLURM_ARRAY_TASK_ID in
    0)  TYPE="singleton"; ADAPTS="1" ;;
    1)  TYPE="singleton"; ADAPTS="2" ;;
    2)  TYPE="singleton"; ADAPTS="3" ;;
    3)  TYPE="singleton"; ADAPTS="4" ;;
    4)  TYPE="singleton"; ADAPTS="5" ;;
    5)  TYPE="singleton"; ADAPTS="10" ;;
    6)  TYPE="singleton"; ADAPTS="20" ;;
    7)  TYPE="singleton"; ADAPTS="25" ;;
    8)  TYPE="singleton"; ADAPTS="30" ;;
    9)  TYPE="singleton"; ADAPTS="40" ;;
    10) TYPE="cumulative"; ADAPTS="1 2" ;;
    11) TYPE="cumulative"; ADAPTS="1 2 3" ;;
    12) TYPE="cumulative"; ADAPTS="1 2 3 4" ;;
    13) TYPE="cumulative"; ADAPTS="1 2 3 4 5" ;;
    14) TYPE="cumulative"; ADAPTS="1 2 3 4 5 10" ;;
    15) TYPE="cumulative"; ADAPTS="1 2 3 4 5 10 20" ;;
    16) TYPE="cumulative"; ADAPTS="1 2 3 4 5 10 20 25" ;;
    17) TYPE="cumulative"; ADAPTS="1 2 3 4 5 10 20 25 30" ;;
    18) TYPE="cumulative"; ADAPTS="1 2 3 4 5 10 20 25 30 40" ;;
esac

ADAPTS_KEY=$(echo $ADAPTS | tr ' ' '_')
if [ "$TYPE" = "singleton" ]; then
    OUTPUT_DIR="results/singleton_${ADAPTS_KEY}"
else
    OUTPUT_DIR="results/cumulative_${ADAPTS_KEY}"
fi

python run_sqd_nosym.py \
    --circuit_dir "../../circuits" \
    --hamiltonian_dir "../../hamiltonians" \
    --results_dir "../../results" \
    --sbd_exe "${SBD_EXE:?Set SBD_EXE to path of diag executable}" \
    --output_dir "$OUTPUT_DIR" \
    --max_iterations 2000 \
    --resume \
    ${ADAPTS}
