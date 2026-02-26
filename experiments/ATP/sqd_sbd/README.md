# SQD + SBD (GPU) for Molecular Fragments

GPU-accelerated [Sample-based Quantum Diagonalization (SQD)](https://arxiv.org/abs/2405.05068) using [SBD](https://github.com/r-ccs-cms/sbd) (Selected Basis Diagonalization) as the CI solver, replacing PySCF's CPU-based Selected CI.

Applied to three molecular fragments using measurements from IBM quantum hardware:
- **ATP f4** (32 orbitals, 32 electrons) — hardware, nosym, and semiclassical variants
- **ATP f2** (44 orbitals, 44 electrons) — hardware
- **Metaphosphate** (22 orbitals, 32 electrons) — hardware (via symlink)

## Directory structure

```
sqd_sbd/
├── run_sqd_sbd.py               # Unified SQD loop (sym, nosym, semiclassical)
├── check_status.py              # Zero-argument status report for all fragments
├── plot_all.py                  # Zero-argument plot generation for all fragments
├── generate_random_bitstrings.py
├── README.md
├── results.md                   # Legacy summary tables
│
├── f4/                          # ATP fragment f4 (32 orb, 32 elec)
│   ├── results/
│   │   ├── hardware/            # singleton_N/, cumulative_X_Y_.../ (sym)
│   │   ├── nosym/               # singleton_N/, cumulative_X_Y_.../ (no spin sym)
│   │   └── semiclassical/       # results_10k/, results_50k/, etc.
│   ├── data/                    # counts_*.pkl, f4.wf, f4-energies.txt, ipr_results/
│   ├── plots/
│   ├── submit_singletons.sh
│   ├── submit_cumulatives.sh
│   ├── sample_wavefunction.py   # f4-specific ASCI wavefunction sampler
│   ├── compute_ipr.py           # f4-specific IPR analysis
│   └── NOSYM_README.md
│
├── f2/                          # ATP fragment f2 (44 orb, 44 elec)
│   ├── results/
│   │   └── hardware/            # singleton_N/, cumulative_X_Y_.../
│   ├── plots/
│   ├── submit_singletons.sh
│   └── submit_cumulatives.sh
│
└── (metaphosphate via ../../metaphosphate/sqd_sbd/)
    ├── run_sqd_sbd.py -> symlink to ATP/sqd_sbd/run_sqd_sbd.py
    ├── results/
    │   └── hardware/            # singleton_N/, cumulative_X_Y_.../
    ├── plots/
    ├── submit_singletons.sh
    └── submit_cumulatives.sh
```

Each result subdirectory contains:
- `sqd_energies_<label>.txt` — energy per iteration
- `sqd_stats_<label>.txt` — iter, energy, ci_strings, dets, pool_size, carryover, wall_sec
- `sqd_occupancies_<label>.txt` — orbital occupancies per iteration
- `checkpoint_<label>.pkl` — resumable state
- `sqd_convergence_<label>.pdf` — convergence plot

## Requirements

**SBD solver** (external, not included — see [jdrowland/sbd](https://github.com/jdrowland/sbd)):
- NVHPC compiler with CUDA support (tested with NVHPC 23.7, CUDA 12.1)
- MPI (OpenMPI)
- LAPACK/BLAS (FlexiBLAS or OpenBLAS)
- NVIDIA GPU (A100 or H200)

**Python environment:**
- numpy
- pyscf
- qiskit-addon-sqd
- matplotlib

## Building SBD

Follow the instructions in the [SBD repository](https://github.com/jdrowland/sbd). On MSU ICER:

```bash
module load NVHPC/23.7-CUDA-12.1.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.1.1
module load FlexiBLAS/3.3.1-NVHPC-23.7-CUDA-12.1.1

cd sbd/apps/chemistry_tpb_selected_basis_diagonalization
# Edit Configuration: set -gpu=cc80 (A100) or -gpu=cc90 (H200)
make
```

This produces the `diag` executable.

## Running

Scripts use relative paths to reference data already in this repository (circuits, hamiltonians, and measurement results in `../circuits`, `../hamiltonians`, `../results`). The only external dependency is the SBD executable.

```bash
cd experiments/ATP/sqd_sbd

# Symmetrized spin (default)
python run_sqd_sbd.py \
    --sbd_exe /path/to/sbd/diag \
    --output_dir f4/results/hardware/cumulative_1_2_3_4_5 \
    --max_iterations 2000 \
    --resume \
    1 2 3 4 5

# No spin symmetrization
python run_sqd_sbd.py \
    --nosym \
    --sbd_exe /path/to/sbd/diag \
    --output_dir f4/results/nosym/singleton_5 \
    --max_iterations 2000 \
    5

# Semiclassical (merge ASCI-sampled bitstrings with hardware data)
python run_sqd_sbd.py \
    --semiclassical_counts f4/data/counts_100000.pkl \
    --sbd_exe /path/to/sbd/diag \
    --output_dir f4/results/semiclassical/results_100k \
    --max_iterations 2000

# SLURM batch (see per-fragment submit scripts)
sbatch f4/submit_singletons.sh
sbatch f2/submit_cumulatives.sh
```

The positional arguments are ADAPT-VQE iteration indices. A single index runs a "singleton" experiment; multiple indices pool the measurement data ("cumulative").

Multiple bitstring files per ADAPT iteration are automatically merged (enabling pooled shot counts from multiple hardware runs).

## Status and plots

```bash
python check_status.py    # prints tables for all fragments, writes status.json
python plot_all.py        # generates plots in each fragment's plots/ directory
```

## Results

Best energies:
- **f4**: -261.9171 Ha (sym cumulative [1,...,20], 501 iterations)
- **f2**: -422.6237 Ha (cumulative [1,...,25], 71 iterations, not converged)
- **Metaphosphate**: -1121.8057 Ha (cumulative [1,...,10], 76 iterations, not converged)

See `results.md` for legacy tables, or run `python check_status.py` for current status.

## References

- **SBD library**: Tomonori Shirakawa, RIKEN Center for Computational Science — https://github.com/r-ccs-cms/sbd
  - [Closed-loop calculations of electronic structure on a quantum processor and a classical supercomputer at full scale](https://arxiv.org/abs/2511.00224)
  - [GPU-Accelerated Selected Basis Diagonalization with Thrust for SQD-based Algorithms](https://arxiv.org/abs/2601.16637)
- **SQD algorithm**: J. Robledo-Moreno et al., [Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068)
