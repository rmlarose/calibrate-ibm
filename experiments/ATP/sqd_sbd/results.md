# SQD + SBD (GPU) Results

ATP fragment `atp_0_be2_f4` (32 orbitals, 32 electrons).
Convergence criterion: ΔE < 1e-8 AND ΔOcc < 1e-8.

## Singletons — Symmetrized (Sym)

| ADAPT | Converged? | Iters | Best Energy (Ha) | Dets (Sym) | Reference |
|-------|-----------|-------|-------------------|------------|-----------|
| 1 | YES | 12 | -261.8131 | 27,889 | `results/singleton_1/sqd_energies_1.txt` |
| 2 | YES | 14 | -261.8020 | 27,889 | `results/singleton_2/sqd_energies_2.txt` |
| 3 | YES | 34 | -261.8358 | 32,400 | `results/singleton_3/sqd_energies_3.txt` |
| 4 | YES | 51 | -261.8498 | 28,224 | `results/singleton_4/sqd_energies_4.txt` |
| 5 | YES | 45 | -261.8632 | 26,896 | `results/singleton_5/sqd_energies_5.txt` |
| 10 | YES | 275 | -261.8388 | 23,104 | `results/singleton_10/sqd_energies_10.txt` |
| 20 | YES | 353 | -261.8477 | 30,976 | `results/singleton_20/sqd_energies_20.txt` |
| 25 | YES | 376 | -261.8479 | 37,636 | `results/singleton_25/sqd_energies_25.txt` |
| 30 | NO | 372 | -261.8271 | 55,696 | `results/singleton_30/sqd_energies_30.txt` |
| 40 | YES | 220 | -261.8134 | 49,284 | `results/singleton_40/sqd_energies_40.txt` |

## Cumulatives — Symmetrized (Sym)

| Set | Converged? | Iters | Best Energy (Ha) | Dets (Sym) | Reference |
|-----|-----------|-------|-------------------|------------|-----------|
| [1,2] | YES | 5 | -261.8578 | 68,644 | `results/cumulative_1_2/sqd_energies_1_2.txt` |
| [1,2,3] | YES | 38 | -261.8894 | 44,944 | `results/cumulative_1_2_3/sqd_energies_1_2_3.txt` |
| [1,2,3,4] | YES | 49 | -261.8971 | 37,636 | `results/cumulative_1_2_3_4/sqd_energies_1_2_3_4.txt` |
| [1,2,3,4,5] | YES | 78 | -261.9010 | 35,344 | `results/cumulative_1_2_3_4_5/sqd_energies_1_2_3_4_5.txt` |
| [1,...,10] | YES | 339 | -261.9124 | 33,856 | `results/cumulative_1_2_3_4_5_10/sqd_energies_1_2_3_4_5_10.txt` |
| [1,...,20] | YES | 501 | -261.9171 | 33,856 | `results/cumulative_1_2_3_4_5_10_20/sqd_energies_1_2_3_4_5_10_20.txt` |
| [1,...,25] | NO | 289 | -261.9155 | 38,025 | `results/cumulative_1_2_3_4_5_10_20_25/sqd_energies_1_2_3_4_5_10_20_25.txt` |
| [1,...,30] | NO | 341 | -261.9157 | 53,824 | `results/cumulative_1_2_3_4_5_10_20_25_30/sqd_energies_1_2_3_4_5_10_20_25_30.txt` |
| [1,...,40] | NO | 354 | -261.9154 | 62,001 | `results/cumulative_1_2_3_4_5_10_20_25_30_40/sqd_energies_1_2_3_4_5_10_20_25_30_40.txt` |

## Singletons — No Symmetrization (No Sym)

| ADAPT | Converged? | Iters | Best Energy (Ha) | Dets (α×β) | Reference |
|-------|-----------|-------|-------------------|------------|-----------|
| 1 | YES | 11 | -261.7687 | 10,998 | `nosym/results/singleton_1/sqd_energies_1.txt` |
| 2 | YES | 7 | -261.7556 | 12,519 | `nosym/results/singleton_2/sqd_energies_2.txt` |
| 3 | YES | 26 | -261.7963 | 14,508 | `nosym/results/singleton_3/sqd_energies_3.txt` |
| 4 | YES | 36 | -261.8118 | 12,726 | `nosym/results/singleton_4/sqd_energies_4.txt` |
| 5 | YES | 62 | -261.8278 | 12,444 | `nosym/results/singleton_5/sqd_energies_5.txt` |
| 10 | YES | 282 | -261.8063 | 10,112 | `nosym/results/singleton_10/sqd_energies_10.txt` |
| 20 | NO | 256 | -261.8143 | 13,334 | `nosym/results/singleton_20/sqd_energies_20.txt` |
| 25 | YES | 227 | -261.8114 | 14,040 | `nosym/results/singleton_25/sqd_energies_25.txt` |
| 30 | NO | 159 | -261.8020 | 26,196 | `nosym/results/singleton_30/sqd_energies_30.txt` |
| 40 | NO | 156 | -261.8032 | 23,760 | `nosym/results/singleton_40/sqd_energies_40.txt` |

## Cumulatives — No Symmetrization (No Sym)

| Set | Converged? | Iters | Best Energy (Ha) | Dets (α×β) | Reference |
|-----|-----------|-------|-------------------|------------|-----------|
| [1,2] | YES | 14 | -261.8117 | 31,140 | `nosym/results/cumulative_1_2/sqd_energies_1_2.txt` |
| [1,2,3] | YES | 36 | -261.8510 | 19,599 | `nosym/results/cumulative_1_2_3/sqd_energies_1_2_3.txt` |
| [1,2,3,4] | YES | 64 | -261.8755 | 14,224 | `nosym/results/cumulative_1_2_3_4/sqd_energies_1_2_3_4.txt` |
| [1,...,5] | YES | 108 | -261.8859 | 15,390 | `nosym/results/cumulative_1_2_3_4_5/sqd_energies_1_2_3_4_5.txt` |
| [1,...,10] | NO | 300 | -261.9016 | 12,220 | `nosym/results/cumulative_1_2_3_4_5_10/sqd_energies_1_2_3_4_5_10.txt` |
| [1,...,20] | NO | 221 | -261.8919 | 16,368 | `nosym/results/cumulative_1_2_3_4_5_10_20/sqd_energies_1_2_3_4_5_10_20.txt` |
| [1,...,25] | NO | 188 | -261.8887 | 15,609 | `nosym/results/cumulative_1_2_3_4_5_10_20_25/sqd_energies_1_2_3_4_5_10_20_25.txt` |
| [1,...,30] | NO | 139 | -261.8783 | 24,940 | `nosym/results/cumulative_1_2_3_4_5_10_20_25_30/sqd_energies_1_2_3_4_5_10_20_25_30.txt` |
| [1,...,40] | NO | 127 | -261.8643 | 24,624 | `nosym/results/cumulative_1_2_3_4_5_10_20_25_30_40/sqd_energies_1_2_3_4_5_10_20_25_30_40.txt` |

## Reference Energies

- **CCSD**: -261.9430 Ha
- **HF**: -261.6958 Ha
