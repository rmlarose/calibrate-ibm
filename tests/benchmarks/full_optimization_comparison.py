"""Full optimization journey: Original v1 → v5 → v5_parallel."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets, get_si_sets_v5, get_si_sets_v5_parallel
from Hamiltonians.load_h2o import load_h2o_hamiltonian


def time_algorithm(func, op, blocks, name, num_runs=3):
    """Time an algorithm multiple times."""
    times = []
    num_groups = None

    print(f"\n{name}:")
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...", end=" ", flush=True)
        start = time.time()
        groups = func(op, blocks, verbosity=0)
        elapsed = time.time() - start
        times.append(elapsed)
        if num_groups is None:
            num_groups = len(groups)
        print(f"{elapsed:.4f}s")

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.4f}s ({num_groups} groups)")
    return avg, num_groups


if __name__ == "__main__":
    print("="*80)
    print("OPTIMIZATION JOURNEY: v1 (Original) → v5 → v5_parallel")
    print("="*80)

    # Load H2O Hamiltonian
    hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
    hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

    print(f"\nH2O Hamiltonian: {nqubits} qubits, {nterms} terms")
    print(f"Testing with k=N (full commuting)")

    blocks_kN = [list(range(nqubits))]

    # v1 needs Cirq format blocks
    import cirq
    blocks_kN_cirq = [[cirq.LineQubit(i) for i in block] for block in blocks_kN]

    print(f"\n{'='*80}")
    print("BENCHMARKING")
    print(f"{'='*80}")

    # Benchmark v1 (original) - needs Cirq format blocks
    time_v1, groups_v1 = time_algorithm(get_si_sets, hamiltonian, blocks_kN_cirq, "v1 (Original - Cirq-based)", num_runs=3)

    # Benchmark v5 (optimized sequential)
    time_v5, groups_v5 = time_algorithm(get_si_sets_v5, hamiltonian, blocks_kN, "v5 (Optimized sequential)", num_runs=3)

    # Benchmark v5_parallel
    time_v5_parallel, groups_v5_parallel = time_algorithm(
        get_si_sets_v5_parallel, hamiltonian, blocks_kN, "v5_parallel (Parallel exclusion)", num_runs=3
    )

    # Verify all produce same results
    assert groups_v1 == groups_v5 == groups_v5_parallel, "Group counts differ!"

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    print(f"\n{'Version':<30} {'Time (s)':<12} {'Speedup vs v1':<15} {'Groups':<8}")
    print("-"*80)
    print(f"{'v1 (Original)':<30} {time_v1:<12.4f} {'1.00x':<15} {groups_v1:<8}")

    speedup_v5 = time_v1 / time_v5
    print(f"{'v5 (Optimized)':<30} {time_v5:<12.4f} {f'{speedup_v5:.2f}x':<15} {groups_v5:<8}")

    speedup_v5_parallel = time_v1 / time_v5_parallel
    print(f"{'v5_parallel (Parallel)':<30} {time_v5_parallel:<12.4f} {f'{speedup_v5_parallel:.2f}x':<15} {groups_v5_parallel:<8}")

    print("-"*80)

    print(f"\n{'='*80}")
    print("OPTIMIZATION BREAKDOWN")
    print(f"{'='*80}")

    print(f"\nv1 → v5:")
    improvement_v5 = (time_v1 - time_v5) / time_v1 * 100
    print(f"  {speedup_v5:.2f}x speedup ({improvement_v5:.1f}% faster)")
    print(f"  Optimizations:")
    print(f"    - Pre-computed block masks")
    print(f"    - Inlined k-commuting checks")
    print(f"    - Fast-path for k=N (single block)")
    print(f"    - Group-level caching (O(|group|) → O(1))")

    print(f"\nv5 → v5_parallel:")
    speedup_v5_to_parallel = time_v5 / time_v5_parallel
    improvement_parallel = (time_v5 - time_v5_parallel) / time_v5 * 100
    print(f"  {speedup_v5_to_parallel:.2f}x speedup ({improvement_parallel:.1f}% faster)")
    print(f"  Optimizations:")
    print(f"    - Back-to-front worker process")
    print(f"    - Parallel exclusion pre-computation")
    print(f"    - Skips groups known to not work")

    print(f"\nv1 → v5_parallel (Total):")
    total_improvement = (time_v1 - time_v5_parallel) / time_v1 * 100
    print(f"  {speedup_v5_parallel:.2f}x speedup ({total_improvement:.1f}% faster)")

    print(f"\n{'='*80}")
    print(f"🎉 FINAL: {speedup_v5_parallel:.2f}x FASTER THAN ORIGINAL!")
    print(f"{'='*80}")

    # Extrapolate to large system
    print(f"\n{'='*80}")
    print("EXTRAPOLATION TO 600K TERMS")
    print(f"{'='*80}")

    # Scaling: roughly n^2 with some constant factors
    scale_factor = (600000 / 1620) ** 2 * (44 / 14)  # Terms squared × qubit factor

    print(f"\nScaling factor: ~{scale_factor:.0f}x")
    print(f"  - Terms: (600k/1.6k)² ≈ {(600000/1620)**2:.0f}x")
    print(f"  - Qubits: 44/14 ≈ {44/14:.1f}x")

    est_v1 = time_v1 * scale_factor
    est_v5 = time_v5 * scale_factor
    est_v5_parallel = time_v5_parallel * scale_factor

    print(f"\nEstimated times for 600k terms:")
    print(f"  v1:          {est_v1/3600:.1f} hours")
    print(f"  v5:          {est_v5/3600:.1f} hours")
    print(f"  v5_parallel: {est_v5_parallel/3600:.1f} hours")

    print(f"\nTime saved by optimizations: {(est_v1 - est_v5_parallel)/3600:.1f} hours!")
