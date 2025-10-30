"""Benchmark OWP Hamiltonian with random subsets to project scaling.

Tests v5 (serial) and v5_parallel_v2 (multi-worker) on random subsets:
P_1 (5k terms) ⊂ P_2 (10k terms) ⊂ P_3 (20k terms) ⊂ H (575k terms)

Fits parabolic time complexity to project full system runtime.
"""

import os
import sys
import time
import numpy as np
from openfermion import QubitOperator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets_v5, get_si_sets_v5_parallel_v2
from Hamiltonians.load_owp import load_owp_hamiltonian


def create_random_subset(hamiltonian, n_terms, seed=42):
    """Create a random subset of terms from the Hamiltonian.

    Args:
        hamiltonian: QubitOperator
        n_terms: Number of terms to sample
        seed: Random seed for reproducibility

    Returns:
        QubitOperator with n_terms randomly selected terms
    """
    np.random.seed(seed)

    # Get all terms
    all_terms = list(hamiltonian.terms.items())

    # Sample without replacement
    if n_terms >= len(all_terms):
        return hamiltonian

    indices = np.random.choice(len(all_terms), size=n_terms, replace=False)
    sampled_terms = [all_terms[i] for i in indices]

    # Build new QubitOperator
    subset = QubitOperator()
    for pauli_string, coeff in sampled_terms:
        subset += QubitOperator(pauli_string, coeff)

    return subset


def benchmark_version(func, hamiltonian, blocks, name, num_runs=3, **kwargs):
    """Benchmark a grouping algorithm.

    Returns:
        tuple: (avg_time, num_groups)
    """
    times = []
    num_groups = None

    for run in range(num_runs):
        start = time.time()
        groups = func(hamiltonian, blocks, verbosity=0, **kwargs)
        elapsed = time.time() - start
        times.append(elapsed)

        if num_groups is None:
            num_groups = len(groups)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return avg_time, std_time, num_groups


def fit_parabolic(n_terms_list, times):
    """Fit parabolic (quadratic) model: t = a*n^2 + b*n + c

    Returns:
        tuple: (a, b, c) coefficients
    """
    n = np.array(n_terms_list)
    t = np.array(times)

    # Fit: t = a*n^2 + b*n + c
    A = np.column_stack([n**2, n, np.ones(len(n))])
    coeffs, _, _, _ = np.linalg.lstsq(A, t, rcond=None)

    return coeffs[0], coeffs[1], coeffs[2]


def predict_time(n, a, b, c):
    """Predict time for n terms using parabolic model."""
    return a * n**2 + b * n + c


if __name__ == "__main__":
    print("="*80)
    print("OWP HAMILTONIAN SCALING BENCHMARK")
    print("="*80)

    # Load full Hamiltonian
    print("\nLoading OWP Hamiltonian...")
    npz_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'owp_reactant.npz')
    hamiltonian_full, nqubits, nterms_full = load_owp_hamiltonian(npz_path)

    print(f"Full system: {nqubits} qubits, {nterms_full} terms")

    # Create nested random subsets: P_1 ⊂ P_2 ⊂ P_3 ⊂ H
    subset_sizes = [5000, 10000, 20000]

    print("\n" + "="*80)
    print("CREATING RANDOM SUBSETS")
    print("="*80)

    # First create P_3 (20k terms)
    print(f"\nCreating P_3 (20k terms) from full Hamiltonian...")
    P_3 = create_random_subset(hamiltonian_full, 20000, seed=42)

    # Then create P_2 (10k terms) from P_3
    print(f"Creating P_2 (10k terms) from P_3...")
    P_2 = create_random_subset(P_3, 10000, seed=43)

    # Finally create P_1 (5k terms) from P_2
    print(f"Creating P_1 (5k terms) from P_2...")
    P_1 = create_random_subset(P_2, 5000, seed=44)

    subsets = [P_1, P_2, P_3]

    print("\nSubset hierarchy:")
    print(f"  P_1: {len(P_1.terms)} terms")
    print(f"  P_2: {len(P_2.terms)} terms")
    print(f"  P_3: {len(P_3.terms)} terms")
    print(f"  H (full): {nterms_full} terms")

    blocks_kN = [list(range(nqubits))]

    # Test configurations
    worker_configs = [1, 2, 4, 8]

    # Store results
    results = {}

    print("\n" + "="*80)
    print("BENCHMARKING")
    print("="*80)

    for subset_idx, (subset, size) in enumerate(zip(subsets, subset_sizes)):
        print(f"\n{'='*80}")
        print(f"Subset P_{subset_idx+1}: {len(subset.terms)} terms")
        print(f"{'='*80}")

        # Benchmark v5 (serial)
        print(f"\nv5 (serial)...")
        avg_time, std_time, num_groups = benchmark_version(
            get_si_sets_v5, subset, blocks_kN, "v5", num_runs=3
        )
        print(f"  Time: {avg_time:.3f} ± {std_time:.3f} s ({num_groups} groups)")

        if 'v5' not in results:
            results['v5'] = {'sizes': [], 'times': [], 'stds': []}
        results['v5']['sizes'].append(len(subset.terms))
        results['v5']['times'].append(avg_time)
        results['v5']['stds'].append(std_time)

        # Benchmark v5_parallel_v2 with different worker counts
        for num_workers in worker_configs:
            config_name = f'v5_parallel_v2_{num_workers}w'
            print(f"\nv5_parallel_v2 ({num_workers} workers)...")

            avg_time, std_time, num_groups = benchmark_version(
                get_si_sets_v5_parallel_v2, subset, blocks_kN,
                config_name, num_runs=3, num_workers=num_workers
            )
            print(f"  Time: {avg_time:.3f} ± {std_time:.3f} s ({num_groups} groups)")

            if config_name not in results:
                results[config_name] = {'sizes': [], 'times': [], 'stds': []}
            results[config_name]['sizes'].append(len(subset.terms))
            results[config_name]['times'].append(avg_time)
            results[config_name]['stds'].append(std_time)

    # Fit parabolic models and project to full system
    print("\n" + "="*80)
    print("PARABOLIC FITTING & PROJECTION")
    print("="*80)

    projections = {}

    for version_name, data in results.items():
        print(f"\n{version_name}:")

        # Fit parabolic model
        a, b, c = fit_parabolic(data['sizes'], data['times'])
        print(f"  Model: t = {a:.3e}*n² + {b:.3e}*n + {c:.3e}")

        # Show fit quality
        print(f"  Measured times:")
        for size, time_measured in zip(data['sizes'], data['times']):
            time_predicted = predict_time(size, a, b, c)
            error = abs(time_measured - time_predicted) / time_measured * 100
            print(f"    {size:6d} terms: {time_measured:7.3f}s (predicted: {time_predicted:7.3f}s, error: {error:.1f}%)")

        # Project to full system
        time_full = predict_time(nterms_full, a, b, c)
        projections[version_name] = time_full
        print(f"  → Projected time for {nterms_full} terms: {time_full:.1f}s = {time_full/3600:.2f} hours")

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY: PROJECTED TIMES FOR FULL SYSTEM")
    print("="*80)

    v5_time = projections['v5']

    print(f"\n{'Version':<30} {'Time (hours)':<15} {'Speedup vs v5':<15}")
    print("-"*80)

    for version_name in sorted(projections.keys()):
        time_hours = projections[version_name] / 3600
        speedup = v5_time / projections[version_name]
        print(f"{version_name:<30} {time_hours:<15.2f} {speedup:<15.2f}x")

    print("-"*80)

    # Best configuration
    best_version = min(projections.keys(), key=lambda k: projections[k])
    best_time = projections[best_version] / 3600
    best_speedup = v5_time / projections[best_version]

    print(f"\nBest configuration: {best_version}")
    print(f"  Projected time: {best_time:.2f} hours")
    print(f"  Speedup: {best_speedup:.2f}x vs v5 serial")
    print(f"  Time saved: {(v5_time - projections[best_version])/3600:.2f} hours")
