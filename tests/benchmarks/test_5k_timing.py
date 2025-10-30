"""Quick timing test with 5k terms."""

import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Hamiltonians.load_owp import load_owp_hamiltonian
from openfermion import QubitOperator


def create_random_subset(hamiltonian, n_terms, seed=42):
    """Create a random subset of terms."""
    np.random.seed(seed)
    all_terms = list(hamiltonian.terms.items())
    if n_terms >= len(all_terms):
        return hamiltonian
    indices = np.random.choice(len(all_terms), size=n_terms, replace=False)
    sampled_terms = [all_terms[i] for i in indices]
    subset = QubitOperator()
    for pauli_string, coeff in sampled_terms:
        subset += QubitOperator(pauli_string, coeff)
    return subset


if __name__ == "__main__":
    print("5k term timing test\n")

    # Load and create small subset
    npz_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'owp_reactant.npz')
    hamiltonian_full, nqubits, _ = load_owp_hamiltonian(npz_path)

    print("Creating 5k term subset...")
    P_small = create_random_subset(hamiltonian_full, 5000, seed=44)

    blocks_kN = [list(range(nqubits))]

    from kcommute.sorted_insertion import get_si_sets_v5, get_si_sets_v5_parallel_v2

    # v5 serial
    print("\nRunning v5 (serial)...")
    start = time.time()
    groups_v5 = get_si_sets_v5(P_small, blocks_kN, verbosity=0)
    time_v5 = time.time() - start
    print(f"  v5: {time_v5:.3f}s, {len(groups_v5)} groups")

    # v5_parallel_v2 with 2 workers
    print("\nRunning v5_parallel_v2 (2 workers)...")
    start = time.time()
    groups_par = get_si_sets_v5_parallel_v2(P_small, blocks_kN, verbosity=0, num_workers=2)
    time_par = time.time() - start
    print(f"  v5_parallel_v2: {time_par:.3f}s, {len(groups_par)} groups")

    speedup = time_v5 / time_par
    print(f"\nSpeedup: {speedup:.2f}x")
