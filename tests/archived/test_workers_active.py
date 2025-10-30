"""Test if workers are actually doing useful work in v5_parallel_v2."""

import os
import sys
import time
import numpy as np

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
    print("Testing if workers are active and providing exclusions...")

    # Load small subset for quick test
    print("\nLoading OWP Hamiltonian...")
    npz_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'owp_reactant.npz')
    hamiltonian_full, nqubits, _ = load_owp_hamiltonian(npz_path)

    print("Creating 5k term subset...")
    P_3_full = create_random_subset(hamiltonian_full, 20000, seed=42)
    P_2 = create_random_subset(P_3_full, 10000, seed=43)
    P_1 = create_random_subset(P_2, 5000, seed=44)

    blocks_kN = [list(range(nqubits))]

    # Run with verbosity to see worker activity
    print("\n" + "="*80)
    print("Running v5_parallel_v2 with 2 workers (verbosity=1)")
    print("="*80)

    from kcommute.sorted_insertion import get_si_sets_v5_parallel_v2

    start = time.time()
    groups = get_si_sets_v5_parallel_v2(P_1, blocks_kN, verbosity=1, num_workers=2)
    elapsed = time.time() - start

    print(f"\nCompleted: {len(groups)} groups in {elapsed:.3f}s")
