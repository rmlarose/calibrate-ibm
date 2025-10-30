"""Test v5_parallel_v3 with 5k terms."""

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
    print("5k term v3 test\n")

    # Check for cached 5k subset
    cache_path = os.path.join(os.path.dirname(__file__), 'cached_5k_subset.pkl')
    if os.path.exists(cache_path):
        print("Loading cached 5k subset...")
        import pickle
        with open(cache_path, 'rb') as f:
            P_small, nqubits = pickle.load(f)
        print(f"  Loaded {len(P_small.terms)} terms")
    else:
        print("Creating 5k subset from scratch (will cache for next time)...")
        npz_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'owp_reactant.npz')
        hamiltonian_full, nqubits, _ = load_owp_hamiltonian(npz_path)
        P_small = create_random_subset(hamiltonian_full, 5000, seed=44)
        # Cache for next time
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump((P_small, nqubits), f)
        print(f"  Created and cached {len(P_small.terms)} terms")

    blocks_kN = [list(range(nqubits))]

    from kcommute.sorted_insertion import get_si_sets_v5, get_terms_ordered_by_abscoeff
    from kcommute.sorted_insertion_v5_parallel_v3 import get_si_sets_v5_parallel_v3

    # Compute terms_ord once for both tests
    print("\nPreparing terms...")
    terms_ord = get_terms_ordered_by_abscoeff(P_small)
    terms_ord = [term for term in terms_ord if () not in term.terms.keys()]
    print(f"  {len(terms_ord)} terms prepared")

    # v5 serial
    print("\nRunning v5 (serial)...")
    start = time.time()
    groups_v5 = get_si_sets_v5(P_small, blocks_kN, verbosity=0)
    time_v5 = time.time() - start
    print(f"  v5: {time_v5:.3f}s, {len(groups_v5)} groups")

    # v5_parallel_v3 with 2 workers
    print("\nRunning v5_parallel_v3 (2 workers)...")
    start = time.time()
    groups_v3 = get_si_sets_v5_parallel_v3(terms_ord, blocks_kN, verbosity=1, num_workers=2)
    time_v3 = time.time() - start
    print(f"  v5_parallel_v3: {time_v3:.3f}s, {len(groups_v3)} groups")

    speedup = time_v5 / time_v3
    print(f"\nSpeedup: {speedup:.2f}x")
