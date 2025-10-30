"""Test if queue connections are correct by adding debug logging."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Temporarily patch the worker to add logging
import kcommute.sorted_insertion as si

original_worker = si._multi_worker_process

def debug_worker(*args, **kwargs):
    worker_id = args[0]
    print(f"[DEBUG] Worker {worker_id} starting", flush=True)

    # Call original but we can't easily intercept queue puts
    # So let's just check the logic
    return original_worker(*args, **kwargs)

# Patch it
si._multi_worker_process = debug_worker

# Now run test
from Hamiltonians.load_owp import load_owp_hamiltonian
from openfermion import QubitOperator
import numpy as np

def create_random_subset(hamiltonian, n_terms, seed=42):
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

print("Loading OWP and creating 1k subset...")
npz_path = 'Hamiltonians/owp_reactant.npz'
ham_full, nq, _ = load_owp_hamiltonian(npz_path)
P_3 = create_random_subset(ham_full, 20000, seed=42)
P_2 = create_random_subset(P_3, 10000, seed=43)
P_1 = create_random_subset(P_2, 1000, seed=44)  # Small for quick test

blocks = [list(range(nq))]

print(f"\nRunning v5_parallel_v2 with 2 workers on {len(P_1.terms)} terms...")
groups = si.get_si_sets_v5_parallel_v2(P_1, blocks, verbosity=1, num_workers=2)
print(f"Completed: {len(groups)} groups")
