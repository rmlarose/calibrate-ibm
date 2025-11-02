"""Test H2O Hamiltonian with timing comparison across all three implementations."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets, get_si_sets_v2, get_terms_ordered_by_abscoeff
from kcommute.sorted_insertion_v2_parallel import get_si_sets_v2_parallel
from Hamiltonians.load_h2o import load_h2o_hamiltonian
from openfermion import qubit_operator_to_pauli_sum


print("="*80)
print("Testing H2O Hamiltonian with timing comparison")
print("="*80)

hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

hamiltonian_cirq = qubit_operator_to_pauli_sum(hamiltonian)
cirq_qubits = sorted(hamiltonian_cirq.qubits)
blocks = [cirq_qubits]
blocks_int = [list(range(nqubits))]

print(f"\nRunning get_si_sets...")
start = time.time()
groups_v1 = get_si_sets(hamiltonian, blocks, verbosity=0)
t_v1 = time.time() - start
print(f"  Time: {t_v1:.3f}s, {len(groups_v1)} groups")

print(f"\nRunning get_si_sets_v2...")
start = time.time()
groups_v2 = get_si_sets_v2(hamiltonian, blocks_int, verbosity=0)
t_v2 = time.time() - start
speedup_v2 = t_v1 / t_v2
print(f"  Time: {t_v2:.3f}s, {len(groups_v2)} groups, speedup: {speedup_v2:.2f}x")

print(f"\nRunning get_si_sets_v2_parallel...")
terms = get_terms_ordered_by_abscoeff(hamiltonian)
terms = [t for t in terms if () not in t.terms.keys()]
start = time.time()
groups_v2_parallel = get_si_sets_v2_parallel(terms, blocks_int, verbosity=0, num_workers=2)
t_v2_parallel = time.time() - start
speedup_v2_parallel = t_v1 / t_v2_parallel
print(f"  Time: {t_v2_parallel:.3f}s, {len(groups_v2_parallel)} groups, speedup: {speedup_v2_parallel:.2f}x")

print(f"\n{'='*80}")
print(f"Summary:")
print(f"  get_si_sets:             {t_v1:.3f}s, {len(groups_v1)} groups (baseline)")
print(f"  get_si_sets_v2:          {t_v2:.3f}s, {len(groups_v2)} groups ({speedup_v2:.2f}x faster)")
print(f"  get_si_sets_v2_parallel: {t_v2_parallel:.3f}s, {len(groups_v2_parallel)} groups ({speedup_v2_parallel:.2f}x faster)")
print(f"{'='*80}")

if len(groups_v1) == len(groups_v2) == len(groups_v2_parallel):
    print(f"\n✓ PASS: All three implementations produce {len(groups_v1)} groups")
    sys.exit(0)
else:
    print(f"\n✗ FAIL: Implementations produce different numbers of groups")
    sys.exit(1)
