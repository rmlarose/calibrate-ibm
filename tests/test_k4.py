"""Test k=4 validation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets, get_si_sets_v2, get_terms_ordered_by_abscoeff
from kcommute.sorted_insertion_v2_parallel import get_si_sets_v2_parallel
from Hamiltonians.load_h2o import load_h2o_hamiltonian
from openfermion import qubit_operator_to_pauli_sum


def compare_commuting_sets(sets1, sets2):
    """Compare two lists of commuting sets for equality."""
    if len(sets1) != len(sets2):
        print(f"Different number of sets: {len(sets1)} vs {len(sets2)}")
        return False

    def sort_key(commset):
        return (len(commset), str(sorted([str(term) for term in commset])))

    sets1_sorted = sorted(sets1, key=sort_key)
    sets2_sorted = sorted(sets2, key=sort_key)

    for i, (set1, set2) in enumerate(zip(sets1_sorted, sets2_sorted)):
        if len(set1) != len(set2):
            print(f"Set {i}: Different lengths {len(set1)} vs {len(set2)}")
            return False

        terms1 = sorted([str(term) for term in set1])
        terms2 = sorted([str(term) for term in set2])

        if terms1 != terms2:
            print(f"Set {i}: Different terms")
            return False

    return True


print("="*80)
print("Testing k=4")
print("="*80)

hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

print(f"H2O Hamiltonian: {nqubits} qubits, {nterms} terms")

hamiltonian_cirq = qubit_operator_to_pauli_sum(hamiltonian)
cirq_qubits = sorted(hamiltonian_cirq.qubits)

blocks_cirq = [cirq_qubits[i:i+4] for i in range(0, nqubits, 4)]
blocks_int = [list(range(i, i+4)) for i in range(0, nqubits, 4)]
if nqubits % 4 != 0:
    blocks_int[-1] = list(range(blocks_int[-1][0], nqubits))
    blocks_cirq[-1] = cirq_qubits[blocks_cirq[-1][0].x:]

print(f"Blocks: {blocks_int}")
print(f"Running get_si_sets with k=4...")
sets1 = get_si_sets(hamiltonian, blocks_cirq, verbosity=0)

print(f"Running get_si_sets_v2 with k=4...")
sets2 = get_si_sets_v2(hamiltonian, blocks_int, verbosity=0)

print(f"Running get_si_sets_v2_parallel with k=4...")
terms = get_terms_ordered_by_abscoeff(hamiltonian)
terms = [t for t in terms if () not in t.terms.keys()]
sets3 = get_si_sets_v2_parallel(terms, blocks_int, verbosity=0, num_workers=2)

print(f"\nResults:")
print(f"  get_si_sets:             {len(sets1)} commuting groups")
print(f"  get_si_sets_v2:          {len(sets2)} commuting groups")
print(f"  get_si_sets_v2_parallel: {len(sets3)} commuting groups")

if compare_commuting_sets(sets1, sets2) and compare_commuting_sets(sets1, sets3):
    print("\n✓ PASS: All three implementations produce identical results for k=4")
    sys.exit(0)
else:
    print("\n✗ FAIL: Implementations produce different results for k=4")
    sys.exit(1)
