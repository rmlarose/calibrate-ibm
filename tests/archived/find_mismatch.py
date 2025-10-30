"""Find where v2 and v3 first disagree."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets_v2, get_si_sets_v3, get_terms_ordered_by_abscoeff
from Hamiltonians.load_h2o import load_h2o_hamiltonian
from openfermion.transforms import qubit_operator_to_pauli_sum
import kcommute.commute
import cirq

# Load Hamiltonian
hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

# Use k=1
blocks_k1 = [[i] for i in range(nqubits)]
blocks_cirq = [[cirq.LineQubit(idx) for idx in block] for block in blocks_k1]

print(f"Testing first 50 terms with k=1")
print(f"Blocks: {blocks_k1[:3]}...")

# Get ordered terms
terms_ord = get_terms_ordered_by_abscoeff(hamiltonian)
terms_ord = [term for term in terms_ord if () not in term.terms.keys()]

# Take first 50 terms
small_terms = terms_ord[:50]

# Convert to different representations
terms_ps = [next(iter(qubit_operator_to_pauli_sum(term))) for term in small_terms]
terms_symplectic = [kcommute.commute.pauli_to_symplectic(term) for term in small_terms]

# Check each pair
print(f"\nChecking pairwise commutation for first 10 terms:")
for i in range(min(10, len(small_terms))):
    term_i = list(small_terms[i].terms.keys())[0]
    print(f"\nTerm {i}: {term_i}")

    for j in range(i+1, min(10, len(small_terms))):
        term_j = list(small_terms[j].terms.keys())[0]

        # v2 check
        ps_i = terms_ps[i]
        ps_j = terms_ps[j]
        support_i = set(ps_i.qubits)
        support_j = set(ps_j.qubits)

        if support_i.isdisjoint(support_j):
            v2_commutes = True
            v2_reason = "disjoint"
        else:
            v2_commutes = kcommute.commute.commutes(ps_i, ps_j, blocks=blocks_cirq)
            v2_reason = "checked"

        # v3 check
        x_i, z_i = terms_symplectic[i]
        x_j, z_j = terms_symplectic[j]

        anticommute_mask = (x_i & z_j) ^ (z_i & x_j)
        if anticommute_mask == 0:
            v3_commutes = True
            v3_reason = "1-commute"
        else:
            v3_commutes = kcommute.commute.commutes_symplectic(x_i, z_i, x_j, z_j, blocks_k1)
            v3_reason = "checked"

        if v2_commutes != v3_commutes:
            print(f"  ✗ MISMATCH vs term {j} ({term_j}):")
            print(f"    v2: {v2_commutes} ({v2_reason})")
            print(f"    v3: {v3_commutes} ({v3_reason})")
            print(f"    Support i: {support_i}")
            print(f"    Support j: {support_j}")
            print(f"    Symplectic i: x={bin(x_i)}, z={bin(z_i)}")
            print(f"    Symplectic j: x={bin(x_j)}, z={bin(z_j)}")
            print(f"    Anticommute mask: {bin(anticommute_mask)}")
