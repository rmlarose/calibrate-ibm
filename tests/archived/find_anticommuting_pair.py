"""Find a pair of terms that don't commute."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_terms_ordered_by_abscoeff
from Hamiltonians.load_h2o import load_h2o_hamiltonian
import kcommute.commute
from openfermion.transforms import qubit_operator_to_pauli_sum

print("Loading H2O Hamiltonian...")
hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

terms_ord = get_terms_ordered_by_abscoeff(hamiltonian)
terms_ord = [term for term in terms_ord if () not in term.terms.keys()]

blocks = [list(range(nqubits))]

print(f"Searching {len(terms_ord)} terms for anticommuting pair...\n")

# Check first 20 terms against each other
for i in range(min(20, len(terms_ord))):
    term1 = terms_ord[i]
    x1, z1 = kcommute.commute.pauli_to_symplectic(term1)

    for j in range(i+1, min(20, len(terms_ord))):
        term2 = terms_ord[j]
        x2, z2 = kcommute.commute.pauli_to_symplectic(term2)

        result_symplectic = kcommute.commute.commutes_symplectic(x1, z1, x2, z2, blocks)

        if not result_symplectic:
            print(f"Found anticommuting pair!")
            print(f"Term {i}: {term1}")
            print(f"Term {j}: {term2}")

            # Now test with v2 approach
            ps1 = next(iter(qubit_operator_to_pauli_sum(term1)))
            ps2 = next(iter(qubit_operator_to_pauli_sum(term2)))

            support1 = set(ps1.qubits)
            support2 = set(ps2.qubits)

            print(f"\nSupport 1: {support1}")
            print(f"Support 2: {support2}")
            print(f"Disjoint: {support1.isdisjoint(support2)}")

            if support1.isdisjoint(support2):
                print(f"\nv2 bug: Disjoint supports but symplectic says they don't commute!")
                print(f"v2 would skip commutation check and assume they commute")
            else:
                result_v2 = kcommute.commute.commutes(ps1, ps2, blocks=blocks)
                print(f"\nv2 commutes result: {result_v2}")
                print(f"v3 commutes result: {result_symplectic}")
                if result_v2 != result_symplectic:
                    print(f"MISMATCH!")
            break
    else:
        continue
    break

print("\nDone.")
