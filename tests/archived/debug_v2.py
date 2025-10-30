"""Debug v2 commutation checking."""

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

print(f"H2O Hamiltonian: {nqubits} qubits, {nterms} terms")

terms_ord = get_terms_ordered_by_abscoeff(hamiltonian)
terms_ord = [term for term in terms_ord if () not in term.terms.keys()]

blocks = [list(range(nqubits))]

# Convert first few terms to PauliSum (v2 approach)
print(f"\nTesting v2 approach (PauliSum + cirq commutes):")
term1 = terms_ord[0]
term2 = terms_ord[1]

print(f"\nTerm 1: {term1}")
print(f"Term 2: {term2}")

ps1 = next(iter(qubit_operator_to_pauli_sum(term1)))
ps2 = next(iter(qubit_operator_to_pauli_sum(term2)))

print(f"\nPauliSum 1: {ps1}")
print(f"PauliSum 2: {ps2}")

support1 = set(ps1.qubits)
support2 = set(ps2.qubits)

print(f"\nSupport 1: {support1}")
print(f"Support 2: {support2}")
print(f"Disjoint supports: {support1.isdisjoint(support2)}")

if not support1.isdisjoint(support2):
    result = kcommute.commute.commutes(ps1, ps2, blocks=blocks)
    print(f"kcommute.commute.commutes result: {result}")

# Now test v3 approach (symplectic)
print(f"\n\nTesting v3 approach (symplectic):")
x1, z1 = kcommute.commute.pauli_to_symplectic(term1)
x2, z2 = kcommute.commute.pauli_to_symplectic(term2)

print(f"Term 1 symplectic: x={bin(x1)}, z={bin(z1)}")
print(f"Term 2 symplectic: x={bin(x2)}, z={bin(z2)}")

result_symplectic = kcommute.commute.commutes_symplectic(x1, z1, x2, z2, blocks)
print(f"commutes_symplectic result: {result_symplectic}")

print(f"\n\nDo they agree? {result == result_symplectic if not support1.isdisjoint(support2) else 'N/A (disjoint)'}")
