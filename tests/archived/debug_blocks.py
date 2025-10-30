"""Debug the blocks issue."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_terms_ordered_by_abscoeff
from Hamiltonians.load_h2o import load_h2o_hamiltonian
import kcommute.commute
from openfermion.transforms import qubit_operator_to_pauli_sum
import cirq

print("Loading H2O Hamiltonian...")
hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

terms_ord = get_terms_ordered_by_abscoeff(hamiltonian)
terms_ord = [term for term in terms_ord if () not in term.terms.keys()]

# The anticommuting pair we found
term1 = terms_ord[3]   # Z2
term2 = terms_ord[19]  # Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10

ps1 = next(iter(qubit_operator_to_pauli_sum(term1)))
ps2 = next(iter(qubit_operator_to_pauli_sum(term2)))

print(f"Term 1: {term1}")
print(f"Term 2: {term2}")
print(f"\nPauliSum 1: {ps1}")
print(f"PauliSum 2: {ps2}")
print(f"\nQubits in ps1: {ps1.qubits}")
print(f"Qubits in ps2: {ps2.qubits}")
print(f"Type of qubit: {type(list(ps1.qubits)[0])}")

# Test with integer blocks (what we're using)
blocks_int = [list(range(nqubits))]
print(f"\nBlocks (integers): {blocks_int}")
print(f"Type of block element: {type(blocks_int[0][0])}")

result_int = kcommute.commute.commutes(ps1, ps2, blocks=blocks_int)
print(f"Result with integer blocks: {result_int}")

# Test with cirq.LineQubit blocks (what it might expect)
blocks_cirq = [[cirq.LineQubit(i) for i in range(nqubits)]]
print(f"\nBlocks (cirq.LineQubit): {blocks_cirq}")
print(f"Type of block element: {type(blocks_cirq[0][0])}")

result_cirq = kcommute.commute.commutes(ps1, ps2, blocks=blocks_cirq)
print(f"Result with cirq.LineQubit blocks: {result_cirq}")

# Direct cirq.commutes check
direct_result = cirq.commutes(ps1, ps2)
print(f"\nDirect cirq.commutes(ps1, ps2): {direct_result}")
