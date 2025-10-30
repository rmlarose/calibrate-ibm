"""Quick test of get_si_sets_v2 only."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets_v2
from Hamiltonians.load_h2o import load_h2o_hamiltonian

print("Loading H2O Hamiltonian...")
hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

print(f"H2O Hamiltonian: {nqubits} qubits, {nterms} terms")

# Test with k=N (all qubits in one block)
blocks = [list(range(nqubits))]
print(f"\nTesting get_si_sets_v2 with k=N...")
print(f"Blocks: {blocks}")

try:
    sets = get_si_sets_v2(hamiltonian, blocks, verbosity=0)
    print(f"\n✓ SUCCESS: get_si_sets_v2 completed!")
    print(f"  Number of commuting groups: {len(sets)}")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
