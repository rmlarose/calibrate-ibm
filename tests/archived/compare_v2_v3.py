"""Compare v2 and v3 outputs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets_v2, get_si_sets_v3
from Hamiltonians.load_h2o import load_h2o_hamiltonian

print("Loading H2O Hamiltonian...")
hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

print(f"H2O Hamiltonian: {nqubits} qubits, {nterms} terms")

# Test with k=N (all qubits in one block)
blocks = [list(range(nqubits))]
print(f"\nBlocks (k=N): {blocks}")

print(f"\nRunning get_si_sets_v2...")
sets_v2 = get_si_sets_v2(hamiltonian, blocks, verbosity=0)
print(f"  v2: {len(sets_v2)} commuting groups")

print(f"\nRunning get_si_sets_v3...")
sets_v3 = get_si_sets_v3(hamiltonian, blocks, verbosity=0)
print(f"  v3: {len(sets_v3)} commuting groups")

if len(sets_v2) != len(sets_v3):
    print(f"\n⚠ WARNING: Different number of groups!")
    print(f"This suggests the algorithms are producing different results.")
else:
    print(f"\n✓ Both produced {len(sets_v2)} groups")
