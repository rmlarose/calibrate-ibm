"""Debug k-commuting to find the issue."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openfermion import QubitOperator
from openfermion.transforms import qubit_operator_to_pauli_sum
import kcommute.commute
import cirq

# Create a simple test case for k=1 (qubitwise)
# X0 and X1 should k-commute for k=1 (they act on different qubits)
# X0 and Y0 should NOT k-commute for k=1 (they anticommute on qubit 0)

term1 = QubitOperator('X0', 1.0)
term2 = QubitOperator('X1', 1.0)
term3 = QubitOperator('Y0', 1.0)

blocks_k1 = [[0], [1]]

print("="*80)
print("Testing k=1 (qubitwise) commutation")
print("="*80)

# Convert to different representations
ps1 = next(iter(qubit_operator_to_pauli_sum(term1)))
ps2 = next(iter(qubit_operator_to_pauli_sum(term2)))
ps3 = next(iter(qubit_operator_to_pauli_sum(term3)))

x1, z1 = kcommute.commute.pauli_to_symplectic(term1)
x2, z2 = kcommute.commute.pauli_to_symplectic(term2)
x3, z3 = kcommute.commute.pauli_to_symplectic(term3)

print(f"\nTerm 1: X0")
print(f"  Symplectic: x={bin(x1)}, z={bin(z1)}")

print(f"\nTerm 2: X1")
print(f"  Symplectic: x={bin(x2)}, z={bin(z2)}")

print(f"\nTerm 3: Y0")
print(f"  Symplectic: x={bin(x3)}, z={bin(z3)}")

# Test X0 vs X1 (should k-commute for k=1)
print(f"\n{'='*80}")
print("X0 vs X1 (should k-commute for k=1 - different qubits)")
print(f"{'='*80}")

blocks_cirq = [[cirq.LineQubit(idx) for idx in block] for block in blocks_k1]
v2_result = kcommute.commute.commutes(ps1, ps2, blocks=blocks_cirq)
v3_result = kcommute.commute.commutes_symplectic(x1, z1, x2, z2, blocks_k1)

print(f"v2 (Cirq): {v2_result}")
print(f"v3 (symplectic): {v3_result}")
print(f"Match: {v2_result == v3_result}")

# Test X0 vs Y0 (should NOT k-commute for k=1)
print(f"\n{'='*80}")
print("X0 vs Y0 (should NOT k-commute for k=1 - anticommute on qubit 0)")
print(f"{'='*80}")

v2_result = kcommute.commute.commutes(ps1, ps3, blocks=blocks_cirq)
v3_result = kcommute.commute.commutes_symplectic(x1, z1, x3, z3, blocks_k1)

print(f"v2 (Cirq): {v2_result}")
print(f"v3 (symplectic): {v3_result}")
print(f"Match: {v2_result == v3_result}")

# Manual check of symplectic logic
print(f"\n{'='*80}")
print("Manual symplectic check for X0 vs Y0 with k=1")
print(f"{'='*80}")

anticommute_mask = (x1 & z3) ^ (z1 & x3)
print(f"Anticommute mask: {bin(anticommute_mask)} = {anticommute_mask}")

for block in blocks_k1:
    block_mask = 0
    for qubit_idx in block:
        block_mask |= (1 << qubit_idx)
    print(f"\nBlock {block}:")
    print(f"  Block mask: {bin(block_mask)} = {block_mask}")
    inner_product = anticommute_mask & block_mask
    print(f"  Inner product: {bin(inner_product)} = {inner_product}")
    parity = bin(inner_product).count('1') % 2
    print(f"  Parity (count of 1s mod 2): {parity}")
    print(f"  Commutes in this block: {parity == 0}")
