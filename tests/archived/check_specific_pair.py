"""Check if a specific pair actually k-commutes."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openfermion import QubitOperator
from openfermion.transforms import qubit_operator_to_pauli_sum
import kcommute.commute
import cirq

# Two terms from the v3 group
term1 = QubitOperator('Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12', 1.0)
term2 = QubitOperator('X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12', 1.0)

blocks_k1 = [[i] for i in range(14)]
blocks_cirq = [[cirq.LineQubit(idx) for idx in block] for block in blocks_k1]

print("Term 1: Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12")
print("Term 2: X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12")
print(f"\nBlocks (k=1): {blocks_k1[:5]}...")

# v2 check
ps1 = next(iter(qubit_operator_to_pauli_sum(term1)))
ps2 = next(iter(qubit_operator_to_pauli_sum(term2)))

print(f"\nv2 (Cirq):")
support1 = set(ps1.qubits)
support2 = set(ps2.qubits)
print(f"  Support 1: {sorted([q.x for q in support1])}")
print(f"  Support 2: {sorted([q.x for q in support2])}")
print(f"  Disjoint: {support1.isdisjoint(support2)}")

v2_commutes = kcommute.commute.commutes(ps1, ps2, blocks=blocks_cirq)
print(f"  k-commutes (v2): {v2_commutes}")

# v3 check
x1, z1 = kcommute.commute.pauli_to_symplectic(term1)
x2, z2 = kcommute.commute.pauli_to_symplectic(term2)

print(f"\nv3 (symplectic):")
print(f"  Symplectic 1: x={bin(x1)}, z={bin(z1)}")
print(f"  Symplectic 2: x={bin(x2)}, z={bin(z2)}")

anticommute_mask = (x1 & z2) ^ (z1 & x2)
print(f"  Anticommute mask: {bin(anticommute_mask)}")
print(f"  1-commutes: {anticommute_mask == 0}")

v3_commutes = kcommute.commute.commutes_symplectic(x1, z1, x2, z2, blocks_k1)
print(f"  k-commutes (v3): {v3_commutes}")

print(f"\n{'='*80}")
if v2_commutes == v3_commutes:
    print(f"✓ v2 and v3 AGREE: {v2_commutes}")
else:
    print(f"✗ v2 and v3 DISAGREE: v2={v2_commutes}, v3={v3_commutes}")

# Manual check for each qubit
print(f"\n{'='*80}")
print("Manual check per qubit:")
print(f"{'='*80}")

term1_dict = dict(list(term1.terms.keys())[0])
term2_dict = dict(list(term2.terms.keys())[0])

for qubit in range(14):
    p1 = term1_dict.get(qubit, 'I')
    p2 = term2_dict.get(qubit, 'I')

    # Check if they commute on this qubit
    commutes_on_qubit = True
    if (p1, p2) in [('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y')]:
        commutes_on_qubit = False

    print(f"  Qubit {qubit:2d}: {p1} vs {p2} -> {'commute' if commutes_on_qubit else 'ANTICOMMUTE'}")
