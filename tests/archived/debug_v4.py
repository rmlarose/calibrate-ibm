"""Debug v4 to understand why it differs from v3."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets_v3, get_si_sets_v4, get_terms_ordered_by_abscoeff
from openfermion import QubitOperator
import kcommute.commute

# Create a simple test case
test_op = QubitOperator()
test_op += QubitOperator('X0 X1', 1.0)  # XXII
test_op += QubitOperator('X0 Y1', 0.9)  # XYII - should commute with above
test_op += QubitOperator('Z0 Y1', 0.8)  # ZYII - k-commutes but not 1-commutes with above
test_op += QubitOperator('X0 X1', 0.7)  # XXII - exact duplicate, should go in first group

blocks = [[0, 1, 2, 3]]  # k=4

print("Test Hamiltonian:")
for term, coeff in test_op.terms.items():
    print(f"  {term}: {coeff}")

print("\n" + "="*80)
print("Running v3...")
groups_v3 = get_si_sets_v3(test_op, blocks, verbosity=0)
print(f"v3 produced {len(groups_v3)} groups:")
for i, group in enumerate(groups_v3):
    print(f"  Group {i+1}:")
    for term_op in group:
        term = list(term_op.terms.keys())[0]
        coeff = term_op.terms[term]
        print(f"    {term}: {coeff}")

print("\n" + "="*80)
print("Running v4...")
groups_v4 = get_si_sets_v4(test_op, blocks, verbosity=0)
print(f"v4 produced {len(groups_v4)} groups:")
for i, group in enumerate(groups_v4):
    print(f"  Group {i+1}:")
    for term_op in group:
        term = list(term_op.terms.keys())[0]
        coeff = term_op.terms[term]
        print(f"    {term}: {coeff}")

# Verify commutation manually
print("\n" + "="*80)
print("Manual verification:")
terms_ord = get_terms_ordered_by_abscoeff(test_op)
terms_ord = [term for term in terms_ord if () not in term.terms.keys()]
terms_symplectic = [kcommute.commute.pauli_to_symplectic(term) for term in terms_ord]

for i, (t1, (x1, z1)) in enumerate(zip(terms_ord, terms_symplectic)):
    term1 = list(t1.terms.keys())[0]
    print(f"\nTerm {i}: {term1} -> x={bin(x1)}, z={bin(z1)}")
    for j, (t2, (x2, z2)) in enumerate(zip(terms_ord, terms_symplectic)):
        if i >= j:
            continue
        term2 = list(t2.terms.keys())[0]
        anticommute = (x1 & z2) ^ (z1 & x2)
        k_commutes = kcommute.commute.commutes_symplectic(x1, z1, x2, z2, blocks)
        print(f"  vs Term {j} {term2}: anticommute_mask={bin(anticommute)}, k_commutes={k_commutes}")
