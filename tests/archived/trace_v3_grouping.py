"""Trace v3 grouping to see why terms are grouped incorrectly."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openfermion import QubitOperator
from kcommute.sorted_insertion import get_terms_ordered_by_abscoeff
import kcommute.commute

# Create a small test with known issue
term1 = QubitOperator('Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12', 1.0)
term2 = QubitOperator('X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Z13 X13', 1.0)  # disjoint from term1
term3 = QubitOperator('X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12', 1.0)  # should NOT commute with term1

test_ham = term1 + term2 + term3

blocks_k1 = [[i] for i in range(14)]

print("Test Hamiltonian:")
for i, (term, coeff) in enumerate(test_ham.terms.items()):
    print(f"  {i}: {term}")

# Manually run SI algorithm with debug output
terms_ord = get_terms_ordered_by_abscoeff(test_ham)
terms_ord = [term for term in terms_ord if () not in term.terms.keys()]

terms_symplectic = [kcommute.commute.pauli_to_symplectic(term) for term in terms_ord]
terms_supports = [x_bits | z_bits for x_bits, z_bits in terms_symplectic]

# Pre-compute block mask
block_mask = 0
for block in blocks_k1:
    for qubit_idx in block:
        block_mask |= (1 << qubit_idx)

print(f"\nBlock mask: {bin(block_mask)}")

commuting_sets = []

for i, (pstring, (x_bits, z_bits), support_mask) in enumerate(zip(terms_ord, terms_symplectic, terms_supports)):
    term = list(pstring.terms.keys())[0]
    print(f"\n{'='*80}")
    print(f"Processing term {i}: {term}")
    print(f"  Symplectic: x={bin(x_bits)}, z={bin(z_bits)}")
    print(f"  Support: {bin(support_mask)}")

    found_commuting_set = False

    for j, commset in enumerate(commuting_sets):
        print(f"\n  Checking against group {j} ({len(commset)} terms)...")

        all_strings_in_commset_commute = True
        for k, (pstring2, (x2, z2), support_mask2) in enumerate(commset):
            term2 = list(pstring2.terms.keys())[0]
            anticommute_mask = (x_bits & z2) ^ (z_bits & x2)

            print(f"    vs term {k} ({term2}):")
            print(f"      Anticommute mask: {bin(anticommute_mask)}")

            if anticommute_mask == 0:
                print(f"      -> 1-commutes (fast path)")
                continue

            masked_anticommute = anticommute_mask & block_mask
            parity = bin(masked_anticommute).count('1') % 2

            print(f"      Masked anticommute: {bin(masked_anticommute)}")
            print(f"      Parity: {parity}")

            if parity != 0:
                print(f"      -> Does NOT k-commute")
                all_strings_in_commset_commute = False
                break
            else:
                print(f"      -> k-commutes")

        if all_strings_in_commset_commute:
            print(f"\n  ✓ Adding to group {j}")
            found_commuting_set = True
            commset.append((pstring, (x_bits, z_bits), support_mask))
            break
        else:
            print(f"\n  ✗ Does not fit in group {j}")

    if not found_commuting_set:
        print(f"\n  Creating new group {len(commuting_sets)}")
        commuting_sets.append([(pstring, (x_bits, z_bits), support_mask)])

print(f"\n{'='*80}")
print(f"Final: {len(commuting_sets)} groups")
for i, group in enumerate(commuting_sets):
    print(f"  Group {i}: {len(group)} terms")
