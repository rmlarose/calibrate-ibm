"""Debug why v2 produces more groups than v3."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openfermion import QubitOperator
from kcommute.sorted_insertion import get_si_sets_v2, get_si_sets_v3

# Load Hamiltonian
from Hamiltonians.load_h2o import load_h2o_hamiltonian

hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

# Use k=1
blocks_k1 = [[i] for i in range(nqubits)]

# Use a small subset to make debugging easier
small_ham = QubitOperator()
for i, (term, coeff) in enumerate(hamiltonian.terms.items()):
    if i < 100:
        small_ham += QubitOperator(term, coeff)

print(f"Testing with {len(small_ham.terms)} terms, k=1")

# Run both
groups_v2 = get_si_sets_v2(small_ham, blocks_k1, verbosity=0)
groups_v3 = get_si_sets_v3(small_ham, blocks_k1, verbosity=0)

print(f"\nv2: {len(groups_v2)} groups")
print(f"v3: {len(groups_v3)} groups")

# Find which terms are grouped differently
def term_to_str(qo):
    term = list(qo.terms.keys())[0]
    return str(term)

def find_group_for_term(term_str, groups):
    """Find which group contains a given term."""
    for i, group in enumerate(groups):
        for qo in group:
            if term_to_str(qo) == term_str:
                return i
    return None

# Get all unique terms
all_terms = set()
for group in groups_v2:
    for qo in group:
        all_terms.add(term_to_str(qo))

# Check if any terms that are in the same group in v3 are in different groups in v2
print(f"\nChecking for terms grouped differently...")

mismatches = []
for i, group_v3 in enumerate(groups_v3):
    if len(group_v3) > 1:
        # Get the v2 groups for all terms in this v3 group
        v2_groups_for_this = set()
        terms_in_group = []
        for qo in group_v3:
            term_str = term_to_str(qo)
            terms_in_group.append(term_str)
            v2_group_idx = find_group_for_term(term_str, groups_v2)
            v2_groups_for_this.add(v2_group_idx)

        if len(v2_groups_for_this) > 1:
            mismatches.append((i, terms_in_group, v2_groups_for_this))

print(f"Found {len(mismatches)} v3 groups that are split across multiple v2 groups")

if mismatches:
    print(f"\nFirst mismatch:")
    v3_group_idx, terms, v2_group_indices = mismatches[0]
    print(f"  v3 group {v3_group_idx} contains {len(terms)} terms")
    print(f"  These are split across v2 groups: {v2_group_indices}")
    print(f"  First few terms:")
    for term_str in terms[:5]:
        print(f"    {term_str}")
