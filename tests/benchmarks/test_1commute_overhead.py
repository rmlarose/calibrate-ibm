"""Test whether 1-commuting check adds overhead or speeds things up."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_terms_ordered_by_abscoeff
from Hamiltonians.load_h2o import load_h2o_hamiltonian
import kcommute.commute
from openfermion import QubitOperator

def get_si_sets_v3_no_fast_path(op, blocks, verbosity: int = 0):
    '''v3 without 1-commuting fast path - always does full k-commuting check.'''
    nterms = len(op.terms)
    assert isinstance(op, QubitOperator)
    commuting_sets = []
    terms_ord = get_terms_ordered_by_abscoeff(op)
    terms_ord = [term for term in terms_ord if () not in term.terms.keys()]
    terms_symplectic = [kcommute.commute.pauli_to_symplectic(term) for term in terms_ord]
    terms_supports = [x_bits | z_bits for x_bits, z_bits in terms_symplectic]

    for i, (pstring, (x_bits, z_bits), support_mask) in enumerate(zip(terms_ord, terms_symplectic, terms_supports)):
        if verbosity > 0:
            print(f"Status: On Pauli string {i} / {nterms}", end="\r")
        found_commuting_set = False

        for commset in commuting_sets:
            all_strings_in_commset_commute = True
            for pstring2, (x2, z2), support_mask2 in commset:
                # NO fast path - always do full check
                if not kcommute.commute.commutes_symplectic(x_bits, z_bits, x2, z2, blocks):
                    all_strings_in_commset_commute = False
                    break
            if all_strings_in_commset_commute:
                found_commuting_set = True
                commset.append((pstring, (x_bits, z_bits), support_mask))
                break

        if not found_commuting_set:
            commuting_sets.append([(pstring, (x_bits, z_bits), support_mask)])

    return [[pstring for pstring, _, support_mask in commset] for commset in commuting_sets]


def get_si_sets_v3_with_fast_path(op, blocks, verbosity: int = 0):
    '''v3 with 1-commuting fast path.'''
    nterms = len(op.terms)
    assert isinstance(op, QubitOperator)
    commuting_sets = []
    terms_ord = get_terms_ordered_by_abscoeff(op)
    terms_ord = [term for term in terms_ord if () not in term.terms.keys()]
    terms_symplectic = [kcommute.commute.pauli_to_symplectic(term) for term in terms_ord]
    terms_supports = [x_bits | z_bits for x_bits, z_bits in terms_symplectic]

    for i, (pstring, (x_bits, z_bits), support_mask) in enumerate(zip(terms_ord, terms_symplectic, terms_supports)):
        if verbosity > 0:
            print(f"Status: On Pauli string {i} / {nterms}", end="\r")
        found_commuting_set = False

        for commset in commuting_sets:
            all_strings_in_commset_commute = True
            for pstring2, (x2, z2), support_mask2 in commset:
                anticommute_mask = (x_bits & z2) ^ (z_bits & x2)
                if not (anticommute_mask == 0 or kcommute.commute.commutes_symplectic(x_bits, z_bits, x2, z2, blocks)):
                    all_strings_in_commset_commute = False
                    break
            if all_strings_in_commset_commute:
                found_commuting_set = True
                commset.append((pstring, (x_bits, z_bits), support_mask))
                break

        if not found_commuting_set:
            commuting_sets.append([(pstring, (x_bits, z_bits), support_mask)])

    return [[pstring for pstring, _, support_mask in commset] for commset in commuting_sets]


print("="*80)
print("Testing 1-commuting fast path overhead")
print("="*80)

hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

print(f"\nH2O Hamiltonian: {nqubits} qubits, {nterms} terms")

blocks = [list(range(nqubits))]
print(f"Blocks (k=N): {blocks}")

# Test without fast path
print(f"\n\nTiming WITHOUT 1-commuting fast path...")
start = time.time()
sets_no_fast = get_si_sets_v3_no_fast_path(hamiltonian, blocks, verbosity=0)
time_no_fast = time.time() - start
print(f"  Time: {time_no_fast:.4f} seconds")
print(f"  Groups: {len(sets_no_fast)}")

# Test with fast path
print(f"\nTiming WITH 1-commuting fast path...")
start = time.time()
sets_with_fast = get_si_sets_v3_with_fast_path(hamiltonian, blocks, verbosity=0)
time_with_fast = time.time() - start
print(f"  Time: {time_with_fast:.4f} seconds")
print(f"  Groups: {len(sets_with_fast)}")

print("\n" + "="*80)
print("Comparison")
print("="*80)
print(f"Without fast path: {time_no_fast:.4f} seconds")
print(f"With fast path:    {time_with_fast:.4f} seconds")

if time_no_fast > time_with_fast:
    improvement = (time_no_fast - time_with_fast) / time_no_fast * 100
    speedup = time_no_fast / time_with_fast
    print(f"\nFast path is BENEFICIAL: {improvement:.1f}% faster ({speedup:.2f}x speedup)")
elif time_with_fast > time_no_fast:
    overhead = (time_with_fast - time_no_fast) / time_no_fast * 100
    slowdown = time_with_fast / time_no_fast
    print(f"\nFast path adds OVERHEAD: {overhead:.1f}% slower ({slowdown:.2f}x slowdown)")
else:
    print(f"\nNo significant difference")
