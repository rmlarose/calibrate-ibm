"""Compare v5 and v5_parallel for correctness and performance."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets_v5, get_si_sets_v5_parallel
from Hamiltonians.load_h2o import load_h2o_hamiltonian


def compare_groupings(groups1, groups2):
    """Check if two groupings are equivalent."""
    if len(groups1) != len(groups2):
        print(f"  ✗ Different number of groups: v5={len(groups1)} vs v5_parallel={len(groups2)}")
        return False

    def group_to_set(group):
        term_set = set()
        for qo in group:
            term = list(qo.terms.keys())[0]
            coeff = qo.terms[term]
            term_set.add((term, coeff))
        return term_set

    groups1_sets = [group_to_set(g) for g in groups1]
    groups2_sets = [group_to_set(g) for g in groups2]

    groups1_sets.sort(key=lambda s: (len(s), sorted(s)))
    groups2_sets.sort(key=lambda s: (len(s), sorted(s)))

    for i, (g1, g2) in enumerate(zip(groups1_sets, groups2_sets)):
        if g1 != g2:
            print(f"  ✗ Group {i} differs")
            return False

    print(f"  ✓ Groupings are IDENTICAL")
    return True


if __name__ == "__main__":
    print("="*80)
    print("Testing v5_parallel vs v5")
    print("="*80)

    # Load H2O Hamiltonian
    hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
    hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

    print(f"\nH2O Hamiltonian: {nqubits} qubits, {nterms} terms")

    blocks_kN = [list(range(nqubits))]

    print(f"\n{'='*80}")
    print("Correctness Test (k=N)")
    print(f"{'='*80}")

    print("\nRunning v5...")
    start = time.time()
    groups_v5 = get_si_sets_v5(hamiltonian, blocks_kN, verbosity=0)
    time_v5 = time.time() - start
    print(f"v5: {len(groups_v5)} groups in {time_v5:.4f}s")

    print("\nRunning v5_parallel...")
    start = time.time()
    groups_v5_parallel = get_si_sets_v5_parallel(hamiltonian, blocks_kN, verbosity=1)
    time_v5_parallel = time.time() - start
    print(f"v5_parallel: {len(groups_v5_parallel)} groups in {time_v5_parallel:.4f}s")

    print("\nComparing results...")
    if compare_groupings(groups_v5, groups_v5_parallel):
        print("\n✓ CORRECTNESS VERIFIED")
    else:
        print("\n✗ CORRECTNESS FAILED")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("Performance Comparison")
    print(f"{'='*80}")
    print(f"v5:          {time_v5:.4f}s")
    print(f"v5_parallel: {time_v5_parallel:.4f}s")

    if time_v5_parallel < time_v5:
        speedup = time_v5 / time_v5_parallel
        print(f"\n✓ v5_parallel is {speedup:.2f}x FASTER")
    else:
        slowdown = time_v5_parallel / time_v5
        print(f"\n✗ v5_parallel is {slowdown:.2f}x SLOWER (expected for small problem)")
        print(f"  Overhead dominates for H2O's 1620 terms")
