"""Compare v5, v5_parallel, and v5_parallel_v2 for correctness and performance."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets_v5, get_si_sets_v5_parallel, get_si_sets_v5_parallel_v2
from Hamiltonians.load_h2o import load_h2o_hamiltonian


def compare_groupings(groups1, groups2, name1, name2):
    """Check if two groupings are equivalent."""
    if len(groups1) != len(groups2):
        print(f"  ✗ Different number of groups: {name1}={len(groups1)} vs {name2}={len(groups2)}")
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
    print("Testing v5_parallel_v2 (multi-worker) vs v5_parallel vs v5")
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

    print("\nRunning v5_parallel (1 worker)...")
    start = time.time()
    groups_v5_parallel = get_si_sets_v5_parallel(hamiltonian, blocks_kN, verbosity=0)
    time_v5_parallel = time.time() - start
    print(f"v5_parallel: {len(groups_v5_parallel)} groups in {time_v5_parallel:.4f}s")

    print("\nRunning v5_parallel_v2 (2 workers)...")
    start = time.time()
    groups_v5_parallel_v2 = get_si_sets_v5_parallel_v2(hamiltonian, blocks_kN, verbosity=1, num_workers=2)
    time_v5_parallel_v2 = time.time() - start
    print(f"v5_parallel_v2: {len(groups_v5_parallel_v2)} groups in {time_v5_parallel_v2:.4f}s")

    print("\nRunning v5_parallel_v2 (4 workers)...")
    start = time.time()
    groups_v5_parallel_v2_4w = get_si_sets_v5_parallel_v2(hamiltonian, blocks_kN, verbosity=1, num_workers=4)
    time_v5_parallel_v2_4w = time.time() - start
    print(f"v5_parallel_v2 (4w): {len(groups_v5_parallel_v2_4w)} groups in {time_v5_parallel_v2_4w:.4f}s")

    print("\nComparing results...")
    print("\nv5 vs v5_parallel:")
    correct_parallel = compare_groupings(groups_v5, groups_v5_parallel, "v5", "v5_parallel")

    print("\nv5 vs v5_parallel_v2 (2 workers):")
    correct_v2_2w = compare_groupings(groups_v5, groups_v5_parallel_v2, "v5", "v5_parallel_v2")

    print("\nv5 vs v5_parallel_v2 (4 workers):")
    correct_v2_4w = compare_groupings(groups_v5, groups_v5_parallel_v2_4w, "v5", "v5_parallel_v2_4w")

    if correct_parallel and correct_v2_2w and correct_v2_4w:
        print("\n✓ ALL CORRECTNESS VERIFIED")
    else:
        print("\n✗ CORRECTNESS FAILED")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("Performance Comparison")
    print(f"{'='*80}")
    print(f"\n{'Version':<30} {'Time (s)':<12} {'Speedup vs v5':<15}")
    print("-"*80)
    print(f"{'v5 (sequential)':<30} {time_v5:<12.4f} {'1.00x':<15}")

    speedup_parallel = time_v5 / time_v5_parallel
    print(f"{'v5_parallel (1 worker)':<30} {time_v5_parallel:<12.4f} {f'{speedup_parallel:.2f}x':<15}")

    speedup_v2_2w = time_v5 / time_v5_parallel_v2
    print(f"{'v5_parallel_v2 (2 workers)':<30} {time_v5_parallel_v2:<12.4f} {f'{speedup_v2_2w:.2f}x':<15}")

    speedup_v2_4w = time_v5 / time_v5_parallel_v2_4w
    print(f"{'v5_parallel_v2 (4 workers)':<30} {time_v5_parallel_v2_4w:<12.4f} {f'{speedup_v2_4w:.2f}x':<15}")

    print("-"*80)

    print(f"\n{'='*80}")
    print("Multi-worker Analysis")
    print(f"{'='*80}")
    print(f"\nv5_parallel → v5_parallel_v2 (2 workers):")
    speedup_1to2 = time_v5_parallel / time_v5_parallel_v2
    print(f"  {speedup_1to2:.2f}x speedup")

    print(f"\nv5_parallel_v2: 2 workers → 4 workers:")
    speedup_2to4 = time_v5_parallel_v2 / time_v5_parallel_v2_4w
    print(f"  {speedup_2to4:.2f}x speedup")

    if speedup_v2_4w > speedup_v2_2w:
        print(f"\n✓ Scaling with more workers!")
    else:
        print(f"\n✗ Not scaling well (overhead may dominate for small problem)")
