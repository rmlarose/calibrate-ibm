"""Compare v5 and v6 for correctness and performance."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets_v5, get_si_sets_v6
from Hamiltonians.load_h2o import load_h2o_hamiltonian


def compare_groupings(groups1, groups2):
    """Check if two groupings are equivalent."""
    if len(groups1) != len(groups2):
        print(f"  ✗ Different number of groups: v5={len(groups1)} vs v6={len(groups2)}")
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


def test_and_time(hamiltonian, blocks, description, num_runs=3):
    """Test correctness and compare timing."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")

    nterms = len(hamiltonian.terms)
    print(f"Terms: {nterms}, Blocks: {len(blocks)} blocks")

    # Correctness check
    print(f"\nChecking correctness...")
    groups_v5 = get_si_sets_v5(hamiltonian, blocks, verbosity=0)
    groups_v6 = get_si_sets_v6(hamiltonian, blocks, verbosity=0)

    if not compare_groupings(groups_v5, groups_v6):
        print("CORRECTNESS FAILED - aborting timing")
        return None

    # Timing v5
    print(f"\nTiming v5 ({num_runs} runs)...")
    times_v5 = []
    for _ in range(num_runs):
        start = time.time()
        get_si_sets_v5(hamiltonian, blocks, verbosity=0)
        times_v5.append(time.time() - start)
    avg_v5 = sum(times_v5) / len(times_v5)
    print(f"  v5: {avg_v5:.4f}s")

    # Timing v6
    print(f"\nTiming v6 ({num_runs} runs)...")
    times_v6 = []
    for _ in range(num_runs):
        start = time.time()
        get_si_sets_v6(hamiltonian, blocks, verbosity=0)
        times_v6.append(time.time() - start)
    avg_v6 = sum(times_v6) / len(times_v6)
    print(f"  v6: {avg_v6:.4f}s")

    # Compare
    speedup = avg_v5 / avg_v6
    improvement = (avg_v5 - avg_v6) / avg_v5 * 100

    print(f"\n{'='*80}")
    print("Timing Comparison")
    print(f"{'='*80}")
    print(f"v5: {avg_v5:.4f}s")
    print(f"v6: {avg_v6:.4f}s")

    if speedup > 1.0:
        print(f"\n✓ v6 is FASTER: {speedup:.2f}x speedup ({improvement:.1f}% improvement)")
    elif speedup < 1.0:
        print(f"\n✗ v6 is SLOWER: {1/speedup:.2f}x slowdown ({-improvement:.1f}% worse)")
    else:
        print(f"\nNo significant difference")

    return {
        'description': description,
        'nterms': nterms,
        'avg_v5': avg_v5,
        'avg_v6': avg_v6,
        'speedup': speedup,
        'groups': len(groups_v5)
    }


if __name__ == "__main__":
    print("="*80)
    print("Testing v6 (NumPy arrays) vs v5")
    print("="*80)

    # Load H2O Hamiltonian
    hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
    hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

    print(f"\nH2O Hamiltonian: {nqubits} qubits, {nterms} terms")

    results = []

    # Test k=N (most important - single block fast path)
    blocks_kN = [list(range(nqubits))]
    results.append(test_and_time(hamiltonian, blocks_kN, "k=N (full commuting - CRITICAL)", num_runs=5))

    # Test k=4
    blocks_k4 = [list(range(i, min(i+4, nqubits))) for i in range(0, nqubits, 4)]
    results.append(test_and_time(hamiltonian, blocks_k4, "k=4", num_runs=3))

    # Test k=1
    blocks_k1 = [[i] for i in range(nqubits)]
    results.append(test_and_time(hamiltonian, blocks_k1, "k=1 (qubitwise)", num_runs=3))

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\n{'Configuration':<35} {'v5 (s)':<12} {'v6 (s)':<12} {'Speedup':<10}")
    print("-"*80)
    for r in results:
        if r:
            print(f"{r['description']:<35} {r['avg_v5']:<12.4f} {r['avg_v6']:<12.4f} {r['speedup']:<10.2f}x")

    if results:
        overall_speedup = sum(r['avg_v5'] for r in results if r) / sum(r['avg_v6'] for r in results if r)
        print("-"*80)
        print(f"Overall speedup: {overall_speedup:.2f}x")
