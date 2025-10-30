"""Unit tests to verify get_si_sets and get_si_sets_v2 produce identical results."""

import os
import sys

# Add parent directory to path to import kcommute
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets, get_si_sets_v2
from Hamiltonians.load_h2o import load_h2o_hamiltonian


def compare_commuting_sets(sets1, sets2):
    """Compare two lists of commuting sets for equality.

    Args:
        sets1: First list of commuting sets
        sets2: Second list of commuting sets

    Returns:
        True if the sets are identical, False otherwise
    """
    if len(sets1) != len(sets2):
        print(f"Different number of sets: {len(sets1)} vs {len(sets2)}")
        return False

    # Sort both lists by length and number of terms for consistent comparison
    def sort_key(commset):
        return (len(commset), str(sorted([str(term) for term in commset])))

    sets1_sorted = sorted(sets1, key=sort_key)
    sets2_sorted = sorted(sets2, key=sort_key)

    for i, (set1, set2) in enumerate(zip(sets1_sorted, sets2_sorted)):
        if len(set1) != len(set2):
            print(f"Set {i}: Different lengths {len(set1)} vs {len(set2)}")
            return False

        # Convert to strings for comparison
        terms1 = sorted([str(term) for term in set1])
        terms2 = sorted([str(term) for term in set2])

        if terms1 != terms2:
            print(f"Set {i}: Different terms")
            return False

    return True


def test_get_si_sets_v2_k1():
    """Test that get_si_sets and get_si_sets_v2 produce identical results for k=1."""
    print("\n" + "="*80)
    print("Testing k=1 (single qubit commutation)")
    print("="*80)

    # Load H2O Hamiltonian
    hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
    hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

    print(f"H2O Hamiltonian: {nqubits} qubits, {nterms} terms")

    # Run both versions with k=1
    blocks = [[i] for i in range(nqubits)]
    print(f"Running get_si_sets with k=1...")
    sets1 = get_si_sets(hamiltonian, blocks, verbosity=0)

    print(f"Running get_si_sets_v2 with k=1...")
    sets2 = get_si_sets_v2(hamiltonian, blocks, verbosity=0)

    print(f"\nResults:")
    print(f"  get_si_sets:    {len(sets1)} commuting groups")
    print(f"  get_si_sets_v2: {len(sets2)} commuting groups")

    # Compare results
    if compare_commuting_sets(sets1, sets2):
        print("\n✓ PASS: Both functions produce identical results for k=1")
        return True
    else:
        print("\n✗ FAIL: Functions produce different results for k=1")
        return False


def test_get_si_sets_v2_k4():
    """Test that get_si_sets and get_si_sets_v2 produce identical results for k=4."""
    print("\n" + "="*80)
    print("Testing k=4")
    print("="*80)

    # Load H2O Hamiltonian
    hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
    hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

    print(f"H2O Hamiltonian: {nqubits} qubits, {nterms} terms")

    # Run both versions with k=4
    blocks = [list(range(i, i+4)) for i in range(0, nqubits, 4)]
    # Handle edge case if nqubits not divisible by 4
    if nqubits % 4 != 0:
        blocks[-1] = list(range(blocks[-1][0], nqubits))

    print(f"Blocks: {blocks}")
    print(f"Running get_si_sets with k=4...")
    sets1 = get_si_sets(hamiltonian, blocks, verbosity=0)

    print(f"Running get_si_sets_v2 with k=4...")
    sets2 = get_si_sets_v2(hamiltonian, blocks, verbosity=0)

    print(f"\nResults:")
    print(f"  get_si_sets:    {len(sets1)} commuting groups")
    print(f"  get_si_sets_v2: {len(sets2)} commuting groups")

    # Compare results
    if compare_commuting_sets(sets1, sets2):
        print("\n✓ PASS: Both functions produce identical results for k=4")
        return True
    else:
        print("\n✗ FAIL: Functions produce different results for k=4")
        return False


def test_get_si_sets_v2_kN():
    """Test that get_si_sets and get_si_sets_v2 produce identical results for k=N."""
    print("\n" + "="*80)
    print("Testing k=N (all qubits in one block)")
    print("="*80)

    # Load H2O Hamiltonian
    hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
    hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

    print(f"H2O Hamiltonian: {nqubits} qubits, {nterms} terms")

    # Run both versions with k=N (all qubits in one block)
    blocks = [list(range(nqubits))]
    print(f"Blocks: {blocks}")
    print(f"Running get_si_sets with k=N...")
    sets1 = get_si_sets(hamiltonian, blocks, verbosity=0)

    print(f"Running get_si_sets_v2 with k=N...")
    sets2 = get_si_sets_v2(hamiltonian, blocks, verbosity=0)

    print(f"\nResults:")
    print(f"  get_si_sets:    {len(sets1)} commuting groups")
    print(f"  get_si_sets_v2: {len(sets2)} commuting groups")

    # Compare results
    if compare_commuting_sets(sets1, sets2):
        print("\n✓ PASS: Both functions produce identical results for k=N")
        return True
    else:
        print("\n✗ FAIL: Functions produce different results for k=N")
        return False


if __name__ == "__main__":
    print("="*80)
    print("Unit Tests: get_si_sets vs get_si_sets_v2")
    print("="*80)

    results = []

    # Run all tests
    results.append(("k=1", test_get_si_sets_v2_k1()))
    results.append(("k=4", test_get_si_sets_v2_k4()))
    results.append(("k=N", test_get_si_sets_v2_kN()))

    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + "="*80)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("="*80)

    sys.exit(0 if all_passed else 1)
