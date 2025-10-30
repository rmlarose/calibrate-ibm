"""Compare v3 and v4 implementations for correctness."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kcommute.sorted_insertion import get_si_sets_v3, get_si_sets_v4
from Hamiltonians.load_h2o import load_h2o_hamiltonian
from openfermion import QubitOperator


def compare_groupings(groups1, groups2):
    """Check if two groupings are equivalent (same terms in same groups).

    Args:
        groups1: List of groups from first algorithm
        groups2: List of groups from second algorithm

    Returns:
        True if groupings are identical, False otherwise
    """
    if len(groups1) != len(groups2):
        print(f"Different number of groups: {len(groups1)} vs {len(groups2)}")
        return False

    # Convert each group to a set of terms for comparison
    def group_to_set(group):
        term_set = set()
        for qo in group:
            # Extract the single term from the QubitOperator
            term = list(qo.terms.keys())[0]
            coeff = qo.terms[term]
            term_set.add((term, coeff))
        return term_set

    groups1_sets = [group_to_set(g) for g in groups1]
    groups2_sets = [group_to_set(g) for g in groups2]

    # Sort for comparison
    groups1_sets.sort(key=lambda s: (len(s), sorted(s)))
    groups2_sets.sort(key=lambda s: (len(s), sorted(s)))

    for i, (g1, g2) in enumerate(zip(groups1_sets, groups2_sets)):
        if g1 != g2:
            print(f"Group {i} differs:")
            print(f"  v3: {len(g1)} terms")
            print(f"  v4: {len(g2)} terms")
            return False

    return True


def test_with_hamiltonian(hamiltonian, blocks, name="Test"):
    """Test v3 vs v4 with a given Hamiltonian."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    nterms = len(hamiltonian.terms)
    print(f"Terms: {nterms}, Blocks: {blocks}")

    # Run v3
    print("\nRunning v3...")
    groups_v3 = get_si_sets_v3(hamiltonian, blocks, verbosity=0)
    print(f"v3: {len(groups_v3)} groups")

    # Run v4
    print("Running v4...")
    groups_v4 = get_si_sets_v4(hamiltonian, blocks, verbosity=0)
    print(f"v4: {len(groups_v4)} groups")

    # Compare
    print("\nComparing groupings...")
    if compare_groupings(groups_v3, groups_v4):
        print("✓ Groupings are IDENTICAL")
        return True
    else:
        print("✗ Groupings DIFFER")
        return False


if __name__ == "__main__":
    print("="*80)
    print("Testing v3 vs v4 Correctness")
    print("="*80)

    # Load H2O Hamiltonian
    hdf5_path = os.path.join(os.path.dirname(__file__), '..', 'Hamiltonians', 'monomer_eqb.hdf5')
    hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

    print(f"\nH2O Hamiltonian: {nqubits} qubits, {nterms} terms")

    all_passed = True

    # Test with k=N (full commuting)
    blocks_kN = [list(range(nqubits))]
    all_passed &= test_with_hamiltonian(hamiltonian, blocks_kN, "Test 1: k=N (full)")

    # Test with k=1 (qubitwise)
    blocks_k1 = [[i] for i in range(nqubits)]
    all_passed &= test_with_hamiltonian(hamiltonian, blocks_k1, "Test 2: k=1 (qubitwise)")

    # Test with k=4
    blocks_k4 = [list(range(i, min(i+4, nqubits))) for i in range(0, nqubits, 4)]
    all_passed &= test_with_hamiltonian(hamiltonian, blocks_k4, "Test 3: k=4")

    # Test with small subset
    small_ham = QubitOperator()
    for i, (term, coeff) in enumerate(hamiltonian.terms.items()):
        if i < 100:  # First 100 terms
            small_ham += QubitOperator(term, coeff)
    all_passed &= test_with_hamiltonian(small_ham, blocks_kN, "Test 4: Small subset (100 terms)")

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    if all_passed:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED")
