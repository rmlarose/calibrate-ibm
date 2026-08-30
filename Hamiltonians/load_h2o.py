"""Load the H2O Hamiltonian from HDF5 file.

This loads a water molecule Hamiltonian for testing purposes.
Uses OpenFermion's MolecularData to match the notebook approach.
Expected: 14 qubits, 1620 terms after Jordan-Wigner transformation.
"""

import openfermion as of


def load_h2o_hamiltonian(hdf5_path):
    """Load H2O Hamiltonian from HDF5 file.

    Args:
        hdf5_path: Path to .hdf5 file

    Returns:
        tuple: (hamiltonian, nqubits, nterms)
            - hamiltonian: QubitOperator after Jordan-Wigner transformation
            - nqubits: Number of qubits
            - nterms: Number of terms in the Hamiltonian
    """
    # Load using MolecularData with get_fermion_operator (matches repacking repo)
    hamiltonian = of.jordan_wigner(
        of.get_fermion_operator(
            of.chem.MolecularData(filename=hdf5_path).get_molecular_hamiltonian()
        )
    )

    nqubits = of.utils.count_qubits(hamiltonian)
    nterms = len(hamiltonian.terms)

    print(f"Loaded H2O Hamiltonian: {nqubits} qubits, {nterms} terms")

    return hamiltonian, nqubits, nterms


if __name__ == "__main__":
    import os

    hdf5_path = os.path.join(os.path.dirname(__file__), 'monomer_eqb.hdf5')
    hamiltonian, nqubits, nterms = load_h2o_hamiltonian(hdf5_path)

    # Test grouping
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from kcommute.sorted_insertion import get_si_sets_v5

    blocks_kN = [list(range(nqubits))]
    groups = get_si_sets_v5(hamiltonian, blocks_kN, verbosity=0)

    print(f"\nGrouping result: {len(groups)} groups (for k=N)")
