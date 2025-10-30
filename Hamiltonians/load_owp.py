"""Load the OWP (Organic Water Peroxide) reactant Hamiltonian.

This loads the 44-qubit Hamiltonian from owp_reactant.npz
based on the notebook: https://github.com/rmlarose/calibrate-ibm/blob/main/owp_ibm.ipynb

Note: The notebook reports ~938k terms for the FermionOperator before Jordan-Wigner.
After Jordan-Wigner transformation, we get ~67k qubit operator terms.
"""

import numpy as np
from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner, count_qubits
import openfermion as of


def load_owp_hamiltonian(npz_path):
    """Load OWP Hamiltonian from .npz file.

    Args:
        npz_path: Path to owp_reactant.npz file

    Returns:
        tuple: (hamiltonian, nqubits, nterms)
            - hamiltonian: QubitOperator after Jordan-Wigner transformation
            - nqubits: Number of qubits (should be 44)
            - nterms: Number of terms in the Hamiltonian
    """
    # Load data
    data = np.load(npz_path)

    ECORE = float(data["ECORE"])
    H1 = data["H1"]
    H2 = data["H2"]
    NORB = int(data["NORB"])
    NELEC = int(data["NELEC"])

    print(f"Loaded OWP Hamiltonian:")
    print(f"  Core energy: {ECORE}")
    print(f"  Number of orbitals: {NORB}")
    print(f"  Number of electrons: {NELEC}")
    print(f"  H1 shape: {H1.shape}")
    print(f"  H2 shape: {H2.shape}")

    # Follow notebook approach: use InteractionOperator
    # First convert to spin-orbital basis
    H2_reordered = 0.5 * np.asarray(H2.transpose(0, 2, 3, 1), order="C")
    h1_spinorb, h2_spinorb = of.chem.molecular_data.spinorb_from_spatial(H1, H2_reordered)

    # Create InteractionOperator
    interaction_op = InteractionOperator(ECORE, h1_spinorb, h2_spinorb)

    # Convert to FermionOperator
    hamiltonian_ferm = get_fermion_operator(interaction_op)

    nterms_ferm = len(hamiltonian_ferm.terms)
    print(f"  FermionOperator terms: {nterms_ferm}")

    # Jordan-Wigner transformation
    print("Applying Jordan-Wigner transformation...")
    hamiltonian_qubit = jordan_wigner(hamiltonian_ferm)

    nterms = len(hamiltonian_qubit.terms)

    # Determine number of qubits (should be 2 * NORB = 44)
    nqubits = 2 * NORB

    print(f"  QubitOperator: {nqubits} qubits, {nterms} terms")

    return hamiltonian_qubit, nqubits, nterms


if __name__ == "__main__":
    import os

    npz_path = os.path.join(os.path.dirname(__file__), 'owp_reactant.npz')
    hamiltonian, nqubits, nterms = load_owp_hamiltonian(npz_path)

    print(f"\nFinal Hamiltonian: {nqubits} qubits, {nterms} terms")
