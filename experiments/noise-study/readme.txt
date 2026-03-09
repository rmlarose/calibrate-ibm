Each file in this directory corresponds to a fixed target accuracy. This accuracy is achieved by circuits in different basis sets: STO-3G, 6-31G and 6-311G (for details see h2o-basis-qubits.txt and ccsd_t-water.txt).

For the energies to be correct, one must use the remapped Hamiltonian included in the corresponding folder as a pickled QubitOperator.

The readme files in each folder specify the energy and energy error associated with each circuit.