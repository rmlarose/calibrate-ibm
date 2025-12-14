# Q4Bio hardware runs and results

Files are organized as

```text
experiments/
  experiment_name/
    circuits/      # Usually .qasm files of circuit(s). Sometimes generated in scripts.
    hamiltonians/  # Usually .npz or .hdf5 files of Hamiltonian(s). Sometimes generated in scripts.
    results/       # Usually .pkl of bitstrings sampled from hardware.
    experiment_name.ipynb  # Script to read in circuit and Hamiltonian, run on hardware, and save results.
```

## Table of experiments and results

| Experiment                   | Number of qubits | Number of terms in Hamiltonian | Number of two-qubit gates         | Algorithm(s) | Best accuracy on hardware                                                            |
|------------------------------|------------------|--------------------------------|-----------------------------------|--------------|--------------------------------------------------------------------------------------|
| Water                        | 14               | 1620                           | 144                               | Direct       | 50 mHa (Fez, ZNE)                                                                    |
| Water (UCJ)                  | 26               | 1620                           | 956                               | SQD          | <1 mHa                                                                               |
| One water phosphate product  | 44               |                                | 78 [owp_circuit.qasm]()           | SQD          | 13 mHa (Kingston, 50k shots, compared to CCSD(T) energy. Similar accuracy on Boston) |
| One water phosphate product  | 44               |                                | 1031 [owp_circuit_2.qasm]()       | SQD          |                                                                                      |
| One water phosphate product  | 44               |                                | 2325 [owp_circuit_3.qasm]()       | SQD          |                                                                                      |
| One water phosphate reactant | 44               | 575751                         | 5347 [po3_1w-22o_reactant.qasm]() | SQD          |                                                                                      |
| Two water phosphate reactant | 56               |                                | 3966 [po3_2w-28o_reactant.qasm]() | SQD          |                                                                                      |
| Two water phosphate product  | 56               |                                | 3914 [po3_2w-28o_product.qasm]()  | SQD          |                                                                                      |
