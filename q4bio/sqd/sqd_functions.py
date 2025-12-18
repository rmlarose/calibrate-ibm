import numpy as np
from functools import partial
import os
from time import perf_counter

from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import BitArray

import pickle
import sys


def run_sqd(one_body_integrals: np.ndarray,
            two_body_integrals: np.ndarray,
            n_orbitals: int,
            num_electrons: tuple[int,int],
            spin_sq: float,
            nuclear_repulsion_energy: float,
            e_core: float,
            num_batches: int,
            samples_per_batch: int,
            max_config_recovery_iterations: int,
            max_davidson_cycles: int,
            symmetrize_spin: bool,
            ansatz_circuit: QuantumCircuit,
            sampler,
            force_fci: bool = False,
            bitstrings_file: str = None,
            save_dir: str = None,
            ) -> SCIResult:
    

    """
    
    one_body_integrals: One-body integrals (h1) as a numpy array.
    two_body_integrals: Two-body integrals (h2) as a numpy array.
    n_orbitals: Total number of spatial orbitals in the system.
    num_electrons: Tuple of alpha and beta electron numbers.
    spin_sq: Desired total spin squared value for the system.
    nuclear_repulsion_energy: Nuclear repulsion energy of the molecule.
    num_batches: Number of batches of bitstrings passed to SQD.
    samples_per_batch: Number of bitstrings per batch.
    max_config_recovery_iterations: Maximum number of configuration recovery iterations in SQD.
    max_davidson_cycles: Maximum number of Davidson cycles for the SCI solver.
    symmetrize_spin: Boolean flag indicating whether to symmetrize spin in SQD.
    ansatz_circuit: QuantumCircuit representing the ansatz used to generate bitstrings.
    sampler: Qiskit sampler primitive for executing the ansatz circuit.
    force_fci: Boolean flag to force FCI calculation instead of using sampled bitstrings.
    bitstrings_file: Optional file path to load pre-sampled bitstrings from.
    """

    # redirect output to file

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # log_file = open(os.path.join(save_dir, f"sqd_samples_per_batch_{samples_per_batch}_log.txt"), "w")
        # sys.stdout = log_file
        # sys.stderr = log_file

    #### hardcoded SQD parameters ####
    energy_tol = 1e-8
    occupancies_tol = 1e-8
    carryover_threshold = 1e-5

    #################################

    print(f'Number of orbitals: {n_orbitals}, Number of electrons: {num_electrons[0] + num_electrons[1]}')
    print(f'e_core: {e_core}')
    print(f'energy tolerance: {energy_tol} occupancies tolerance: {occupancies_tol}')
    print(f'carryover threshold: {carryover_threshold}')

    if bitstrings_file is not None:
        
        print(f'Loading bitstrings from file: {bitstrings_file}')
        bitstring_counts = pickle.load(open(bitstrings_file, 'rb'))
        bitstring_counts_rearranged = {}

        for bitstr in bitstring_counts:

            # need to convert from alpha, beta, alpha, beta, ... to alpha, alpha, ..., beta, beta, ...
            alpha_bits = [bitstr[int(2*n)] for n in range(n_orbitals)]
            beta_bits = [bitstr[int(2*n)+1] for n in range(n_orbitals)]

            # reverse bits to keep with the convention that qiskit-addon-sqd expects.
            rearranged_bitstr = ''.join(alpha_bits + beta_bits)#[::-1]
            bitstring_counts_rearranged[rearranged_bitstr] = bitstring_counts[bitstr]

        bit_array = BitArray.from_counts(bitstring_counts_rearranged)
        counts = bit_array.get_counts()
        max_key = max(counts, key=counts.get)
        print(f'Most common bitstring: {max_key} with count {counts[max_key]}')
        print(f'Total number of bitstrings: {len(counts)}')
        print(f"Total number of samples:", sum(bitstring_counts.values()))

    else:

        if force_fci == True:
            
            print("Warning: Forcing FCI calculation. Make sure you are using this for testing purposes only.")
            def bitstring_to_bool(bitstring):

                return [bit == '1' for bit in bitstring]
            

            alpha_bin_list = [bin(m)[2:].zfill(n_orbitals) for m in range(2**n_orbitals) if bin(m).count("1") == num_electrons[0]]
            beta_bin_list = [bin(m)[2:].zfill(n_orbitals) for m in range(2**n_orbitals) if bin(m).count("1") == num_electrons[1]]

            batch = np.asarray([bitstring_to_bool(b+a) for a in alpha_bin_list for b in beta_bin_list])
            bit_array = BitArray.from_bool_array(batch)
            samples_per_batch = len(batch)
    

        else:

            start_time = perf_counter()
            job = sampler.run([ansatz_circuit])
            job_result = job.result()[0]
            bit_array = job_result.data.meas
            stop_time = perf_counter()
            print(f'Circuit runtime: {stop_time - start_time} seconds.')
            print(f"Job ID: {job.job_id()}")

    sci_solver = partial(solve_sci_batch, spin_sq=spin_sq, max_cycle=max_davidson_cycles)

    result_history = []

    def callback(results: list[SCIResult]):
        result_history.append(results)
        iteration = len(result_history)
        print(f"Iteration {iteration}")
        for i, result in enumerate(results):
            print(f"\tSubsample {i}")
            print(f"\t\tEnergy: {result.energy + nuclear_repulsion_energy + e_core}")
            print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")

    result = diagonalize_fermionic_hamiltonian(
        one_body_tensor=one_body_integrals,
        two_body_tensor=two_body_integrals,
        bit_array=bit_array,
        samples_per_batch=samples_per_batch,
        norb=n_orbitals,
        nelec=num_electrons,
        num_batches=num_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_config_recovery_iterations,
        sci_solver=sci_solver,
        symmetrize_spin=symmetrize_spin,
        carryover_threshold=carryover_threshold,
        callback=callback
        )
    
    # save result to specified directory
    if save_dir is not None:

        results_history_save_path = os.path.join(save_dir, f'sqd_results_history_{samples_per_batch}_samples_per_batch.pkl')
        with open(results_history_save_path, 'wb') as f:
            pickle.dump(result_history, f)
        print(f'Saved SQD results history to {results_history_save_path}')

        final_results_save_path = os.path.join(save_dir, f'sqd_final_results_{samples_per_batch}_samples_per_batch.pkl')
        with open(final_results_save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f'Saved SQD final result to {final_results_save_path}')


    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    # log_file.close()
    return result
