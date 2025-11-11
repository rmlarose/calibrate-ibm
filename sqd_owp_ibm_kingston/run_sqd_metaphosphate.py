import pyscf
from pyscf import scf, cc, gto, ao2mo

import numpy as np
from functools import partial
import os
from time import perf_counter

from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import BitArray

from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer import AerSimulator

import ffsim
import pickle

from sqd_functions import get_lucj_ansatz, run_sqd


file = os.path.expanduser(f'./owp_reactant.npz')

integrals_data = np.load(file, allow_pickle=True)


h1 = integrals_data['H1']
h2 = integrals_data['H2']
n_orbitals = integrals_data['NORB']
num_electrons = integrals_data['NELEC']
ecore = integrals_data['ECORE']


for experiment_iter in [1,2,3,4,5,6,7,8,9,10]:

    for samples_per_batch in [1, 10, 100, 1000, 10**4]:

        result = run_sqd(
            one_body_integrals=h1,
            two_body_integrals=h2,
            n_orbitals=n_orbitals,
            num_electrons=(num_electrons // 2, num_electrons // 2),
            spin_sq=0.0,
            nuclear_repulsion_energy=0.0,
            e_core=ecore,
            num_batches=5,
            samples_per_batch=samples_per_batch,
            max_config_recovery_iterations=5,
            max_davidson_cycles=10**4,
            symmetrize_spin=True,
            ansatz_circuit=None,
            sampler=None,
            force_fci=False, # set this to True to skip circuit sampling and just use all bitstrings in FCI space.
            bitstrings_file="./owp_counts_ibm_kingston.pkl",
            save_dir=f"./saved_results_owp_ibm_kingston/experiment_{experiment_iter}/",
        )

        print(f'Result for {samples_per_batch} samples per batch:')
        print(result.energy)
        print(f'total energy: {result.energy + ecore} Hartree')
