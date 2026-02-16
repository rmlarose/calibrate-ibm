"""Like isqd.ipynb, but made friendly to batch processing."""

import argparse

import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif"})
import numpy as np
import pickle
import glob

import pyscf.tools
from pyscf import ao2mo

import collections
from functools import partial
import os
import pickle

from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch

def transform_bitstring(bits):
    """
    Convert a given bitstring from Openfermion convention 
    (alternating alpha/beta, big endian) to Qiskit (all alpha
    then all beta, little endian).
    """

    left = [bits[i] for i in range(len(bits)) if i % 2 == 1]   # beta
    right = [bits[i] for i in range(len(bits)) if i % 2 == 0]  # alpha

    # Reverse each half
    left.reverse()
    right.reverse()

    # Concatenate
    return ''.join(left + right)

# fragment = "atp_0_be2_f4"
# circuit_dir = "circuits"
# hamiltonian_dir = "hamiltonians"
# results_dir = "results"
# all_adapt_iterations = [1, 2, 3]

parser = argparse.ArgumentParser()
parser.add_argument("--fragment", type=str, default="atp_0_be2_f4")
parser.add_argument("--circuit_dir", type=str, default="circuits")
parser.add_argument("--hamiltonian_dir", type=str, default="hamiltonians")
parser.add_argument("--results_dir", type=str, default="results")
parser.add_argument("all_adapt_iterations", type=int, nargs='*')
args = parser.parse_args()

fragment = args.fragment
hamiltonian_dir = args.hamiltonian_dir
circuit_dir = args.circuit_dir
results_dir = args.results_dir
all_adapt_iterations = args.all_adapt_iterations

fcidump = pyscf.tools.fcidump.read(f"{hamiltonian_dir}/{fragment}.fcidump")

n_orbitals = fcidump.get("NORB")
num_electrons = fcidump.get("NELEC")
ecore = fcidump.get("ECORE")
h1 = fcidump.get("H1")
h2 = fcidump.get("H2")
h2 = ao2mo.restore(1, h2, n_orbitals)

nqubits = 2 * n_orbitals

all_counts = []
fnames = []

for adapt_iterations in all_adapt_iterations:
    fname = glob.glob(f"{results_dir}/{fragment}/*{adapt_iterations:03d}*.qasm*")[0]
    counts = pickle.load(
        open(f"{fname}", "rb")
    )
    mode_order = pickle.load(
        open(f"{circuit_dir}/{fragment}/{fragment}_mode_order_{adapt_iterations:03d}_adaptiterations.pkl", "rb")
    )
    qubit_order = pickle.load(
        open(f"{circuit_dir}/{fragment}/{fragment}_qubit_order_{adapt_iterations:03d}_adaptiterations.pkl", "rb")
    )

    measurement_outcomes = counts
    permuted_outcomes = {}
    for original_bitstring in measurement_outcomes.keys():
        qubit_permuted_bitstring = "".join([original_bitstring[qubit_order.index(n)] for n in range(nqubits)])
        mode_permuted_bitstring = "".join([qubit_permuted_bitstring[mode_order.index(n)] for n in range(nqubits)])

        final_permuted_bitstring = transform_bitstring(mode_permuted_bitstring)
        permuted_outcomes[final_permuted_bitstring[::]] = measurement_outcomes[original_bitstring]
    
    counts = permuted_outcomes
    all_counts.append(counts)

    print("ADAPT iteration", adapt_iterations)
    max_key = max(counts, key=counts.get)
    print(f'Most common bitstring: {max_key} with count {counts[max_key]}')
    print(f'Total number of bitstrings: {len(counts)}')
    print(f"Total number of samples:", sum(counts.values()))

# TODO: Implement strategies to cap the number of shots when concatenating.
counts = collections.Counter()
for c in all_counts:
    for bitstring, count in c.items():
        counts[bitstring] += count
bit_array = BitArray.from_counts(counts)

energy_tol = 1e-8
occupancies_tol = 1e-8
carryover_threshold = 1e-5

sci_solver = partial(solve_sci_batch, spin_sq=0, max_cycle=10000)
result_history = []

def callback(results: list[SCIResult]):
    result_history.append(results)
    iteration = len(result_history)
    print(f"Iteration {iteration}")
    for i, result in enumerate(results):
        print(f"\tSubsample {i}")
        print(f"\t\tEnergy: {result.energy + ecore}")
        print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")


result = diagonalize_fermionic_hamiltonian(
    one_body_tensor=h1,
    two_body_tensor=h2,
    bit_array=bit_array,
    samples_per_batch=500,
    norb=n_orbitals,
    nelec=(num_electrons // 2, num_electrons // 2),
    num_batches=2,
    energy_tol=energy_tol,
    occupancies_tol=occupancies_tol,
    max_iterations=100,
    sci_solver=sci_solver,
    symmetrize_spin=True,
    carryover_threshold=carryover_threshold,
    callback=callback,
)
min_e = [
    min(result, key=lambda res: res.energy).energy + ecore
    for result in result_history
]
min_e

iterations_key = "_".join(map(str, all_adapt_iterations))
save_name = f"sqd_energies_{iterations_key}_adaptiterations.txt"
np.savetxt(save_name, min_e)