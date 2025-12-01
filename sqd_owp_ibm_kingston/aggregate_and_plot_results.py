import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# will need qiskit-addon-sqd 0.12.0
from qiskit_addon_sqd.fermion import SCIResult


integrals_file = os.path.expanduser(f'./owp_reactant.npz')

integrals_data = np.load(integrals_file, allow_pickle=True)

ecore = integrals_data['ECORE']

subspace_dimensions = []
energies = []

for experiment_num in [1,2,3,4,5,6,7,8,9,10]:

    for samples_per_batch in [1,10,100,1000,10000]:

        final_results_file = os.path.expanduser(f'./saved_results_owp_ibm_kingston/experiment_{experiment_num}/sqd_final_results_{samples_per_batch}_samples_per_batch.pkl')

        final_results = pickle.load(open(final_results_file, 'rb'))

        print(f'\nFinal Results for experiment {experiment_num}, {samples_per_batch} samples per batch:')
        print(f'Energy = {final_results.energy + ecore}, SCI subspace dimension = {np.prod(final_results.sci_state.amplitudes.shape)}')
        subspace_dimensions.append(np.prod(final_results.sci_state.amplitudes.shape))
        energies.append(final_results.energy + ecore)

np.savez(os.path.expanduser('./saved_results_owp_ibm_kingston/sqd_owp_ibm_kingston_energies_and_sqd_subspace_dims.npz'), energies=energies, subspace_dimensions=subspace_dimensions)

print(f'Best energy result: {min(energies)}')
# get index of best energy
best_index = energies.index(min(energies))
print(f'Corresponding subspace dimension: {subspace_dimensions[best_index]}')

# increase size of plot
plt.figure(figsize=(10, 6))
plt.scatter(subspace_dimensions, energies)
plt.xscale('log')
plt.xlabel('SCI Subspace Dimension')
plt.ylabel('Energy (Hartree)')
plt.title('SQD energies on IBM Kingston')
plt.savefig(os.path.expanduser('./saved_results_owp_ibm_kingston/sqd_owp_ibm_kingston_energies.png'))
plt.show()