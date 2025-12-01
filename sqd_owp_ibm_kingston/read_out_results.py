import numpy as np
import pickle
import os

# will need qiskit-addon-sqd 0.12.0
from qiskit_addon_sqd.fermion import SCIResult


experiment_num = 1
samples_per_batch = 1000

results_history_file = os.path.expanduser(f'./saved_results_owp_ibm_kingston/experiment_{experiment_num}/sqd_results_history_{samples_per_batch}_samples_per_batch.pkl')
final_results_file = os.path.expanduser(f'./saved_results_owp_ibm_kingston/experiment_{experiment_num}/sqd_final_results_{samples_per_batch}_samples_per_batch.pkl')

results_history = pickle.load(open(results_history_file, 'rb'))
final_results = pickle.load(open(final_results_file, 'rb'))


integrals_file = os.path.expanduser(f'./owp_reactant.npz')

integrals_data = np.load(integrals_file, allow_pickle=True)

ecore = integrals_data['ECORE']

print('Results History:')
for config_recovery_iter, list_of_sci_results in enumerate(results_history):
    print(f'Config Recovery Iteration {config_recovery_iter + 1}:')
    for batch_num, sci_result in enumerate(list_of_sci_results):
        assert isinstance(sci_result)
        print(f'  Batch {batch_num + 1}: Energy = {sci_result.energy + ecore}, SCI subspace dimension = {np.prod(sci_result.sci_state.amplitudes.shape)}')

print('\nFinal Results:')
print(f'Energy = {final_results.energy + ecore}, SCI subspace dimension = {np.prod(final_results.sci_state.amplitudes.shape)}')
