import openfermion as of
# from openfermionpyscf import run_pyscf

from sqd_functions import run_sqd


# geometry = of.chem.geometry_from_pubchem("water")
# basis = "sto-3g"
# multiplicity = 1
# charge = 0
# ham = of.chem.MolecularData(geometry, basis, multiplicity, charge)
# mol = run_pyscf(ham, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)
# mol.save()
# ham = of.chem.MolecularData(filename=mol.filename)
# h1 = ham.one_body_integrals
# h2 = ham.two_body_integrals
# n_orbitals = h1.shape[0]
# num_electrons = 10
# ecore = 0.0

mol = of.chem.MolecularData(filename="../monomer_eqb.hdf5")
ecore = 0.0


for experiment_iter in [1,2,3,4,5,6,7,8,9,10]:

    for samples_per_batch in [1, 10, 100, 1000, 10**4]:

        result = run_sqd(
            one_body_integrals=mol.one_body_integrals,
            two_body_integrals=mol.two_body_integrals,
            n_orbitals=mol.n_orbitals,
            num_electrons=(mol.get_n_alpha_electrons(), mol.get_n_beta_electrons()),
            spin_sq=0.0,
            nuclear_repulsion_energy=mol.nuclear_repulsion,
            e_core=ecore,
            num_batches=5,
            samples_per_batch=samples_per_batch,
            max_config_recovery_iterations=5,
            max_davidson_cycles=10**4,
            symmetrize_spin=True,
            ansatz_circuit=None,
            sampler=None,
            force_fci=False, # set this to True to skip circuit sampling and just use all bitstrings in FCI space.
            bitstrings_file="./water_mar.pkl",
            save_dir=f"./water_sqd/experiment_{experiment_iter}/",
        )

        print(f'Result for {samples_per_batch} samples per batch:')
        print(result.energy)
        print(f'total energy: {result.energy + ecore} Hartree')
