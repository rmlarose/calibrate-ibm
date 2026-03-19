I optimized geometry of the water monomer (h2o), dimer, and trimer at the wB97X-D/6-311+G**
level of theory using Gaussian 16. 

The water monomer lies in the Y-Z plane. I copied its coordinates and translated by 100 Angstroms
along the X-axis to create the 2h2o system. I added another water molecule another 100 Angstroms
away from the orginal to create the 3h2o system. The 2h2o and 3h2o systems were created in case
we wanted non-interacting water molecules to make the scaling arguments.

I created the FCIDUMP files in the STO-3G basis with freezing the core (1s on Oxygen) orbitals. 
The system sizes are given in the following table.

System      N_electrons     N_orbitals      N_qubits
h2o         8               6               12
2h2o        16              12              24
dimer       16              12              24
3h2o        24              18              36
trimer      24              18              36

I am including the FCI energies for all five systems in the file water-reference-energies.txt.
I also have the FCI wavefunctions if they can be used in any way for the scaling arguments.

For h2o, 2h2o, and dimer, I chose the circuit where the error with FCI drops below 1.59 mEh.
I included the fcidump.txt and summary.txt, which provides the ADAPT energy at that iteration, 
the FCI energy, and the error for that iteration. For the 3h2o and trimer system, I included
the final circuit that I have at this time even though they did not reach the 1.59 mEh 
threshold. We can ignore those for now if you do not think them useful. I can restart 
those calculations once NERSC is back online.





