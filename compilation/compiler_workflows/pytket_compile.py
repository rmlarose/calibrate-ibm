from pytket import Circuit
from pytket.extensions.qiskit import AerBackend


def pytket_compile_all_to_all(pytketcircuit: Circuit):
    backend = AerBackend()
    compiled_pytketcircuit = backend.get_compiled_circuit(
        pytketcircuit, optimisation_level=3
    )
    return compiled_pytketcircuit
