from cirq import (
    Circuit,
    eject_phased_paulis,
    drop_negligible_operations,
    drop_empty_moments,
    synchronize_terminal_measurements,
)
from cirq import (
    expand_composite,
    optimize_for_target_gateset,
    CZTargetGateset,
    TransformerContext,
    eject_z,
)


def cirq_compile_all_to_all(cirqcircuit: Circuit):
    gateset = CZTargetGateset()
    compiled_circuit = optimize_for_target_gateset(circuit=cirqcircuit, gateset=gateset)
    return compiled_circuit
