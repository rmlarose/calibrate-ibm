from qiskit import transpile, QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService

COMPILE_OPT_LEVEL = 1
TRANSPILE_OPT_LEVEL = 3

service = QiskitRuntimeService()


def qiskit_compile_all_to_all(circuit: QuantumCircuit):
    return transpile(
        circuit,
        basis_gates=["u3", "cx", "cz"],
        seed_transpiler=42,
        optimization_level=COMPILE_OPT_LEVEL,
        # callback=_compile_progress,
        # layout_method="sabre",              # layout: ['default', 'dense', 'sabre', 'trivial']
        # routing_method="sabre",             # routing: ['basic', 'default', 'lookahead', 'none', 'sabre']
        # translation_method="default",       # translation: ['ibm_backend', 'ibm_dynamic_circuits', 'ibm_fractional', 'default', 'synthesis', 'translator']
        # optimization_method="default",      # optimization: ['default']
        # scheduling_method="default",        # scheduling: ['alap', 'asap', 'default']
        # unitary_synthesis_method="default", # synthesis: ['aqc', 'clifford', 'default', 'sk']
        # approximation_degree=1,             # 1 (exact) to 0 (max approximation)
    )


def qiskit_transpile_to_heavy_hex(
    all_to_all_circuit: QuantumCircuit, use_frac_gates: bool
):
    heavy_hex_backend = service.backend("ibm_fez", use_fractional_gates=use_frac_gates)
    return transpile(
        all_to_all_circuit,
        backend=heavy_hex_backend,
        seed_transpiler=42,
        optimization_level=TRANSPILE_OPT_LEVEL,
        # callback=_compile_progress,
        # layout_method="sabre",              # layout: ['default', 'dense', 'sabre', 'trivial']
        # routing_method="sabre",             # routing: ['basic', 'default', 'lookahead', 'none', 'sabre']
        # translation_method="default",       # translation: ['ibm_backend', 'ibm_dynamic_circuits', 'ibm_fractional', 'default', 'synthesis', 'translator']
        # optimization_method="default",      # optimization: ['default']
        # scheduling_method="default",        # scheduling: ['alap', 'asap', 'default']
        # unitary_synthesis_method="default", # synthesis: ['aqc', 'clifford', 'default', 'sk']
        # approximation_degree=1,             # 1 (exact) to 0 (max approximation)
    )


def _compile_progress(**kwargs):
    pass_object = kwargs["pass_"]
    time_spent_in_pass = kwargs["time"]
    pass_index = kwargs["count"]
    dag = kwargs["dag"]
    print(
        f"[{pass_index:03d}] {pass_object.name():38s}  {time_spent_in_pass:6.2f}s  nodes={dag.count_ops()}"
    )
