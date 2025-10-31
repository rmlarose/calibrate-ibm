from collections import OrderedDict
from pathlib import Path
from typing import Dict
from contextlib import redirect_stdout

from compiler_workflows.bqskit_compile import bqskit_compile_all_to_all
from compiler_workflows.cirq_compile import cirq_compile_all_to_all
from compiler_workflows.pytket_compile import pytket_compile_all_to_all
from compiler_workflows.qiskit_compile import (
    qiskit_compile_all_to_all,
    qiskit_transpile_to_heavy_hex,
)


from bqskit import Circuit as BQSCircuit
from bqskit.ext import bqskit_to_qiskit

from pytket.extensions.qiskit import tk_to_qiskit
from pytket.qasm import (
    circuit_from_qasm as pytket_circuit_from_qasm,
    circuit_to_qasm as pytket_circuit_to_qasm,
)

from cirq.contrib.qasm_import import circuit_from_qasm as cirq_circuit_from_qasm
from cirq import qasm as cirq_qasm

from qiskit import QuantumCircuit
from qiskit.qasm2 import load, loads, dump

INPUT_FILENAME = "owp_circuit.qasm"
INPUT_DIRECTORY = "to_compile"
SAVE_OUTPUT_CIRCUITS = True

COMPILE_TO_HEAVY_HEX_USING_QISKIT = True

COMPILE_USING_BQSKIT = False
COMPILE_USING_PYTKET = False
COMPILE_USING_CIRQ = False
COMPILE_USING_QISKIT = True
QISKIT_DIRECT_TO_HEAVY_HEX = True

USE_FRAC_GATES = False

OUT_ALL_TO_ALL_DIRECTORY_PATH = "compiled/all_to_all"
OUT_HEAVY_HEX_DIRECTORY_PATH = "compiled/heavy_hex"

REDIRECT_PRINT_OUTPUT = False


def main():
    circuits = _compile_all_to_all()
    if COMPILE_TO_HEAVY_HEX_USING_QISKIT:
        _compile_heavy_hex(circuits=circuits)


def _compile_all_to_all():
    path = INPUT_DIRECTORY + "/" + INPUT_FILENAME
    circuits = dict()
    if COMPILE_USING_BQSKIT:
        bqscircuit = BQSCircuit.from_file(path)
        bqskit_circuit_all_to_all = bqskit_compile_all_to_all(bqscircuit=bqscircuit)
        SAVE_OUTPUT_CIRCUITS and bqskit_circuit_all_to_all.save(
            f"{OUT_ALL_TO_ALL_DIRECTORY_PATH}/bqskit/{INPUT_FILENAME}"
        )
        bqskit_circuit_in_qiskit = bqskit_to_qiskit(bqskit_circuit_all_to_all)
        circuits["BQSKIT"] = bqskit_circuit_in_qiskit

    if COMPILE_USING_PYTKET:
        pytket_circuit = pytket_circuit_from_qasm(path)
        pytket_circuit_all_to_all = pytket_compile_all_to_all(pytket_circuit)
        SAVE_OUTPUT_CIRCUITS and pytket_circuit_to_qasm(
            pytket_circuit_all_to_all,
            f"{OUT_ALL_TO_ALL_DIRECTORY_PATH}/pytket/{INPUT_FILENAME}",
        )
        pytket_circuit_in_qiskit = tk_to_qiskit(pytket_circuit_all_to_all)
        circuits["PYTKET"] = pytket_circuit_in_qiskit

    if COMPILE_USING_CIRQ:
        path_object = Path(path)
        cirq_circuit = cirq_circuit_from_qasm(path_object.read_text())
        cirq_circuit_all_to_all = cirq_compile_all_to_all(cirq_circuit)
        cirq_qasm_str = cirq_qasm(cirq_circuit_all_to_all)
        if SAVE_OUTPUT_CIRCUITS:
            with open(
                f"{OUT_ALL_TO_ALL_DIRECTORY_PATH}/cirq/{INPUT_FILENAME}", "w"
            ) as cirq_qasm_file:
                cirq_qasm_file.write(cirq_qasm_str)
        cirq_circuit_in_qiskit = loads(cirq_qasm_str)
        circuits["CIRQ"] = cirq_circuit_in_qiskit

    if COMPILE_USING_QISKIT:
        qiskit_circuit = load(path)
        qiskit_all_to_all = qiskit_compile_all_to_all(circuit=qiskit_circuit)
        SAVE_OUTPUT_CIRCUITS and dump(
            qiskit_all_to_all,
            Path(f"{OUT_ALL_TO_ALL_DIRECTORY_PATH}/qiskit/{INPUT_FILENAME}"),
        )
        circuits["QISKIT"] = qiskit_all_to_all

        if QISKIT_DIRECT_TO_HEAVY_HEX:
            qiskit_direct_circuit = qiskit_transpile_to_heavy_hex(
                all_to_all_circuit=qiskit_circuit, use_frac_gates=USE_FRAC_GATES
            )
            SAVE_OUTPUT_CIRCUITS and dump(
                qiskit_direct_circuit,
                Path(f"{OUT_HEAVY_HEX_DIRECTORY_PATH}/qiskit_direct/{INPUT_FILENAME}"),
            )
            _print_circuit_data(
                title="QISKIT_DIRECT" + "_HEAVY_HEX", circuit=qiskit_direct_circuit
            )

    for compiler, circuit in circuits.items():
        _print_circuit_data(title=compiler + "_ALL_TO_ALL", circuit=circuit)

    return circuits


def _compile_heavy_hex(circuits: Dict[str, QuantumCircuit]):
    transpiled = dict()

    for compiler, circuit in circuits.items():
        heavy_hex_transpiled = qiskit_transpile_to_heavy_hex(
            all_to_all_circuit=circuit,
            use_frac_gates=USE_FRAC_GATES,
        )
        transpiled[compiler] = heavy_hex_transpiled
        _print_circuit_data(title=compiler + "_HEAVY_HEX", circuit=heavy_hex_transpiled)
        dump(
            heavy_hex_transpiled,
            Path(f"{OUT_HEAVY_HEX_DIRECTORY_PATH}/{compiler.lower()}/{INPUT_FILENAME}"),
        )
    return transpiled


def _print_circuit_data(title: str, circuit: QuantumCircuit):
    ops: OrderedDict[str, int] = circuit.count_ops()
    two = sum([ops.get("cx", 0), ops.get("cz", 0), ops.get("rzz", 0)])
    one = int(sum(ops.values()) - two)
    lines = []
    lines.append("================")
    lines.append(title)
    lines.append(f"operation counts: {str(ops)}")
    lines.append(f"depth = {circuit.depth()}")
    lines.append(f"two-qubit gate count={two}")
    lines.append(f"one-qubit gate count={one}")
    lines.append("================")
    print("\n".join(lines))


if __name__ == "__main__":
    if REDIRECT_PRINT_OUTPUT:
        with open(
            f"most_recent_output/{INPUT_FILENAME[0:-5]}", "w"
        ) as f, redirect_stdout(f):
            main()
    else:
        main()
