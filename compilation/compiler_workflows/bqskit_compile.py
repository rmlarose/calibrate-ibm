from bqskit import Circuit
from bqskit.passes import (
    ToVariablePass,
    QuickPartitioner,
    ForEachBlockPass,
    ScanningGateRemovalPass,
    UnfoldPass,
    ToU3Pass,
    CompressPass,
)
from bqskit.compiler import Compiler


def bqskit_compile_all_to_all(bqscircuit: Circuit):
    instantiate_options = {
        "multistarts": 8,
        "method": "qfactor"  # qfactor requires a specific input gateset
    }

    passes = [
        ToVariablePass(),
        QuickPartitioner(block_size=4),
        ToU3Pass(convert_all_single_qubit_gates=True),
        ForEachBlockPass(
            [
                ScanningGateRemovalPass(
                    instantiate_options=instantiate_options,
                    # start_from_left=True,
                    # success_threshold=1e-8,
                    # collection_filter=None,
                ),
            ]
        ),
        UnfoldPass(),
        ToU3Pass(convert_all_single_qubit_gates=True),
        # CompressPass(),
    ]
    with Compiler() as comp:
        compiled_bqscircuit = comp.compile(bqscircuit, passes)
    return compiled_bqscircuit
