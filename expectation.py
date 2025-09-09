import cirq


def _compute_expectation(
    pauli: cirq.PauliString,
    counts: dict[str, int],
    little_endian: bool = True
) -> float:
    if pauli is cirq.PauliString():
        return pauli.coefficient

    expectation = 0.0

    indices = [q.x for q in pauli.qubits]
    for key, value in counts.items():
        if little_endian:
            key = list(map(int, list(key)))
        else:
            key = list(map(int, list(key[::-1])))
        expectation += (-1) ** sum([key[i] for i in indices]) * value

    return pauli.coefficient * expectation / sum(counts.values())


def compute_expectation(
    pauli_sum: cirq.PauliSum,
    counts: dict[str, int],
    little_endian: bool = True
) -> float:
    expval = 0.0
    for pauli in pauli_sum:
        expval += _compute_expectation(pauli, counts, little_endian)
    return expval