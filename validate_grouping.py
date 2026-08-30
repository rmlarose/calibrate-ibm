#!/usr/bin/env python3

import sys
import pickle
from openfermion import qubit_operator_to_pauli_sum
import cirq


def validate_groups(groups):
    for group in groups:
        cirq_terms = []
        for term in group:
            pauli_sum = qubit_operator_to_pauli_sum(term)
            cirq_term = list(pauli_sum)[0]
            cirq_terms.append(cirq_term)

        for i, term_i in enumerate(cirq_terms):
            for term_j in cirq_terms[i+1:]:
                assert cirq.commutes(term_i, term_j)


if __name__ == "__main__":
    with open(sys.argv[1], 'rb') as f:
        groups = pickle.load(f)

    validate_groups(groups)
    print(f"All {len(groups)} groups validated successfully")
