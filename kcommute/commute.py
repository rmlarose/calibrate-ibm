#!/usr/bin/env python3

"""Defines k-qubit wise commuting."""

import math
from typing import Iterable, List

import cirq


def restrict_to(
    pauli: cirq.PauliString, qubits: Iterable[cirq.Qid]
) -> cirq.PauliString:
    return cirq.PauliString(p.on(q) for q, p in pauli.items() if q in qubits)


def compute_blocks(qubits: Iterable[cirq.Qid], k: int) -> List[List[cirq.Qid]]:
    return [qubits[k * i : k * (i + 1)] for i in range(math.ceil(len(qubits) / k))]


def commutes(pauli1: cirq.PauliString, pauli2: cirq.PauliString, blocks: List[List[cirq.Qid]]) -> bool:
    """Returns True if pauli1 k-commutes with pauli2, else False.

    Args:
        pauli1: A Pauli string.
        pauli2: A Pauli string.
        blocks: The block partitioning.
    """
    for block in blocks:
        if not cirq.commutes(restrict_to(pauli1, block), restrict_to(pauli2, block)):
            return False
    return True


def pauli_to_symplectic(term):
    """Convert a QubitOperator term to symplectic (x_bits, z_bits) representation.

    Each Pauli operator is encoded as two bits:
        I → (0,0)
        X → (1,0)
        Y → (1,1)
        Z → (0,1)

    Args:
        term: A QubitOperator with a single term

    Returns:
        Tuple of (x_bits, z_bits) where each is an integer bitmap
    """
    x_bits = 0
    z_bits = 0
    pauli_string = list(term.terms.keys())[0]
    for qubit_idx, pauli_op in pauli_string:
        if pauli_op == 'X':
            x_bits |= (1 << qubit_idx)
        elif pauli_op == 'Y':
            x_bits |= (1 << qubit_idx)
            z_bits |= (1 << qubit_idx)
        elif pauli_op == 'Z':
            z_bits |= (1 << qubit_idx)
    return x_bits, z_bits


def commutes_symplectic(x1: int, z1: int, x2: int, z2: int, blocks: List[List[int]]) -> bool:
    """Check if two Pauli strings commute using symplectic representation.

    Uses the symplectic inner product: (x1 · z2 + z1 · x2) mod 2
    If this is 0, the Paulis commute; if 1, they anticommute.

    For k-commuting, checks that the operators commute within EACH block independently.

    Args:
        x1: X-component bitmask of first Pauli
        z1: Z-component bitmask of first Pauli
        x2: X-component bitmask of second Pauli
        z2: Z-component bitmask of second Pauli
        blocks: List of qubit index lists defining the block structure

    Returns:
        True if the Paulis commute within all blocks, False otherwise
    """
    # Check each block independently
    for block in blocks:
        # Create mask for this block only
        block_mask = 0
        for qubit_idx in block:
            block_mask |= (1 << qubit_idx)

        # Check commutation within this block
        inner_product = ((x1 & z2) ^ (z1 & x2)) & block_mask
        if bin(inner_product).count('1') % 2 != 0:
            # Anticommute in this block
            return False

    # Commute in all blocks
    return True
